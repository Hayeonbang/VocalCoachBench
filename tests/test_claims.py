"""Behavioral coverage for canonical Claims replay; no inference is performed."""
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vocalcoachbench.claims import (BUDGETS, digest, evaluate_claims, normalize_claim_list,
                                    prepare_claims_jobs, validate_judgments, sha256,
                                    _validate_sample_response)
from vocalcoachbench.claims_judge import judge_request, run_claim_judge, _validate_job_population
from vocalcoachbench.io import read_jsonl, write_jsonl


class ClaimRulesTests(unittest.TestCase):
    def test_normalizer_only_repairs_local_ids_and_limits_slots(self):
        raw = [{"id": "d1", "text": "a", "category": "PITCH"},
               {"id": "d1", "text": "b", "category": "BREATH"},
               {"text": "c", "target_diagnosis_id": "unchanged"},
               None, {"text": ""}, {"id": "d4", "text": "d"},
               {"id": "d5", "text": "e"}, {"id": "d6", "text": "overflow"}]
        original = copy.deepcopy(raw)
        kept, audit = normalize_claim_list(raw, "diagnosis")
        self.assertEqual(raw, original)
        self.assertEqual([r["text"] for r in kept], ["a", "b", "c", "d", "e"])
        self.assertEqual(len({r["id"] for r in kept}), 5)
        self.assertEqual(kept[2]["target_diagnosis_id"], "unchanged")
        self.assertEqual(audit["id_repairs"], 2)
        self.assertEqual(audit["overflow_dropped"], 1)

    def test_diagnosis_judge_requires_all_ids_and_real_matches(self):
        job = {"phase": "diagnosis", "gt": [{"id": "g1"}, {"id": "g2"}], "pred": [{"id": "d1"}]}
        invalid = {"judgments": [{"gt_id": "g1", "label": "STRICT_HIT", "matched_model_claim_ids": []},
                                  {"gt_id": "g1", "label": "MISS", "matched_model_claim_ids": ["invented"]}]}
        self.assertEqual(set(validate_judgments(invalid, job)),
                         {"duplicate_ids", "missing_or_extra_ids", "hit_without_match", "invalid_matched_ids"})

    def test_prescription_judge_rejects_duplicate_or_missing_slots(self):
        job = {"phase": "correction", "pred": [{"id": "p1"}, {"id": "p2"}]}
        obj = {"judgments": [{"prescription_id": "p1", "label": "VALID"}] * 2}
        self.assertEqual(set(validate_judgments(obj, job)), {"duplicate_ids", "missing_or_extra_ids"})

    def test_judge_settings_preserve_asymmetric_frozen_policy(self):
        dx = judge_request({"phase": "diagnosis", "messages": []}, "explicit-judge", 2048)
        rx = judge_request({"phase": "correction", "messages": []}, "explicit-judge", 4096)
        self.assertEqual(dx["temperature"], 0)
        self.assertEqual(dx["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", rx)
        self.assertEqual(rx["max_completion_tokens"], 4096)

    def test_simple_extraction_target_cannot_change_with_same_claim_text(self):
        parsed = {"diagnosis_claims": [{"id": "d1", "text": "First issue"}, {"id": "d2", "text": "Second issue"}],
                  "prescription_claims": [{"id": "p1", "text": "Practice", "target_diagnosis_id": "d1"}]}
        sample = {"parsed": copy.deepcopy(parsed), "simple_llm_normalizer": {"raw_text": json.dumps(parsed)}}
        _validate_sample_response(sample, "test")
        sample["parsed"]["prescription_claims"][0]["target_diagnosis_id"] = "d2"
        with self.assertRaisesRegex(ValueError, "prescription targets changed"):
            _validate_sample_response(sample, "test")


class CanonicalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.release = cls.root / "release"
        ann = cls.release / "annotations"
        meta = cls.release / "metadata"
        ann.mkdir(parents=True)
        meta.mkdir()
        (meta / "CANONICAL_EVALUATION.md").write_text("515 recordings; strict_all_attempts\n")
        (meta / "top3_join_policy.json").write_text(json.dumps({"status": "canonical", "recording_inventory_n": 515,
            "top3_annotation_n": 1064, "top3_recording_n": 515, "dataset_recording_counts": {"A": 207, "B": 308},
            "join_key": "physical audio SHA-256"}))
        inventory, blocks, atomic, top3 = [], [], [], []
        for dataset, total in (("A", 207), ("B", 308)):
            for index in range(total):
                rid = f"{dataset}_{index:03d}"
                block = rid + "_block"
                inventory.append({"recording_id": rid, "dataset": dataset})
                blocks.append({"block_uid": block, "recording_id": rid, "annotator_id": "expert",
                               "block_index": 1, "categories": ["PITCH"]})
                count = (10 + (index < 20)) if dataset == "A" else (13 + (index < 68))
                for k in range(count):
                    atomic.append({"claim_uid": f"{block}_c{k}", "block_uid": block, "recording_id": rid,
                                   "annotator_id": "expert", "local_claim_id": f"c{k}", "claim_type": "issue",
                                   "text": f"Expert issue {k}."})
                top3.extend([{"recording_id": rid, "top3_issues": ["PITCH", "BREATH", "PHONATION"]}] * 2)
        top3.extend(copy.deepcopy(top3[:34]))
        for name, rows in (("recordings", inventory), ("feedback_blocks", blocks), ("atomic_claims", atomic),
                           ("top3_issue_annotations", top3)):
            write_jsonl(ann / f"{name}.jsonl", rows)
        # Synthetic reference bytes must never pass the production frozen pins.
        # The test fixture explicitly supplies its own expected hashes.
        cls.pin_patch = patch("vocalcoachbench.claims.ANNOTATION_HASHES",
                              {p.name: sha256(p) for p in ann.glob("*.jsonl")})
        cls.pin_patch.start()
        identities = sorted((row["recording_id"][2:], row["recording_id"]) for row in inventory)
        identity_hash = hashlib.sha256(json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        cls.inventory_patch = patch("vocalcoachbench.claims_judge.INVENTORY_ID_SHA256", identity_hash)
        cls.inventory_patch.start()
        prediction = {"diagnosis_claims": [{"id": "d1", "text": "Pitch is unstable."},
                                            {"id": "d2", "text": "Timing is unstable."}],
                      "prescription_claims": [{"id": "p1", "text": "Practice pitch.", "target_diagnosis_id": "d1"},
                                              {"id": "p2", "text": "Practice timing.", "target_diagnosis_id": "d2"}]}
        cls.predictions = cls.root / "predictions.jsonl"
        write_jsonl(cls.predictions, [{"recording_id": "A_000", "response_text": json.dumps(prediction)}])
        cls.output = cls.root / "prepared"
        cls.manifest = prepare_claims_jobs(cls.predictions, cls.release, cls.output, model="test")
        jobs = read_jsonl(cls.output / "jobs.jsonl")
        outcomes = []
        for job in jobs:
            diagnosis = job["phase"] == "diagnosis"
            judgments = ([{"gt_id": g["id"], "label": "MISS", "matched_model_claim_ids": []} for g in job["gt"]]
                         if diagnosis else [{"prescription_id": p["id"], "label": "VALID"} for p in job["pred"]])
            outcome = {"id": job["id"], "job_sha256": digest(job), "valid": True,
                       "judgments": judgments, "attempts": []}
            if job["pred"]:
                if diagnosis:
                    judgments[0].update(label="STRICT_HIT", matched_model_claim_ids=["d1"])
                    judgments[1].update(label="COARSE_HIT", matched_model_claim_ids=["d1"])
                request = judge_request(job, "frozen-test-judge", BUDGETS[job["phase"]][0])
                parsed = {"judgments": judgments}
                outcome["attempts"] = [{"request": request, "request_sha256": digest(request),
                    "parsed": parsed, "validation_errors": [],
                    "response": {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(parsed)}}]}}]
            else:
                outcome["no_model_claims"] = True
            outcomes.append(outcome)
        cls.outcomes = cls.root / "outcomes.jsonl"
        write_jsonl(cls.outcomes, outcomes)

    @classmethod
    def tearDownClass(cls):
        cls.inventory_patch.stop()
        cls.pin_patch.stop()
        cls.temp.cleanup()

    def evaluate(self, outcomes=None, samples=None):
        return evaluate_claims(samples or self.output / "test_samples.jsonl", self.output / "jobs.jsonl",
                               outcomes or self.outcomes, reference_root=self.release)

    def test_fixed_denominators_keep_missing_outputs_and_e2e_links(self):
        result = self.evaluate()
        self.assertEqual(len(self.manifest["missing_outputs"]), 514)
        self.assertEqual(result["counts"], {"dx_strict": 1, "dx_hit": 2, "rx_valid": 2, "e2e": 1})
        self.assertEqual(result["points"], {"dx_strict": 1 / 6162, "dx_hit": 2 / 6162,
                                            "rx_valid": 2 / 2575, "e2e": 1 / 2575})
        self.assertEqual(result["rx_missing_slots"], 2573)
        self.assertEqual(result["generation_parse_failures"], 514)
        self.assertEqual(result["dataset_counts"]["B"]["diagnosis_n"], 4072)

    def test_duplicate_generation_attempts_cannot_replace_original(self):
        row = read_jsonl(self.predictions)[0]
        path = self.root / "duplicate_predictions.jsonl"
        write_jsonl(path, [{**row, "response_text": "malformed first response"}, row])
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            prepare_claims_jobs(path, self.release, self.root / "must_not_prepare", model="test")

    def test_canonical_aliases_and_failed_status_are_preserved(self):
        row = read_jsonl(self.predictions)[0]
        path = self.root / "aliases.jsonl"
        write_jsonl(path, [{**row, "audio_id": "000", "status": "failed"}])
        output = self.root / "failed_aliases"
        prepare_claims_jobs(path, self.release, output, model="test")
        sample = read_jsonl(output / "test_samples.jsonl")[0]
        self.assertEqual(sample["sample_id"], "A_000")
        self.assertEqual(sample["generation_error"], "status:failed")
        self.assertEqual(sample["parsed"], {})
        write_jsonl(path, [{**row, "audio_id": "B_000"}])
        with self.assertRaisesRegex(ValueError, "conflicting canonical"):
            prepare_claims_jobs(path, self.release, self.root / "conflicting_aliases", model="test")

    def test_interrupted_judge_request_is_durable_and_not_reissued(self):
        tokenizer_path = self.root / "fake_tokenizer"
        tokenizer_path.mkdir(exist_ok=True)
        (tokenizer_path / "tokenizer_config.json").write_text("{}")
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: {"input_ids": [1]})
        calls = []

        def interrupted_request(**request):
            calls.append(request)
            raise KeyboardInterrupt("simulated interruption during remote request")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=interrupted_request)))
        dependencies = {"openai": SimpleNamespace(OpenAI=lambda **kw: client),
                        "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))}
        output = self.root / "interrupted_outcomes.jsonl"
        with patch.dict("sys.modules", dependencies):
            with self.assertRaises(KeyboardInterrupt):
                run_claim_judge(self.output / "jobs.jsonl", output, endpoint="http://127.0.0.1:9999/v1",
                                model="fake-judge", tokenizer_path=tokenizer_path)
            markers = list(output.with_suffix(".attempts").rglob("*.started.json"))
            self.assertEqual(len(markers), 1)
            self.assertEqual(json.loads(markers[0].read_text())["request"], calls[0])
            with self.assertRaisesRegex(ValueError, "Interrupted raw attempts"):
                run_claim_judge(self.output / "jobs.jsonl", output, endpoint="http://127.0.0.1:9999/v1",
                                model="fake-judge", tokenizer_path=tokenizer_path)
        self.assertEqual(len(calls), 1)

    def test_same_size_invalid_job_population_is_rejected(self):
        rows = read_jsonl(self.output / "jobs.jsonl")
        invalid = {str(i): {**rows[0], "id": str(i)} for i in range(1030)}
        with self.assertRaisesRegex(ValueError, "Judge ID disagrees"):
            _validate_job_population(invalid)
        rows[-1]["pred"] = [{"id": "p1"}] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate judge pred"):
            _validate_job_population({row["id"]: row for row in rows})

    def test_duplicate_and_missing_judge_outcomes_stop_replay(self):
        rows = read_jsonl(self.outcomes)
        for suffix, changed, message in (("duplicate", rows + rows[:1], "Duplicate judge outcome"),
                                         ("missing", rows[:-1], "Incomplete or unexpected")):
            with self.subTest(suffix=suffix):
                path = self.root / f"{suffix}.jsonl"
                write_jsonl(path, changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate(path)

    def test_changed_job_hash_and_raw_response_are_rejected(self):
        rows = read_jsonl(self.outcomes)
        with_hash = copy.deepcopy(rows)
        with_hash[0]["job_sha256"] = "changed"
        with_raw = copy.deepcopy(rows)
        with_raw[0]["attempts"][0]["response"]["choices"][0]["message"]["content"] = "{}"
        for suffix, changed, message in (("hash", with_hash, "hash mismatch"), ("raw", with_raw, "differs from raw response")):
            with self.subTest(suffix=suffix):
                path = self.root / f"{suffix}.jsonl"
                write_jsonl(path, changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate(path)

    def test_release_content_mismatch_is_rejected_even_with_equal_counts(self):
        samples = read_jsonl(self.output / "test_samples.jsonl")
        samples[0]["gt"]["claims"][0]["claims"][0]["text"] = "Changed expert claim"
        path = self.root / "changed_samples.jsonl"
        write_jsonl(path, samples)
        with self.assertRaisesRegex(ValueError, "GT content/category mismatch"):
            self.evaluate(samples=path)

    def test_changed_release_cannot_keep_the_canonical_version_label(self):
        release = self.root / "changed_release"
        shutil.copytree(self.release, release)
        atomic = release / "annotations/atomic_claims.jsonl"
        rows = read_jsonl(atomic)
        rows[0]["text"] = "Replacement claim with all counts unchanged"
        write_jsonl(atomic, rows)
        with self.assertRaisesRegex(ValueError, "Canonical reference content mismatch"):
            evaluate_claims(self.output / "test_samples.jsonl", self.output / "jobs.jsonl", self.outcomes,
                            reference_root=release)


if __name__ == "__main__":
    unittest.main()
