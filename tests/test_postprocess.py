from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vocalcoachbench.postprocess import (
    explicit_quality_score, normalize_rows, parse_category, parse_category_list,
    parse_winner, simple_score,
)

ROOT = Path(__file__).resolve().parents[1]


class AlignedParserTests(unittest.TestCase):
    def test_main_explicit_numeric_scores(self):
        accepted = [0, 5, 3.25, "3.25", "+.5", "3e0", "3 out of 5", "3/5"]
        for value in accepted:
            with self.subTest(value=value):
                self.assertIsNotNone(explicit_quality_score(json.dumps({"quality_score_0_5": value})))
        for key in ("quality_score_0_5", "qualityScore_0_5", "qualityScore"):
            self.assertEqual(explicit_quality_score(json.dumps({key: 3})), 3)

    def test_main_rejects_numbers_that_used_to_be_clipped_or_salvaged(self):
        rejected = [-1, 6, 10**400, True, False, None, [], {}, float("nan"), float("inf"),
                    "6/5", "3/10", "3/0", "NaN", "1e999", "score 3", "3 stars", "3 / inf"]
        for value in rejected:
            with self.subTest(value=value):
                self.assertIsNone(explicit_quality_score(json.dumps({"quality_score_0_5": value})))
        for text in ('{"score": 3}', '3', 'Score: 3/5', '{"quality_score_0_5": 3',
                     '{"quality_score_0_5": true, "qualityScore": 4}'):
            with self.subTest(text=text):
                self.assertIsNone(explicit_quality_score(text))

    def test_explicit_score_preserves_frozen_json_wrapper_handling(self):
        for text in ('```json\n{"quality_score_0_5":3}\n```',
                     'Answer: {"quality_score_0_5":3}',
                     '[{"quality_score_0_5":3}]',
                     r'{\"quality_score_0_5\":3}'):
            self.assertEqual(explicit_quality_score(text), 3)
        self.assertIsNone(explicit_quality_score('[{"quality_score_0_5":3}, {"quality_score_0_5":4}]'))

    def test_simple_score_is_a_distinct_surface(self):
        for text in ("3.5", "3.5/5", "3.5 out of 5", " +3.5 "):
            self.assertEqual(simple_score(text), 3.5)
        for text in ("I rate this 3.5", "3e0", "True", "6", "-1", "3/10", "NaN"):
            self.assertIsNone(simple_score(text))
        raw = [{"audio_id": "r1", "response_text": "3.5"}]
        self.assertEqual(normalize_rows("score", raw, protocol="simple")[0]["quality_score_0_5"], 3.5)
        self.assertIsNone(normalize_rows("score", raw)[0]["quality_score_0_5"])
        explicit = [{"audio_id": "r1", "response_text": '{"qualityScore":3}'}]
        self.assertEqual(normalize_rows("score", explicit, protocol="simple")[0]["quality_score_0_5"], 3)
        top = [{"audio_id": "r1", "response_text": '{"top3_issues":["PITCH"],"quality_score_0_5":3}'}]
        self.assertNotIn("quality_score_0_5", normalize_rows("top3_score", top, protocol="simple")[0])

    def test_category_text_fallback_matches_aligned_policy(self):
        self.assertEqual(parse_category_list('BREATH, PHONATION, PITCH'), ["BREATH", "VOCALIZATION", "PITCH"])
        self.assertEqual(parse_category_list('{"issues": {"first": "tempo", "second": "support", "third": "style"}}'),
                         ["RHYTHM", "BREATH", "EXPRESSION"])
        self.assertEqual(parse_category_list('{"top3_issues":["PHONATION","VOCALIZATION","PITCH"]}'),
                         ["VOCALIZATION", "PITCH"])
        # Malformed JSON can still have deterministic category matches in the
        # published parser; do not impose an unrelated JSON-only gate.
        self.assertEqual(parse_category_list('{"top3_issues":["BREATH","PITCH"'), ["BREATH", "PITCH"])
        self.assertEqual(parse_category('{"main_problem":"phonation"}'), "VOCALIZATION")
        self.assertEqual(parse_category('Tuning is the main issue.'), "PITCH")
        self.assertEqual(parse_category_list('I cannot evaluate this audio.'), [])

    def test_error_rows_fail_even_with_good_raw_or_canonical_predictions(self):
        examples = [
            ("top3_score", {"audio_id": "r", "top3_issues": ["PITCH"], "quality_score_0_5": 5,
                            "response_text": '{"top3_issues":["PITCH"],"quality_score_0_5":5}'}, "top3_issues", []),
            ("segment", {"sample_id": "s", "category": "PITCH", "response_text": "PITCH"}, "category", None),
            ("direct_pairwise", {"pair_id": "p", "winner": "A", "response_text": "A"}, "winner", None),
            ("score", {"audio_id": "r", "response_text": '{"quality_score_0_5":5}'}, "quality_score_0_5", None),
        ]
        for task, row, key, expected in examples:
            row["error"] = "provider_timeout"
            with self.subTest(task=task):
                result = normalize_rows(task, [row])[0]
                self.assertEqual(result[key], expected)
                self.assertEqual(result["error"], row["error"])
                self.assertEqual(result["response_text"], row["response_text"])

    def test_invalid_raw_text_cannot_be_rescued_by_convenience_fields(self):
        row = {"audio_id": "r", "top3_issues": ["PITCH"], "quality_score_0_5": 5, "response_text": ""}
        result = normalize_rows("top3_score", [row])[0]
        self.assertEqual(result["top3_issues"], [])
        self.assertIsNone(result["quality_score_0_5"])
        self.assertEqual(normalize_rows("top3_score", [{"audio_id": "r", "top3_issues": ["PHONATION"], "quality_score_0_5": 3}])[0]["quality_score_0_5"], 3)

    def test_duplicate_attempts_rejected_and_missing_ids_not_invented(self):
        for task, id_key in (("top3_score", "audio_id"), ("segment", "sample_id"), ("direct_pairwise", "pair_id"), ("score", "audio_id")):
            with self.subTest(task=task):
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    normalize_rows(task, [{id_key: "r", "error": "first_failure"}, {id_key: "r", "response_text": "A"}])
                with self.assertRaisesRegex(ValueError, "Missing"):
                    normalize_rows(task, [{"audio_filename": "ambiguous.wav", "response_text": "PITCH"}])

    def test_raw_identity_orientation_and_provenance_preserved(self):
        raw = {"pair_id": "p", "audio_a_id": "right", "audio_b_id": "left", "response_text": '  {"winner":"B"}\n',
               "source_row": 4, "source_row_sha256": "receipt", "protocol": "custom-wrapper"}
        result = normalize_rows("direct_pairwise", [raw])[0]
        for key, value in raw.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["winner"], "B")
        self.assertEqual(result["normalization_protocol"], "main")
        self.assertEqual(parse_winner("Audio B"), "B")
        self.assertEqual(parse_winner('{"winner":"Audio B"}'), "B")
        self.assertIsNone(parse_winner("This answer contains letter B."))
        self.assertIsNone(normalize_rows("direct_pairwise", [{"pair_id": "p", "response_text": "unparsed"}])[0]["winner"])

    def test_stored_reports_require_explicit_option(self):
        raw = {"audio_id": "r", "response_text": "PITCH", "parsed": {"top3_issues": ["RHYTHM"]}}
        self.assertEqual(normalize_rows("top3_score", [raw])[0]["top3_issues"], ["PITCH"])
        self.assertEqual(normalize_rows("top3_score", [raw], use_stored_parsed=True)[0]["top3_issues"], ["RHYTHM"])
        failed = normalize_rows("top3_score", [{"audio_id": "r", "error": "missing_output", "response_text": ""}], use_stored_parsed=True)[0]
        self.assertEqual(failed["top3_issues"], [])
        with self.assertRaisesRegex(ValueError, "parsed report"):
            normalize_rows("top3_score", [{"audio_id": "r"}], use_stored_parsed=True)
        with self.assertRaisesRegex(ValueError, "must be text"):
            normalize_rows("top3_score", [{"audio_id": "r", "response_text": {"top3_issues": ["PITCH"]}}])

    def test_frozen_prompt_hashes(self):
        manifest = json.loads((ROOT / "prompts/manifest.json").read_text())
        self.assertEqual(len(manifest["prompts"]), 8)
        for item in manifest["prompts"]:
            data = (ROOT / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(hashlib.sha256(data.decode().strip().encode()).hexdigest(), item["runtime_stripped_sha256"])


class InferenceAttemptTests(unittest.TestCase):
    def test_resume_never_regenerates_a_failed_attempt(self):
        path = ROOT / "scripts/infer_single_audio_template.py"
        spec = importlib.util.spec_from_file_location("inference_template_test", path)
        template = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(template)
        with tempfile.TemporaryDirectory() as temp:
            input_path, output_path = Path(temp) / "input.jsonl", Path(temp) / "raw.jsonl"
            input_path.write_text('{"audio_id":"r1","path":"unloaded.wav"}\n')
            argv = [str(path), "--task", "top3_score", "--inputs", str(input_path), "--out", str(output_path)]
            with patch.object(sys, "argv", argv), patch.object(template, "call_model", side_effect=RuntimeError("first failure")) as model:
                template.main()
                self.assertEqual(model.call_count, 1)
                self.assertEqual(model.call_args.kwargs["generation_config"]["max_audio_length_seconds"], 180)
            first = output_path.read_bytes()
            with patch.object(sys, "argv", argv + ["--resume"]), patch.object(template, "call_model") as model:
                template.main()
                model.assert_not_called()
            self.assertEqual(first, output_path.read_bytes())
            row = json.loads(first)
            self.assertIn("first failure", row["error"])
            self.assertEqual(len(row["request_provenance"]["prompt_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
