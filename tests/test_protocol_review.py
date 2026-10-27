"""Provenance tamper and public scorer integration regressions.

Small fixtures replace the frozen hash constants explicitly; production never
trusts receipt-provided hashes as the source of canonical reference identity.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vocalcoachbench import cli, protocol
from vocalcoachbench.io import read_json, write_json, write_jsonl
from vocalcoachbench.metrics import evaluate_top3
from vocalcoachbench.postprocess import normalize_rows


class ProtocolIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inventory = [("a", "A_a"), ("b", "A_b"), ("c", "A_c"), ("d", "B_d")]
        self.metadata = self.root / "audio_metadata.jsonl"
        write_jsonl(self.metadata, [{"audio_id": audio, "recording_id": rec, "path": None} for audio, rec in self.inventory])
        self.pairs = self.root / "triplet_pairs.jsonl"
        write_jsonl(self.pairs, [
            {"pair_id": a + b, "triplet_id": "t", "audio_a_id": a, "audio_b_id": b, "reference_orders": [["a", "b", "c"]]}
            for a, b in [("a", "b"), ("a", "c"), ("b", "c")]
        ])
        self.refs = self.root / "top3_references.jsonl"
        write_jsonl(self.refs, [{"audio_id": a, "references": [["PITCH", "BREATH", "DICTION"]]} for a, _ in self.inventory])
        self.predictions = self.root / "score_predictions.jsonl"
        write_jsonl(self.predictions, [{"audio_id": a, "recording_id": r, "quality_score_0_5": s} for (a, r), s in zip(self.inventory, [5, 3, 1, 2])])
        self.receipt = self.root / "prepare_summary.json"
        self.receipt_data = {
            "dataset_version": protocol.DATASET_VERSION,
            "canonical_annotation_sha256": dict(protocol.ANNOTATION_HASHES),
            "triplet_policy": "clean",
            "prepared_file_sha256": {p.name: protocol.sha256(p) for p in [self.pairs, self.refs, self.metadata]},
        }
        write_json(self.receipt, self.receipt_data)
        expected_refs = {p.name: protocol.sha256(p) for p in [self.pairs, self.refs]}
        mapping_digest = hashlib.sha256(json.dumps(self.inventory, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        self.reference_patch = patch.object(protocol, "PREPARED_REFERENCE_HASHES", expected_refs)
        self.mapping_patch = patch.object(protocol, "INVENTORY_ID_SHA256", mapping_digest)
        self.reference_patch.start()
        self.mapping_patch.start()
        self.addCleanup(self.reference_patch.stop)
        self.addCleanup(self.mapping_patch.stop)

    def test_forged_receipt_cannot_relabel_a_subset_canonical(self):
        write_jsonl(self.refs, [{"audio_id": "a", "references": [["PITCH", "BREATH", "DICTION"]]}])
        self.receipt_data["prepared_file_sha256"][self.refs.name] = protocol.sha256(self.refs)
        write_json(self.receipt, self.receipt_data)
        with self.assertRaisesRegex(ValueError, "pinned canonical scorer input"):
            protocol.scoring_provenance(self.refs, self.predictions, require_canonical=True)

    def test_inventory_mutation_is_not_covered_by_a_new_receipt_hash(self):
        write_jsonl(self.metadata, [{"audio_id": "typo", "recording_id": "A_typo"}])
        self.receipt_data["prepared_file_sha256"][self.metadata.name] = protocol.sha256(self.metadata)
        write_json(self.receipt, self.receipt_data)
        with self.assertRaisesRegex(ValueError, "canonical 515 recording identities"):
            protocol.canonical_audio_inventory(self.pairs)

    def test_metadata_enriched_normalized_rows_score_successfully(self):
        raw = [{"audio_id": "a", "recording_id": "A_a", "response_text": '{"top3_issues":["PITCH","BREATH","DICTION"]}'}]
        normalized = normalize_rows("top3_score", raw)
        summary, _ = evaluate_top3([{"audio_id": "a", "references": [["PITCH", "BREATH", "DICTION"]]}], normalized)
        self.assertEqual(summary["f1_at_3"], 1)
        with self.assertRaisesRegex(ValueError, "Conflicting prediction ID"):
            evaluate_top3([{"audio_id": "a", "references": [["PITCH", "BREATH", "DICTION"]]}], [{**normalized[0], "recording_id": "A_wrong"}])

    def test_exact_inventory_mapping_accepts_recording_ids_and_detects_wrong_dataset(self):
        rows, expected = protocol.canonical_recording_predictions(self.pairs, [{"recording_id": "B_d", "score": 3}])
        self.assertEqual(rows[0]["audio_id"], "d")
        self.assertEqual(expected, {"a", "b", "c", "d"})
        with self.assertRaisesRegex(ValueError, "does not match canonical"):
            protocol.canonical_recording_predictions(self.pairs, [{"audio_id": "d", "recording_id": "A_d", "score": 3}])
        with self.assertRaisesRegex(ValueError, "Conflicting prediction ID"):
            protocol.canonical_recording_predictions(self.pairs, [{"audio_id": "a", "sample_id": "b", "score": 3}])

    def test_cli_scores_all_inventory_and_rejects_unknown_ids(self):
        args = ["evaluate-score-triplet", "--pairs", str(self.pairs), "--scores", str(self.predictions), "--require-canonical"]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli.main(args)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["summary"]["score_pairwise_accuracy"], 1)
        self.assertEqual(result["summary"]["unused_score_n"], 1)
        self.assertEqual(result["provenance"]["dataset_version"], protocol.DATASET_VERSION)
        write_jsonl(self.predictions, [{"audio_id": "typo", "quality_score_0_5": 5}])
        with self.assertRaisesRegex(ValueError, "Unknown prediction ID"):
            cli.main(args)

    def test_all_metrics_embeds_task_provenance_and_validates_score_inventory(self):
        source = Path(__file__).resolve().parents[1] / "scripts/evaluate_all.py"
        spec = importlib.util.spec_from_file_location("public_evaluate_all_review", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = self.root / "results"
        args = [str(source), "--data-dir", str(self.root), "--predictions-dir", str(self.root),
                "--output-dir", str(output), "--scores", str(self.predictions), "--require-canonical"]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        task = read_json(output / "score_triplet.json")
        self.assertEqual(task["summary"]["unused_score_n"], 1)
        self.assertEqual(task["provenance"]["dataset_version"], protocol.DATASET_VERSION)
        write_jsonl(self.predictions, [{"audio_id": "typo", "score": 5}])
        with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "Unknown prediction ID"):
            module.main()

    def test_absent_receipt_never_claims_canonical(self):
        self.receipt.unlink()
        self.assertEqual(protocol.scoring_provenance(self.refs, self.predictions)["dataset_version"], "custom_reference_set")
        with self.assertRaisesRegex(ValueError, "Canonical scoring requires"):
            protocol.scoring_provenance(self.refs, self.predictions, require_canonical=True)


if __name__ == "__main__":
    unittest.main()
