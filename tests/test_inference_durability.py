"""Offline interruption tests: the adapter must never regenerate a first request."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vocalcoachbench import inference_io

ROOT = Path(__file__).resolve().parents[1]


class InferenceDurabilityTests(unittest.TestCase):
    def fixture(self, directory: Path, kind: str):
        script = ROOT / "scripts" / f"infer_{kind}_template.py"
        spec = importlib.util.spec_from_file_location(f"durability_{kind}", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inputs, output = directory / "input.jsonl", directory / "predictions.jsonl"
        if kind == "single_audio":
            rows = [{"audio_id": f"r{n}", "path": f"not_loaded_{n}.wav"} for n in (1, 2)]
            extra = ["--task", "top3_score", "--inputs", str(inputs)]
            id_key = "audio_id"
        else:
            rows = [{"pair_id": f"p{n}", "audio_a_id": "a", "audio_b_id": "b",
                     "audio_a_path": "not_loaded_a.wav", "audio_b_path": "not_loaded_b.wav"} for n in (1, 2)]
            extra = ["--pairs", str(inputs)]
            id_key = "pair_id"
        inputs.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return module, [str(script), *extra, "--out", str(output)], output, id_key

    def test_first_request_interruption_leaves_durable_evidence_and_blocks_resume(self):
        for kind in ("single_audio", "direct_pairwise"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                module, args, output, _ = self.fixture(Path(temp), kind)

                def interrupt(*unused, **kwargs):
                    markers = list(module.attempt_directory(output).glob("*.json"))
                    self.assertEqual(len(markers), 1, "marker must precede the adapter call")
                    marker = json.loads(markers[0].read_text())
                    self.assertEqual(marker["request_provenance"]["evaluation_policy"], "strict_all_attempts")
                    raise KeyboardInterrupt("request may already have reached provider")

                with patch.object(sys, "argv", args), patch.object(module, "call_model", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main()
                self.assertFalse(output.exists())
                with patch.object(sys, "argv", args + ["--resume"]), patch.object(module, "call_model") as model:
                    with self.assertRaisesRegex(ValueError, "Unresolved started attempt"):
                        module.main()
                    model.assert_not_called()
                # Starting again without --resume cannot bypass the orphan guard.
                with patch.object(sys, "argv", args), patch.object(module, "call_model") as model:
                    with self.assertRaisesRegex(ValueError, "Unresolved started attempt"):
                        module.main()
                    model.assert_not_called()

    def test_interrupted_second_request_preserves_first_output(self):
        for kind in ("single_audio", "direct_pairwise"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                module, args, output, id_key = self.fixture(Path(temp), kind)
                with patch.object(sys, "argv", args), patch.object(module, "call_model", side_effect=[{"response_text": "first"}, KeyboardInterrupt()]):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main()
                completed = output.read_bytes()
                row = json.loads(completed)
                self.assertEqual(row["response_text"], "first")
                self.assertTrue(row[id_key].endswith("1"))
                self.assertEqual(len(list(module.attempt_directory(output).glob("*.json"))), 2)
                with patch.object(sys, "argv", args + ["--resume"]), patch.object(module, "call_model") as model:
                    with self.assertRaisesRegex(ValueError, "Unresolved started attempt"):
                        module.main()
                    model.assert_not_called()
                self.assertEqual(output.read_bytes(), completed)

    def test_write_failure_keeps_previous_snapshot_and_blocks_retry(self):
        for kind in ("single_audio", "direct_pairwise"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                module, args, output, _ = self.fixture(Path(temp), kind)
                with patch.object(sys, "argv", args + ["--limit", "1"]), patch.object(module, "call_model", return_value={"response_text": "first"}):
                    module.main()
                completed = output.read_bytes()
                with patch.object(sys, "argv", args + ["--resume"]), patch.object(module, "call_model", return_value={"response_text": "second"}) as model, patch.object(inference_io.os, "replace", side_effect=OSError("disk write failure")):
                    with self.assertRaisesRegex(OSError, "disk write failure"):
                        module.main()
                    self.assertEqual(model.call_count, 1)
                self.assertEqual(output.read_bytes(), completed)
                self.assertEqual(list(output.parent.glob("*.tmp")), [])
                with patch.object(sys, "argv", args + ["--resume"]), patch.object(module, "call_model") as model:
                    with self.assertRaisesRegex(ValueError, "Unresolved started attempt"):
                        module.main()
                    model.assert_not_called()

    def test_completed_failures_are_persisted_and_never_called_again(self):
        for kind in ("single_audio", "direct_pairwise"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                module, args, output, _ = self.fixture(Path(temp), kind)
                with patch.object(sys, "argv", args), patch.object(module, "call_model", side_effect=RuntimeError("first response failed")) as model:
                    module.main()
                    self.assertEqual(model.call_count, 2)
                completed = output.read_bytes()
                self.assertEqual(len(completed.splitlines()), 2)
                self.assertTrue(all("first response failed" in json.loads(line)["error"] for line in completed.splitlines()))
                with patch.object(sys, "argv", args + ["--resume"]), patch.object(module, "call_model") as model:
                    module.main()
                    model.assert_not_called()
                self.assertEqual(output.read_bytes(), completed)


if __name__ == "__main__":
    unittest.main()
