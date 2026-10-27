"""Numerical contract tests for strict canonical structured evaluation."""
from __future__ import annotations

import math
import unittest

from vocalcoachbench.metrics import (
    evaluate_direct_triplets,
    evaluate_score_triplets,
    evaluate_segment,
    evaluate_top3,
    ndcg_at_3,
)


TOP_REFS = [
    {"audio_id": "r1", "references": [["PITCH", "RHYTHM", "DICTION"], ["PITCH", "BREATH", "VOCALIZATION"]]},
    {"audio_id": "r2", "references": [["PITCH", "RHYTHM", "DICTION"]]},
]
TOP_PREDS = [
    {"audio_id": "r1", "top3_issues": ["PITCH", "RHYTHM", "DICTION"]},
    {"audio_id": "r2", "top3_issues": ["PITCH", "BREATH", "EXPRESSION"]},
]
SEG_REFS = [
    {"sample_id": "s1", "labels": ["PITCH", "RHYTHM"]},
    {"sample_id": "s2", "labels": ["PITCH"]},
    {"sample_id": "s3", "labels": ["DICTION"]},
]
PAIRS = [
    {"pair_id": pair_id, "triplet_id": "t", "audio_a_id": a, "audio_b_id": b,
     "reference_orders": [["a", "b", "c"]]}
    for pair_id, a, b in [("ab", "a", "b"), ("ac", "a", "c"), ("bc", "b", "c")]
]


class Top3Tests(unittest.TestCase):
    def test_equal_recording_weights_and_absent_classes(self):
        summary, details = evaluate_top3(TOP_REFS, TOP_PREDS)
        self.assertEqual(summary["audio_n"], 2)
        self.assertEqual(summary["evaluation_mode"], "strict_all_attempts")
        self.assertAlmostEqual(summary["f1_at_3"], 0.5)
        self.assertAlmostEqual(summary["macro_f1"], (1 + 0.4 + 0.4) / 7)
        classes = {r["category"]: r for r in summary["class_metrics"]}
        self.assertEqual(len(classes), 7)
        self.assertEqual(classes["TECHNIQUE"]["f1"], 0)
        self.assertEqual(classes["RHYTHM"]["tp"], 0.5)
        self.assertEqual(classes["RHYTHM"]["fp"], 0.5)
        self.assertEqual(classes["RHYTHM"]["fn"], 1.0)
        self.assertEqual(classes["RHYTHM"]["reference_annotations"], 2)
        self.assertEqual([r["reference_count"] for r in details], [2, 1])

    def test_missing_and_explicit_errors_do_not_shrink_population(self):
        missing, _ = evaluate_top3(TOP_REFS, TOP_PREDS[:1])
        error, rows = evaluate_top3(TOP_REFS, [TOP_PREDS[0], {**TOP_PREDS[1], "error": "backend failure"}])
        self.assertEqual(missing, error)
        self.assertEqual(missing["n"], 2)
        self.assertEqual(missing["valid_top3_rate"], 0.5)
        self.assertAlmostEqual(missing["macro_f1"], (2 / 3 + 0.4 + 0.4) / 7)
        self.assertEqual(rows[1]["prediction"], [])

    def test_partial_prediction_keeps_canonical_partial_credit(self):
        summary, _ = evaluate_top3(TOP_REFS, [{"audio_id": "r1", "top3_issues": ["PITCH", "invalid"]}])
        self.assertEqual(summary["valid_top3_rate"], 0)
        self.assertAlmostEqual(summary["f1_at_3"], 1 / 6)
        self.assertAlmostEqual(summary["macro_f1"], (2 / 3) / 7)

    def test_aliases_canonical_recording_ids_and_auxiliary_definitions(self):
        ref = [{"recording_id": "r", "top3_issues": ["PHONATION", "BREATH", "TECHNIQUE"]}]
        summary, _ = evaluate_top3(ref, [{"recording_id": "r", "top3_issues": ["VOCALIZATION", "BREATH", "TECHNIQUE"]}])
        self.assertEqual(summary["parent_f1"], 1)
        self.assertEqual(summary["ndcg_at_3"], 1)
        self.assertAlmostEqual(summary["macro_f1"], 3 / 7)
        ideal = 3 + 2 / math.log2(3) + 1 / 2
        self.assertAlmostEqual(ndcg_at_3(["DICTION", "RHYTHM", "PITCH"], ["PITCH", "RHYTHM", "DICTION"]), (1 + 2 / math.log2(3) + 3 / 2) / ideal)

    def test_bad_reference_is_not_silently_removed(self):
        with self.assertRaisesRegex(ValueError, "three distinct"):
            evaluate_top3([{"audio_id": "r", "references": [["PITCH"]]}], [])


class SegmentTests(unittest.TestCase):
    def test_accepted_multi_labels_match_but_all_positives_count(self):
        summary, details = evaluate_segment(SEG_REFS, [{"sample_id": "s1", "category": "PITCH"}, {"sample_id": "s2", "category": "INVALID"}])
        self.assertEqual(summary["segment_n"], 3)
        self.assertAlmostEqual(summary["match"], 1 / 3)
        self.assertAlmostEqual(summary["macro_f1"], (2 / 3) / 7)
        self.assertEqual(summary["singleton_n"], 2)
        self.assertEqual(details[0]["match"], 1)
        classes = {r["category"]: r for r in summary["class_metrics"]}
        self.assertEqual(classes["RHYTHM"]["fn"], 1)
        self.assertEqual(classes["PITCH"]["fn"], 1)
        self.assertEqual(classes["PITCH"]["tp"], 1)

    def test_explicit_error_cannot_receive_credit(self):
        summary, _ = evaluate_segment(SEG_REFS, [{"sample_id": "s1", "category": "PITCH", "error": "timeout"}])
        self.assertEqual(summary["match"], 0)
        self.assertEqual(summary["macro_f1"], 0)
        self.assertEqual(summary["n"], 3)


class RankingTests(unittest.TestCase):
    def test_presented_ab_orientation_is_preserved(self):
        predictions = [{"pair_id": p["pair_id"], "winner": "B", "audio_a_id": p["audio_b_id"], "audio_b_id": p["audio_a_id"]} for p in PAIRS]
        summary, _ = evaluate_direct_triplets(PAIRS, predictions)
        self.assertEqual(summary["direct_pairwise_accuracy"], 1)
        self.assertEqual(summary["direct_exact_accuracy"], 1)
        with self.assertRaisesRegex(ValueError, "physical pair mismatch"):
            evaluate_direct_triplets(PAIRS, [{"pair_id": "ab", "winner": "A", "audio_a_id": "a", "audio_b_id": "x"}])

    def test_missing_pair_error_and_cycle_are_kept(self):
        valid = [{"pair_id": "ab", "winner": "A"}, {"pair_id": "ac", "winner": "A"}]
        missing, _ = evaluate_direct_triplets(PAIRS, valid)
        errored, _ = evaluate_direct_triplets(PAIRS, valid + [{"pair_id": "bc", "winner": "A", "error": "failure"}])
        self.assertEqual(missing, errored)
        self.assertEqual(missing["pair_n"], 3)
        self.assertAlmostEqual(missing["direct_pairwise_accuracy"], 2 / 3)
        self.assertEqual(missing["direct_exact_accuracy"], 0)
        cycle, _ = evaluate_direct_triplets(PAIRS, [{"pair_id": "ab", "winner": "A"}, {"pair_id": "ac", "winner": "B"}, {"pair_id": "bc", "winner": "A"}])
        self.assertEqual(cycle["cycle_or_tie_rate"], 1)
        self.assertEqual(cycle["direct_exact_accuracy"], 0)

    def test_multiple_expert_orders_average_per_triplet(self):
        refs = [{**p, "reference_orders": [["a", "b", "c"], ["a", "c", "b"]]} for p in PAIRS]
        summary, _ = evaluate_direct_triplets(refs, [{"pair_id": p["pair_id"], "winner": "A"} for p in PAIRS])
        self.assertAlmostEqual(summary["direct_pairwise_accuracy"], 5 / 6)
        self.assertEqual(summary["direct_exact_accuracy"], 0.5)

    def test_score_ties_are_failures(self):
        summary, _ = evaluate_score_triplets(PAIRS, [{"audio_id": a, "score": s} for a, s in [("a", 3), ("b", 3), ("c", 1)]])
        self.assertAlmostEqual(summary["score_pairwise_accuracy"], 2 / 3)
        self.assertEqual(summary["score_exact_accuracy"], 0)
        self.assertEqual(summary["score_tie_rate"], 1)

    def test_nonfinite_invalid_score_and_error_are_missing(self):
        for value in [float("nan"), float("inf"), -float("inf"), 10 ** 1000, 6, -1, True, "garbage", None]:
            with self.subTest(value=value):
                summary, _ = evaluate_score_triplets(PAIRS, [{"audio_id": "a", "score": 5}, {"audio_id": "b", "score": value}, {"audio_id": "c", "score": 1}])
                self.assertAlmostEqual(summary["score_pairwise_accuracy"], 1 / 3)
                self.assertEqual(summary["missing_score_rate"], 1)
        summary, _ = evaluate_score_triplets(PAIRS, [{"audio_id": "a", "score": 5}, {"audio_id": "b", "score": 3, "error": "failure"}, {"audio_id": "c", "score": 1}])
        self.assertAlmostEqual(summary["score_pairwise_accuracy"], 1 / 3)

    def test_unused_scores_allowed_and_explicit_inventory_checks_unknowns(self):
        scores = [{"audio_id": "a", "score": 5}, {"audio_id": "b", "score": 3}, {"audio_id": "c", "score": 1}, {"audio_id": "dataset_b_recording", "score": 2}]
        summary, _ = evaluate_score_triplets(PAIRS, scores)
        self.assertEqual(summary["score_pairwise_accuracy"], 1)
        self.assertEqual(summary["unused_score_n"], 1)
        with self.assertRaisesRegex(ValueError, "Unknown prediction ID"):
            evaluate_score_triplets(PAIRS, scores, expected_audio_ids={"a", "b", "c"})


class IdentityTests(unittest.TestCase):
    def test_duplicates_unknowns_and_missing_ids_rejected(self):
        cases = [
            (evaluate_top3, TOP_REFS, {"audio_id": "r1"}, {"audio_id": "typo"}),
            (evaluate_segment, SEG_REFS, {"sample_id": "s1"}, {"sample_id": "typo"}),
            (evaluate_direct_triplets, PAIRS, {"pair_id": "ab"}, {"pair_id": "typo"}),
        ]
        for evaluate, references, known, unknown in cases:
            with self.subTest(evaluate=evaluate.__name__):
                with self.assertRaisesRegex(ValueError, "Duplicate prediction ID"):
                    evaluate(references, [{**known, "error": "first attempt failed"}, known])
                with self.assertRaisesRegex(ValueError, "Unknown prediction ID"):
                    evaluate(references, [unknown])
                with self.assertRaisesRegex(ValueError, "Missing prediction ID"):
                    evaluate(references, [{}])
        with self.assertRaisesRegex(ValueError, "Duplicate prediction ID"):
            evaluate_score_triplets(PAIRS, [{"audio_id": "a", "error": "failure"}, {"audio_id": "a", "score": 5}])
        with self.assertRaisesRegex(ValueError, "Missing prediction ID"):
            evaluate_score_triplets(PAIRS, [{"score": 5}])

    def test_conflicting_id_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting prediction ID"):
            evaluate_top3(TOP_REFS, [{"audio_id": "r1", "sample_id": "r2"}])


if __name__ == "__main__":
    unittest.main()
