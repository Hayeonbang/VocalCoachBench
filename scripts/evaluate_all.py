#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocalcoachbench.io import read_jsonl, write_json
from vocalcoachbench.protocol import DATASET_VERSION, canonical_recording_predictions, scoring_provenance
from vocalcoachbench.metrics import (
    evaluate_direct_triplets,
    evaluate_score_triplets,
    evaluate_segment,
    evaluate_top3,
)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def unique_glob(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        unique = sorted(set(directory.glob(pattern)))
        if len(unique) == 1:
            return unique[0]
    return None


def resolve_prediction(explicit: str | None, directory: Path, patterns: list[str]) -> Path | None:
    if explicit:
        return Path(explicit)
    return unique_glob(directory, patterns)


def evaluate_task(
    name: str,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    output_dir: Path,
    save_details: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    task_payload: dict[str, Any] = {"summary": summary, "provenance": provenance}
    write_json(output_dir / f"{name}.json", task_payload)
    if save_details:
        write_json(output_dir / f"{name}_details.json", details)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all available VocalCoachBench prediction files.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing benchmark reference JSONL files.")
    parser.add_argument("--predictions-dir", default="predictions", help="Directory containing model prediction JSONL files.")
    parser.add_argument("--output-dir", default="results", help="Directory for metric JSON outputs.")
    parser.add_argument("--direct-pairwise", default=None, help="Optional direct pairwise prediction JSONL path.")
    parser.add_argument("--top3", default=None, help="Optional top-3 prediction JSONL path.")
    parser.add_argument("--scores", default=None, help="Optional single-audio score prediction JSONL path.")
    parser.add_argument("--segment", default=None, help="Optional segment prediction JSONL path.")
    parser.add_argument("--details", action="store_true", help="Write per-sample detail files.")
    parser.add_argument("--require-canonical", action="store_true", help="Require canonical prepared references and their receipt.")
    parser.add_argument("--claims-samples", help="Saved canonical Claims samples JSONL.")
    parser.add_argument("--claims-jobs", help="Saved Claims judge jobs JSONL.")
    parser.add_argument("--claims-outcomes", help="Saved Claims judge outcomes JSONL.")
    parser.add_argument("--hf-root", help="Canonical HF release directory for Claims reference validation.")
    parser.add_argument("--model", help="Optional Claims model identifier.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_manifest = first_existing([data_dir / "triplet_pairs.jsonl", data_dir / "direct_pairwise_pairs.jsonl"])
    top3_refs = data_dir / "top3_references.jsonl"
    segment_refs = data_dir / "segment_references.jsonl"

    direct_predictions = resolve_prediction(
        args.direct_pairwise,
        predictions_dir,
        ["direct_pairwise_predictions.jsonl", "*_direct_pairwise_predictions.jsonl", "*direct_pairwise*.jsonl", "*pairwise*.jsonl"],
    )
    top3_predictions = resolve_prediction(
        args.top3,
        predictions_dir,
        ["top3_predictions.jsonl", "*_top3_predictions.jsonl", "*_top3.jsonl"],
    )
    score_predictions = resolve_prediction(args.scores, predictions_dir, ["score_predictions.jsonl", "*_score_predictions.jsonl", "*score*.jsonl", "*scores*.jsonl"])
    segment_predictions = resolve_prediction(
        args.segment,
        predictions_dir,
        ["segment_predictions.jsonl", "*_segment_predictions.jsonl", "*_segment.jsonl"],
    )

    report: dict[str, Any] = {"tasks": {}, "missing": {}}
    report["provenance"] = {}
    for task, reference, prediction in (
        ("direct_pairwise_triplet", pair_manifest, direct_predictions),
        ("top3", top3_refs, top3_predictions),
        ("score_triplet", pair_manifest, score_predictions),
        ("segment", segment_refs, segment_predictions),
    ):
        if reference and prediction and reference.exists() and prediction.exists():
            report["provenance"][task] = scoring_provenance(reference, prediction, require_canonical=args.require_canonical)

    if pair_manifest and direct_predictions and direct_predictions.exists():
        summary, details = evaluate_direct_triplets(read_jsonl(pair_manifest), read_jsonl(direct_predictions))
        report["tasks"]["direct_pairwise_triplet"] = evaluate_task(
            "direct_pairwise_triplet",
            summary,
            details,
            output_dir,
            args.details,
            report["provenance"]["direct_pairwise_triplet"],
        )
    else:
        report["missing"]["direct_pairwise_triplet"] = {
            "reference": str(pair_manifest) if pair_manifest else "triplet_pairs.jsonl",
            "prediction": str(direct_predictions) if direct_predictions else "not found",
        }

    if top3_refs.exists() and top3_predictions and top3_predictions.exists():
        rows = read_jsonl(top3_predictions)
        provenance = report["provenance"]["top3"]
        if provenance["dataset_version"] == DATASET_VERSION:
            rows, _ = canonical_recording_predictions(top3_refs, rows)
        summary, details = evaluate_top3(read_jsonl(top3_refs), rows)
        report["tasks"]["top3"] = evaluate_task("top3", summary, details, output_dir, args.details, provenance)
    else:
        report["missing"]["top3"] = {
            "reference": str(top3_refs),
            "prediction": str(top3_predictions) if top3_predictions else "not found",
        }

    if pair_manifest and score_predictions and score_predictions.exists():
        rows = read_jsonl(score_predictions)
        provenance = report["provenance"]["score_triplet"]
        expected_ids = None
        if provenance["dataset_version"] == DATASET_VERSION:
            rows, expected_ids = canonical_recording_predictions(pair_manifest, rows)
        summary, details = evaluate_score_triplets(read_jsonl(pair_manifest), rows, expected_audio_ids=expected_ids)
        report["tasks"]["score_triplet"] = evaluate_task("score_triplet", summary, details, output_dir, args.details, provenance)
    else:
        report["missing"]["score_triplet"] = {
            "reference": str(pair_manifest) if pair_manifest else "triplet_pairs.jsonl",
            "prediction": str(score_predictions) if score_predictions else "not found",
        }

    if segment_refs.exists() and segment_predictions and segment_predictions.exists():
        summary, details = evaluate_segment(read_jsonl(segment_refs), read_jsonl(segment_predictions))
        report["tasks"]["segment"] = evaluate_task("segment", summary, details, output_dir, args.details, report["provenance"]["segment"])
    else:
        report["missing"]["segment"] = {
            "reference": str(segment_refs),
            "prediction": str(segment_predictions) if segment_predictions else "not found",
        }

    claim_args = [args.claims_samples, args.claims_jobs, args.claims_outcomes, args.hf_root]
    if any(claim_args):
        if not all(claim_args):
            raise ValueError("Claims replay requires --claims-samples, --claims-jobs, --claims-outcomes and --hf-root")
        from vocalcoachbench.claims import evaluate_claims
        claim_report = evaluate_claims(args.claims_samples, args.claims_jobs, args.claims_outcomes,
                                      reference_root=args.hf_root, model=args.model)
        write_json(output_dir / "claims.json", claim_report)
        report["tasks"]["claims"] = claim_report
    else:
        report["missing"]["claims"] = {"reason": "Saved judge jobs/outcomes were not supplied; no judge calls were made."}
    write_json(output_dir / "all_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
