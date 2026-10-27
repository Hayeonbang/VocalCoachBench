from __future__ import annotations

import argparse
import json
from typing import Any

from .io import read_jsonl, write_json
from .metrics import evaluate_direct_triplets, evaluate_score_triplets, evaluate_segment, evaluate_top3
from .protocol import DATASET_VERSION, canonical_recording_predictions, scoring_provenance


def _print_or_write(payload: dict[str, Any], out: str | None) -> None:
    if out:
        write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def _provenance(args: argparse.Namespace, reference: str, predictions: str) -> dict:
    return scoring_provenance(reference, predictions, require_canonical=args.require_canonical)


def cmd_evaluate_triplet(args: argparse.Namespace) -> None:
    pair_rows = read_jsonl(args.pairs)
    prediction_rows = read_jsonl(args.predictions)
    summary, per_triplet = evaluate_direct_triplets(pair_rows, prediction_rows)
    payload: dict[str, Any] = {"summary": summary, "provenance": _provenance(args, args.pairs, args.predictions)}
    if args.details:
        payload["triplets"] = per_triplet
    _print_or_write(payload, args.out)


def cmd_evaluate_top3(args: argparse.Namespace) -> None:
    reference_rows = read_jsonl(args.references)
    prediction_rows = read_jsonl(args.predictions)
    provenance = _provenance(args, args.references, args.predictions)
    if provenance["dataset_version"] == DATASET_VERSION:
        prediction_rows, _ = canonical_recording_predictions(args.references, prediction_rows)
    summary, per_audio = evaluate_top3(reference_rows, prediction_rows)
    payload: dict[str, Any] = {"summary": summary, "provenance": provenance}
    if args.details:
        payload["audios"] = per_audio
    _print_or_write(payload, args.out)


def cmd_evaluate_score_triplet(args: argparse.Namespace) -> None:
    pair_rows = read_jsonl(args.pairs)
    score_rows = read_jsonl(args.scores)
    provenance = _provenance(args, args.pairs, args.scores)
    expected_ids = None
    if provenance["dataset_version"] == DATASET_VERSION:
        score_rows, expected_ids = canonical_recording_predictions(args.pairs, score_rows)
    summary, per_triplet = evaluate_score_triplets(pair_rows, score_rows, expected_audio_ids=expected_ids)
    payload: dict[str, Any] = {"summary": summary, "provenance": provenance}
    if args.details:
        payload["triplets"] = per_triplet
    _print_or_write(payload, args.out)


def cmd_evaluate_segment(args: argparse.Namespace) -> None:
    reference_rows = read_jsonl(args.references)
    prediction_rows = read_jsonl(args.predictions)
    summary, per_segment = evaluate_segment(reference_rows, prediction_rows)
    payload: dict[str, Any] = {"summary": summary, "provenance": _provenance(args, args.references, args.predictions)}
    if args.details:
        payload["segments"] = per_segment
    _print_or_write(payload, args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vocalcoachbench",
        description="Evaluate VocalCoachBench prediction files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    triplet = subparsers.add_parser(
        "evaluate-triplet",
        help="Evaluate direct pairwise triplet ranking predictions.",
    )
    triplet.add_argument("--pairs", required=True, help="Pair manifest JSONL.")
    triplet.add_argument("--predictions", required=True, help="Pairwise prediction JSONL.")
    triplet.add_argument("--out", default=None, help="Optional summary JSON path.")
    triplet.add_argument("--details", action="store_true", help="Include per-triplet metrics.")
    triplet.set_defaults(func=cmd_evaluate_triplet)

    score_triplet = subparsers.add_parser(
        "evaluate-score-triplet",
        help="Evaluate auxiliary score-derived triplet ranking predictions.",
    )
    score_triplet.add_argument("--pairs", required=True, help="Pair manifest JSONL.")
    score_triplet.add_argument("--scores", required=True, help="Single-audio score prediction JSONL.")
    score_triplet.add_argument("--out", default=None, help="Optional summary JSON path.")
    score_triplet.add_argument("--details", action="store_true", help="Include per-triplet metrics.")
    score_triplet.set_defaults(func=cmd_evaluate_score_triplet)

    top3 = subparsers.add_parser(
        "evaluate-top3",
        help="Evaluate top-3 vocal issue predictions.",
    )
    top3.add_argument("--references", required=True, help="Top-3 reference JSONL.")
    top3.add_argument("--predictions", required=True, help="Top-3 prediction JSONL.")
    top3.add_argument("--out", default=None, help="Optional summary JSON path.")
    top3.add_argument("--details", action="store_true", help="Include per-audio metrics.")
    top3.set_defaults(func=cmd_evaluate_top3)

    segment = subparsers.add_parser(
        "evaluate-segment",
        help="Evaluate segment-level issue classification.",
    )
    segment.add_argument("--references", required=True, help="Segment reference JSONL.")
    segment.add_argument("--predictions", required=True, help="Segment prediction JSONL.")
    segment.add_argument("--out", default=None, help="Optional summary JSON path.")
    segment.add_argument("--details", action="store_true", help="Include per-segment metrics.")
    segment.set_defaults(func=cmd_evaluate_segment)

    for command in (triplet, score_triplet, top3, segment):
        command.add_argument("--require-canonical", action="store_true", help="Require the hash-verified canonical preparation receipt alongside references.")

    claims = subparsers.add_parser("evaluate-claims", help="Replay saved Claims judge outcomes with canonical fixed denominators.")
    claims.add_argument("--samples", required=True)
    claims.add_argument("--jobs", required=True)
    claims.add_argument("--outcomes", required=True)
    claims.add_argument("--hf-root", required=True, help="Canonical HF dataset snapshot directory.")
    claims.add_argument("--model", default=None)
    claims.add_argument("--out", default=None)
    claims.set_defaults(func=cmd_evaluate_claims)

    prepare_claims = subparsers.add_parser("prepare-claims", help="Prepare canonical Claims judge inputs offline without inference.")
    prepare_claims.add_argument("--predictions", required=True, help="Original model responses with canonical recording identities.")
    prepare_claims.add_argument("--hf-root", required=True, help="Canonical HF dataset snapshot directory.")
    prepare_claims.add_argument("--output-dir", required=True)
    prepare_claims.add_argument("--model", required=True)
    prepare_claims.add_argument("--parser", choices=("structured", "stored_parsed"), default="structured")
    prepare_claims.set_defaults(func=cmd_prepare_claims)
    return parser


def cmd_evaluate_claims(args: argparse.Namespace) -> None:
    from .claims import evaluate_claims
    result = evaluate_claims(args.samples, args.jobs, args.outcomes, reference_root=args.hf_root, model=args.model)
    _print_or_write(result, args.out)


def cmd_prepare_claims(args: argparse.Namespace) -> None:
    from .claims import prepare_claims_jobs
    result = prepare_claims_jobs(args.predictions, args.hf_root, args.output_dir,
                                model=args.model, parser=args.parser)
    _print_or_write(result, None)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
