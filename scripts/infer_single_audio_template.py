#!/usr/bin/env python3
"""Adapter template: greedy, first 180 seconds, one persisted attempt per ID."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocalcoachbench.io import read_jsonl
from vocalcoachbench.inference_io import (
    attempt_directory,
    check_started_attempts,
    persist_request_start,
    write_atomic_jsonl,
)

PROMPTS = {
    "main": {"top3_score": "main/top3_score.txt", "segment": "main/segment.txt"},
    "simple": {"top3_score": "simple/top3.txt", "segment": "simple/segment.txt", "score": "simple/score.txt"},
}
GENERATION_CONFIG = {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_audio_length_seconds": 180}


def call_model(prompt: str, audio_path: str, task: str, *, generation_config: dict[str, Any]) -> dict[str, Any]:
    """Implement your provider adapter and return {"response_text": raw_text}.

    The adapter must apply the supplied greedy decoding settings and crop audio
    to its first 180 seconds. Preserve the first response, including malformed
    output; disable provider/client retries that regenerate an answer. Record
    model revision, system prompt, audio preprocessing and token budget in the
    returned metadata. This template does not claim an adapter has enforced them.
    """
    raise NotImplementedError("Implement call_model() for your audio-language model.")


def row_id(row: dict[str, Any], task: str) -> str:
    key = "sample_id" if task == "segment" else "audio_id"
    return str(row.get(key) or row.get("recording_id") or row.get("sample_id") or row.get("id") or "")


def row_audio_path(row: dict[str, Any]) -> str | None:
    value = row.get("path") or row.get("audio_path")
    return str(value) if value else None


def unique_rows(rows: list[dict[str, Any]], task: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = row_id(row, task)
        if not key or key in result:
            raise ValueError(f"Missing or duplicate input/output identity: {key!r}; never overwrite an attempt")
        result[key] = row
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["top3_score", "segment", "score"])
    parser.add_argument("--protocol", choices=sorted(PROMPTS), default="main")
    parser.add_argument("--inputs", required=True, help="Input JSONL with audio_id/path or sample_id/path.")
    parser.add_argument("--prompt", default=None, help="Custom prompt; changes the frozen protocol.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip every existing attempt, including errors.")
    parser.add_argument("--dry-run", action="store_true", help="Write requests only; no model is called.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task not in PROMPTS[args.protocol]:
        raise ValueError("Main quality scores use the top3_score prompt; score is a separate Simple surface")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    inputs = unique_rows(read_jsonl(args.inputs), args.task)
    prompt_path = Path(args.prompt) if args.prompt else ROOT / "prompts" / PROMPTS[args.protocol][args.task]
    prompt_bytes = prompt_path.read_bytes()
    prompt = prompt_bytes.decode("utf-8").strip()
    provenance = {
        "protocol": args.protocol, "surface": args.task, "prompt_path": str(prompt_path.resolve()),
        "prompt_file_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "custom_prompt": args.prompt is not None, "generation_config": dict(GENERATION_CONFIG),
        "evaluation_policy": "strict_all_attempts",
    }
    out_path = Path(args.out)
    if out_path.exists() and not args.resume:
        raise ValueError("Output already exists; use --resume or a new output path")
    existing = unique_rows(read_jsonl(out_path), args.task) if args.resume and out_path.exists() else {}
    for key, row in existing.items():
        if key not in inputs or row.get("request_provenance") != provenance:
            raise ValueError(f"Resume provenance/input mismatch: {key}")
        if bool(row.get("dry_run")) != args.dry_run:
            raise ValueError("Dry-run rows cannot be resumed as model attempts (or vice versa)")
    check_started_attempts(out_path, existing, provenance)
    rows = list(existing.values())
    pending = [(key, item) for key, item in inputs.items() if key not in existing]
    if args.limit is not None:
        pending = pending[:args.limit]
    id_key = "sample_id" if args.task == "segment" else "audio_id"
    for item_id, item in pending:
        audio = row_audio_path(item)
        if audio is None and not args.dry_run:
            raise ValueError(f"{item_id} has no audio path; resolve it before inference")
        if args.dry_run:
            result = {"response_text": "", "dry_run": True, "error": "dry_run_not_inference"}
        else:
            persist_request_start(out_path, item_id, {**item, "audio_path": audio}, provenance)
            try:
                result = call_model(prompt, audio, args.task, generation_config=dict(GENERATION_CONFIG))
                if not isinstance(result, dict) or not isinstance(result.get("response_text"), str):
                    raise ValueError("Adapter must preserve response_text as a string")
            except Exception as exc:
                result = {"response_text": "", "error": f"{type(exc).__name__}: {exc}"}
        rows.append({**item, **result, id_key: item_id, "audio_path": audio, "request_provenance": provenance})
        write_atomic_jsonl(out_path, rows)


if __name__ == "__main__":
    main()
