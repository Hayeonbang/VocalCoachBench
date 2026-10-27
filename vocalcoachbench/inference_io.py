"""Durable request markers and response snapshots shared by inference templates."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """A failed persistence step must leave the previous output intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def attempt_directory(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".attempts")


def check_started_attempts(out_path: Path, existing: dict[str, dict[str, Any]], provenance: dict[str, Any]) -> None:
    """A started request without its original result must never be regenerated."""
    for marker in sorted(attempt_directory(out_path).glob("*.json")):
        started = json.loads(marker.read_text(encoding="utf-8"))
        item_id = started.get("request_id")
        if not isinstance(item_id, str) or marker.name != hashlib.sha256(item_id.encode()).hexdigest() + ".json":
            raise ValueError(f"Invalid request start marker: {marker}")
        if started.get("request_provenance") != provenance:
            raise ValueError(f"Request start marker provenance mismatch: {marker}")
        if item_id not in existing:
            raise ValueError(f"Unresolved started attempt for {item_id!r}: {marker}. "
                             "Recover its original response or persist an error row before resuming; do not regenerate it.")


def persist_request_start(out_path: Path, item_id: str, request: dict[str, Any], provenance: dict[str, Any]) -> None:
    directory = attempt_directory(out_path)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / (hashlib.sha256(item_id.encode()).hexdigest() + ".json")
    # Exclusive creation prevents a second adapter call for the same identity.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"request_id": item_id, "request": request, "request_provenance": provenance},
                  stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(directory)
    _fsync_directory(directory.parent)
