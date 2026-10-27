"""Identity checks for one frozen first response per evaluation item."""
from __future__ import annotations

from typing import Any, Iterable


def row_identity(row: dict[str, Any], id_keys: tuple[str, ...]) -> str:
    fields = {key: str(row[key]).strip() for key in id_keys if row.get(key) is not None and str(row[key]).strip()}
    # Prepared metadata carries a bare audio_id and a dataset-prefixed
    # recording_id. The latter is descriptive metadata when a primary ID is
    # present, not a second conflicting prediction ID.
    recording_id = fields.pop("recording_id", None)
    values = list(fields.values())
    if not values and recording_id is not None:
        values = [recording_id]
    if not values:
        raise ValueError(f"Missing prediction ID; expected one of {', '.join(id_keys)}")
    if len(set(values)) != 1:
        raise ValueError(f"Conflicting prediction ID fields: {dict((key, row[key]) for key in id_keys if key in row)}")
    if recording_id is not None and recording_id != values[0]:
        if not (recording_id.startswith(("A_", "B_")) and recording_id[2:] == values[0]):
            raise ValueError(f"Conflicting prediction ID fields: recording_id {recording_id!r} does not identify {values[0]!r}")
    return values[0]


def index_predictions(
    rows: Iterable[dict[str, Any]],
    *,
    id_keys: tuple[str, ...],
    expected_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reject retries and identity mistakes instead of selecting a later success.

    An omitted expected item remains missing and is scored as an error by the
    caller. An unknown or duplicate ID is a malformed input file, not an omitted
    benchmark item, and must be fixed at source before a score can be reported.
    """
    indexed: dict[str, dict[str, Any]] = {}
    for line, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row {line} must be an object")
        identity = row_identity(row, id_keys)
        if identity in indexed:
            raise ValueError(f"Duplicate prediction ID {identity!r} at row {line}; preserve the original first response, including errors")
        if expected_ids is not None and identity not in expected_ids:
            raise ValueError(f"Unknown prediction ID {identity!r} at row {line}")
        indexed[identity] = row
    return indexed


def has_error(row: dict[str, Any]) -> bool:
    return bool(row.get("error")) or str(row.get("status", "")).lower() in {"error", "failed", "failure"}
