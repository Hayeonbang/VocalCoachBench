"""Deterministic parsers used by the 2026-10-26 aligned leaderboard.

Main and Simple share the historical category parser. Quality scores deliberately
use a different parser: Main requires an explicit JSON quality field, while the
separate Simple score surface also accepts an entire scalar response.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from .taxonomy import CATEGORIES, normalize_category, normalize_category_list

PROTOCOLS = ("main", "simple")
TASKS = ("direct_pairwise", "top3_score", "segment", "score")
_ALIASES = {
    "INTONATION": "PITCH", "TUNING": "PITCH", "TIMING": "RHYTHM",
    "TEMPO": "RHYTHM", "ARTICULATION": "DICTION", "PRONUNCIATION": "DICTION",
    "BREATHING": "BREATH", "SUPPORT": "BREATH", "PHONATION": "VOCALIZATION",
    "VOCALISATION": "VOCALIZATION", "TONE": "VOCALIZATION", "PLACEMENT": "TECHNIQUE",
    "STYLE": "EXPRESSION", "MUSICALITY": "EXPRESSION",
}
_KO_TO_EN = {"음정": "PITCH", "리듬": "RHYTHM", "발음": "DICTION", "호흡": "BREATH",
             "발성": "VOCALIZATION", "테크닉": "TECHNIQUE", "표현": "EXPRESSION"}
_RAW_KEYS = ("response_text", "response", "text", "raw_output")
_PREDICTION_KEYS = (
    "top3_issues", "quality_score_0_5", "qualityScore_0_5", "qualityScore", "score",
    "quality_score", "winner", "category", "prediction", "confidence", "score_confidence",
)


def _response_text(row: dict[str, Any]) -> str:
    for key in _RAW_KEYS:
        if key in row:
            value = row[key]
            if not isinstance(value, str):
                raise ValueError(f"{key} must be text, not an extracted object or null")
            return value
    return ""


def _prediction_text(row: dict[str, Any]) -> str:
    if any(key in row for key in _RAW_KEYS):
        return _response_text(row)
    # Backwards compatibility for explicitly structured prediction rows. Never
    # rescue a failed raw response using separately supplied parsed fields.
    return json.dumps({key: row[key] for key in _PREDICTION_KEYS if key in row})


def _category_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _category(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^A-Z_ ]+", " ", str(value).strip().upper())
    text = re.sub(r"\s+", " ", text).strip()
    if text in CATEGORIES:
        return text
    if text in _ALIASES:
        return _ALIASES[text]
    for token in [*CATEGORIES, *_ALIASES]:
        if re.search(rf"\b{re.escape(token)}\b", text):
            return _ALIASES.get(token, token)
    return None


def _stored_category(value: Any) -> str | None:
    return normalize_category(_KO_TO_EN.get(str(value).strip(), value))


def _stored_categories(values: Any) -> list[str]:
    return normalize_category_list([_stored_category(value) for value in values], limit=3) if isinstance(values, list) else []


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return re.split(r"[,;/\n]+|\s+\d+[.)]\s*", str(value))


def parse_category_list(text: str, *, limit: int = 3) -> list[str]:
    """Preserve the final historical category parser, including text fallback."""
    parsed = _category_json(text)
    candidates: list[Any] = []
    if parsed:
        for key in ("top3_issues", "issues", "categories", "prediction", "labels"):
            if key in parsed:
                candidates.extend(_flatten(parsed[key]))
                break
    if not candidates:
        candidates.extend(_flatten(text))
    output: list[str] = []
    for value in candidates:
        category = _category(value)
        if category and category not in output:
            output.append(category)
            if len(output) >= limit:
                return output
    for token in [*CATEGORIES, *_ALIASES]:
        if re.search(rf"\b{re.escape(token)}\b", text.upper()):
            category = _category(token)
            if category and category not in output:
                output.append(category)
                if len(output) >= limit:
                    break
    return output


def parse_category(text: str) -> str | None:
    parsed = _category_json(text)
    if parsed:
        for key in ("category", "prediction", "main_problem", "issue"):
            category = _category(parsed.get(key))
            if category:
                return category
    values = parse_category_list(text, limit=1)
    return values[0] if values else None


def _score_json(text: str) -> dict[str, Any] | None:
    """Port the explicit scorer's JSON wrapper handling without score coercion."""
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.I | re.S)
    if fence:
        stripped = fence.group(1).strip()

    def attempt(candidate: str) -> tuple[bool, dict[str, Any] | None]:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            return False, None
        if isinstance(value, dict):
            return True, value
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            return True, value[0]
        return True, None

    def variants(candidate: str) -> list[str]:
        output = [candidate]
        if '\\"' in candidate or "\\n" in candidate:
            unescaped = candidate.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
            if unescaped != candidate:
                output.append(unescaped)
        return output

    for candidate in variants(stripped):
        found, value = attempt(candidate)
        if found:
            return value
    start = min((pos for pos in (stripped.find("{"), stripped.find("[")) if pos >= 0), default=-1)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if start >= 0 and end > start:
        for candidate in variants(stripped[start:end + 1]):
            found, value = attempt(candidate)
            if found:
                return value
    return None


def explicit_quality_score(text: str) -> float | None:
    parsed = _score_json(text)
    if not isinstance(parsed, dict):
        return None
    value = next((parsed[key] for key in ("quality_score_0_5", "qualityScore_0_5", "qualityScore")
                  if parsed.get(key) is not None), None)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str):
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        value = value.strip()
        if re.fullmatch(number, value):
            value = float(value)
        else:
            ratio = re.fullmatch(rf"({number})\s*(?:/|out\s+of)\s*({number})", value, flags=re.I)
            if not ratio or float(ratio.group(2)) != 5:
                return None
            value = float(ratio.group(1))
    return float(value) if 0 <= value <= 5 and math.isfinite(value) else None


def simple_score(text: str) -> float | None:
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s*(?:/|out\s+of)\s*5)?\s*", text, re.I)
    value = float(match.group(1)) if match else None
    return value if value is not None and 0 <= value <= 5 and math.isfinite(value) else None


def parse_winner(text: str) -> str | None:
    parsed = _category_json(text)
    candidate = str((parsed or {}).get("winner") or "").strip().upper()
    if candidate in {"A", "AUDIO A"}:
        return "A"
    if candidate in {"B", "AUDIO B"}:
        return "B"
    stripped = text.strip()
    first = stripped.splitlines()[0].strip().upper() if stripped else ""
    if first in {"A", "B", "AUDIO A", "AUDIO B"}:
        return "A" if first in {"A", "AUDIO A"} else "B"
    match = re.search(r"\b(?:winner|better|choose|selected)\b[^AB]*(AUDIO\s+)?([AB])\b", stripped, flags=re.I)
    return match.group(2).upper() if match else None


def _base(row: dict[str, Any], protocol: str, id_key: str) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    out = dict(row)
    for key in _PREDICTION_KEYS:
        out.pop(key, None)
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    candidates = (id_key, "sample_id", "recording_id", "audio_id", "id") if id_key != "pair_id" else ("pair_id", "canonical_pair_id", "task_id")
    value = next((row.get(key) or task.get(key) for key in candidates if row.get(key) or task.get(key)), None)
    if value is None:
        raise ValueError(f"Missing {id_key}; filename-only identity resolution is not supported")
    out[id_key] = value
    for key in ("audio_a_id", "audio_b_id"):
        if task.get(key) and not out.get(key):
            out[key] = task[key]
    out["response_text"] = _response_text(row)
    out["normalization_protocol"] = protocol
    return out


def _extras(out: dict[str, Any], parsed: dict[str, Any] | None, confidence_key: str) -> None:
    if not parsed:
        return
    for key in (confidence_key, "rationale", "score_rationale"):
        value = parsed.get(key)
        if key == confidence_key:
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (ValueError, TypeError, OverflowError):
                continue
            if math.isfinite(number) and 0 <= number <= 1:
                out[key] = number
        elif value:
            out[key] = str(value)


def normalize_direct_pairwise_row(row: dict[str, Any], *, protocol: str = "main") -> dict[str, Any]:
    out = _base(row, protocol, "pair_id")
    text = _prediction_text(row)
    out["winner"] = None if row.get("error") else parse_winner(text)
    if not row.get("error"):
        _extras(out, _category_json(text), "confidence")
    return out


def normalize_top3_score_row(row: dict[str, Any], *, protocol: str = "main", use_stored_parsed: bool = False) -> dict[str, Any]:
    out = _base(row, protocol, "audio_id")
    text = _prediction_text(row)
    if use_stored_parsed and not row.get("error") and not isinstance(row.get("parsed"), dict):
        raise ValueError("use_stored_parsed requires an unchanged parsed report object")
    out["top3_issues"] = [] if row.get("error") else (
        _stored_categories(row["parsed"].get("top3_issues")) if use_stored_parsed
        else parse_category_list(text, limit=3))
    if protocol == "main":
        out["quality_score_0_5"] = None if row.get("error") else explicit_quality_score(text)
    if not row.get("error"):
        _extras(out, _category_json(text), "score_confidence")
    return out


def normalize_score_row(row: dict[str, Any], *, protocol: str = "main") -> dict[str, Any]:
    out = _base(row, protocol, "audio_id")
    text = _prediction_text(row)
    score = None
    if not row.get("error"):
        score = simple_score(text) if protocol == "simple" else None
        if score is None:
            score = explicit_quality_score(text)
    out["quality_score_0_5"] = score
    return out


def normalize_segment_row(row: dict[str, Any], *, protocol: str = "main", use_stored_parsed: bool = False) -> dict[str, Any]:
    out = _base(row, protocol, "sample_id")
    text = _prediction_text(row)
    if use_stored_parsed and not row.get("error") and not isinstance(row.get("parsed"), dict):
        raise ValueError("use_stored_parsed requires an unchanged parsed report object")
    out["category"] = None if row.get("error") else (
        _stored_category(row["parsed"].get("segment_category")) if use_stored_parsed else parse_category(text))
    if not row.get("error"):
        _extras(out, _category_json(text), "confidence")
    return out


def normalize_rows(task: str, rows: list[dict[str, Any]], *, protocol: str = "main", use_stored_parsed: bool = False) -> list[dict[str, Any]]:
    normalizers = {"direct_pairwise": normalize_direct_pairwise_row, "top3_score": normalize_top3_score_row,
                   "segment": normalize_segment_row, "score": normalize_score_row}
    if task not in normalizers:
        raise ValueError(f"Unsupported task: {task}. Expected one of: {', '.join(TASKS)}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    if use_stored_parsed and task not in {"top3_score", "segment"}:
        raise ValueError("Stored parsed reports apply only to Top-3 and Segment")
    id_key = "pair_id" if task == "direct_pairwise" else "sample_id" if task == "segment" else "audio_id"
    seen: set[str] = set()
    output = []
    kwargs: dict[str, Any] = {"protocol": protocol}
    if use_stored_parsed:
        kwargs["use_stored_parsed"] = True
    for row in rows:
        result = normalizers[task](row, **kwargs)
        key = str(result[id_key])
        if key in seen:
            raise ValueError(f"Duplicate {id_key}: {key}; retries must not replace the first response")
        seen.add(key)
        output.append(result)
    return output


def category_vocabulary() -> list[str]:
    return list(CATEGORIES)
