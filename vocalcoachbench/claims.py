"""Portable replay and input preparation for corrected515 format-reported Claims.

Metric/normalization/prompt helpers below are ported from the frozen internal
vocalcoach_claim_judge.py, vocalcoach_prescription_validity_judge.py and
schema_penalized_openended_policy.py. No sibling repository is imported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .protocol import ANNOTATION_HASHES
from .strict import has_error

DATASET_VERSION = "canonical515_physical_audio_1064refs"
PROTOCOL = "corrected515_strict_judge_v2; format-reported; fixed-five-slots"
FIXED_SLOTS_PER_RECORDING = 5
DIAGNOSIS_LABELS = {"STRICT_HIT", "COARSE_HIT", "MISS", "CONTRADICT"}
PRESCRIPTION_LABELS = {"VALID", "WEAK", "INVALID", "VOCALLY_INAPPROPRIATE"}
HIT_LABELS = {"STRICT_HIT", "COARSE_HIT"}
BUDGETS = {"diagnosis": [2048, 4096, 4096], "correction": [4096, 8192, 8192]}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> str:
    # Match corrected515 job_sha256 serialization exactly (including spaces).
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.I | re.S)
    if match:
        stripped = match.group(1).strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(stripped[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def claim_text(claim: dict[str, Any]) -> str:
    return str(claim.get("text") or claim.get("claim") or "").strip()


def diagnosis_claims(row: dict[str, Any], claim_type: str) -> list[dict[str, Any]]:
    key = "diagnosis_claims" if claim_type == "diagnosis" else "prescription_claims"
    out: list[dict[str, Any]] = []
    for index, claim in enumerate((row.get("parsed") or {}).get(key) or [], 1):
        if not isinstance(claim, dict):
            continue
        text = claim_text(claim)
        if not text:
            continue
        out.append(
            {
                "id": str(claim.get("id") or f"m{index}"),
                "category": claim.get("category"),
                "text": text,
            }
        )
    return out


def gt_claims(row: dict[str, Any], claim_type: str) -> list[dict[str, Any]]:
    wanted_type = "issue" if claim_type == "diagnosis" else "correction"
    out: list[dict[str, Any]] = []
    for block in (row.get("gt") or {}).get("claims") or []:
        if not isinstance(block, dict):
            continue
        annotator = str(block.get("annotator_id") or "annotator")
        block_id = str(block.get("id") or f"block_{block.get('block_index') or 0}")
        for claim in block.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            if claim.get("type") != wanted_type:
                continue
            text = claim_text(claim)
            if not text:
                continue
            out.append(
                {
                    "id": f"{annotator}:{block_id}:{claim.get('id')}",
                    "annotator_id": annotator,
                    "block_id": block_id,
                    "claim_id": claim.get("id"),
                    "category": block.get("categories") or [],
                    "text": text,
                }
            )
    return out


def build_diagnosis_prompt(claim_type: str, gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "You are a strict text-only evaluator for a vocal-coaching benchmark. "
        "Compare the human ground-truth claims against the model's coaching claims. "
        "Do not use outside knowledge, do not listen to audio, and do not reward generic statements. "
        "Return valid JSON only."
    )
    task_name = "vocal problem diagnosis" if claim_type == "diagnosis" else "corrective coaching prescription"
    user = {
        "task": task_name,
        "labels": {
            "STRICT_HIT": "The model states the same concrete claim as the GT claim, allowing paraphrase.",
            "COARSE_HIT": "The model covers the same broad category or direction but misses important concrete details.",
            "MISS": "The model does not cover this GT claim.",
            "CONTRADICT": "The model says the opposite of the GT claim or gives opposite advice.",
        },
        "rules": [
            "Judge each GT claim independently.",
            "For prescription claims, compare the corrective action itself, not just the problem diagnosis.",
            "Use STRICT_HIT only when the specific content matches.",
            "Use COARSE_HIT for same parent idea with less specificity.",
            "Use CONTRADICT if any model claim directly conflicts with the GT claim.",
            "A model claim may match more than one GT claim only if it truly covers each one.",
            "If there are no relevant model claims, use MISS.",
            "Keep each reason under 25 words.",
        ],
        "ground_truth_claims": gt,
        "model_claims": pred,
        "output_schema": {
            "judgments": [
                {
                    "gt_id": "same id as a ground_truth_claims item",
                    "label": "STRICT_HIT | COARSE_HIT | MISS | CONTRADICT",
                    "matched_model_claim_ids": ["model claim ids, or empty array"],
                    "reason": "short reason",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def _model_claims(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, claim in enumerate((row.get("parsed") or {}).get(key) or [], 1):
        if not isinstance(claim, dict):
            continue
        text = claim_text(claim)
        if not text:
            continue
        item = {
            "id": str(claim.get("id") or f"m{index}"),
            "category": claim.get("category"),
            "text": text,
        }
        if claim.get("target_diagnosis_id") is not None:
            item["target_diagnosis_id"] = str(claim.get("target_diagnosis_id"))
        out.append(item)
    return out


def infer_target_diagnosis_id(prescription: dict[str, Any], diagnoses: list[dict[str, Any]]) -> str | None:
    target = prescription.get("target_diagnosis_id")
    diagnosis_ids = {str(row.get("id")) for row in diagnoses}
    if target and str(target) in diagnosis_ids:
        return str(target)
    match = re.match(r"^[pP](\d+)$", str(prescription.get("id") or ""))
    if match:
        candidate = f"d{match.group(1)}"
        if candidate in diagnosis_ids:
            return candidate
    return None


def model_prescription_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    diagnoses = _model_claims(row, "diagnosis_claims")
    diagnosis_by_id = {str(item["id"]): item for item in diagnoses}
    items: list[dict[str, Any]] = []
    for prescription in _model_claims(row, "prescription_claims"):
        target_id = infer_target_diagnosis_id(prescription, diagnoses)
        target = diagnosis_by_id.get(str(target_id)) if target_id else None
        item = dict(prescription)
        item["target_diagnosis_id"] = target_id
        item["target_diagnosis"] = target
        items.append(item)
    return items


def expert_issue_claims(row: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for block in (row.get("gt") or {}).get("claims") or []:
        if not isinstance(block, dict):
            continue
        annotator = str(block.get("annotator_id") or "annotator")
        block_id = str(block.get("id") or f"block_{block.get('block_index') or 0}")
        for claim in block.get("claims") or []:
            if not isinstance(claim, dict) or claim.get("type") != "issue" or not claim_text(claim):
                continue
            issues.append(
                {
                    "id": f"{annotator}:{block_id}:{claim.get('id')}",
                    "category": block.get("categories") or [],
                    "text": claim_text(claim),
                }
            )
    return issues


def build_prescription_prompt(items: list[dict[str, Any]], expert_issues: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "You are a strict text-only evaluator for a vocal-coaching benchmark. "
        "Judge each model prescription independently against its own target model diagnosis. "
        "Do not listen to audio. Do not require matching the expert correction exactly. "
        "Use standard safe vocal-pedagogy principles to judge whether the prescription is "
        "specific, issue-aligned, and vocally safe. Return valid JSON only."
    )
    user = {
        "task": "model prescription validity for vocal coaching",
        "labels": {
            "VALID": "The prescription is issue-aligned, specific, actionable, and pedagogically sound for its target diagnosis.",
            "WEAK": "The prescription is relevant but generic, incomplete, underspecified, or only weakly actionable.",
            "INVALID": "The prescription is missing a target diagnosis, mismatched to the target diagnosis, too vague to be useful, or pedagogically wrong.",
            "VOCALLY_INAPPROPRIATE": "The prescription encourages potentially unsafe vocal behavior, strain, pain, force, or harmful technique.",
        },
        "rules": [
            "Judge each model_prescription item independently.",
            "Use target_diagnosis as the main problem the prescription is supposed to address.",
            "Expert issue claims are context only; do not require the model prescription to match a particular expert correction.",
            "If target_diagnosis is null, label INVALID unless the prescription itself clearly identifies a safe and relevant target.",
            "VOCALLY_INAPPROPRIATE overrides other labels if the advice could encourage vocal harm.",
            "Keep each reason under 25 words.",
        ],
        "expert_issue_claims_context": expert_issues[:20],
        "model_prescriptions": items,
        "output_schema": {
            "judgments": [
                {
                    "prescription_id": "same id as a model_prescriptions item",
                    "label": "VALID | WEAK | INVALID | VOCALLY_INAPPROPRIATE",
                    "reason": "short reason",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def row_identity(row: dict[str, Any]) -> str:
    return str(row.get("recording_id") or row.get("audio_id") or "").strip()


def id_key(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return ""
    return str(value).strip()


def unique_repair_id(prefix: str, position: int, blocked: set[str]) -> str:
    base = f"__eval_{prefix}_{position}"
    candidate = base
    suffix = 1
    while candidate in blocked:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def normalize_claim_list(value: Any, prefix: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_is_list = isinstance(value, list)
    raw = value if raw_is_list else []
    candidates: list[tuple[int, dict[str, Any]]] = []
    non_object = 0
    empty_text = 0
    for raw_index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            non_object += 1
            continue
        if not claim_text(item):
            empty_text += 1
            continue
        candidates.append((raw_index, item))

    kept_candidates = candidates[:FIXED_SLOTS_PER_RECORDING]
    overflow = len(candidates) - len(kept_candidates)
    reserved_original_ids = {
        key for _, item in kept_candidates if (key := id_key(item.get("id")))
    }
    used: set[str] = set()
    kept: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    repaired_count = 0
    for kept_position, (raw_index, original) in enumerate(kept_candidates, 1):
        output = copy.deepcopy(original)
        original_id = id_key(original.get("id"))
        reason: str | None = None
        if not original_id:
            reason = "missing"
        elif original_id in used:
            reason = "duplicate"
        if reason:
            repaired_id = unique_repair_id(prefix, kept_position, used | reserved_original_ids)
            output["id"] = repaired_id
            effective_id = repaired_id
            repaired_count += 1
        else:
            effective_id = original_id
        used.add(effective_id)

        original_without_id = {key: value for key, value in original.items() if key != "id"}
        output_without_id = {key: value for key, value in output.items() if key != "id"}
        require(
            original_without_id == output_without_id,
            f"normalizer changed non-ID content at {prefix}[{raw_index}]",
        )
        kept.append(output)
        mappings.append(
            {
                "raw_index": raw_index,
                "kept_position": kept_position,
                "original_id": original.get("id"),
                "effective_id": output.get("id"),
                "repair_reason": reason,
            }
        )

    expected_prefix = "d" if prefix == "diagnosis" else "p"
    expected_ids = [f"{expected_prefix}{index}" for index in range(1, 6)]
    raw_ids = [id_key(item.get("id")) for _, item in candidates]
    audit = {
        "raw_is_list": raw_is_list,
        "raw_count": len(raw),
        "nonempty_object_count": len(candidates),
        "kept_count": len(kept),
        "missing_slots": FIXED_SLOTS_PER_RECORDING - len(kept),
        "non_object_dropped": non_object,
        "empty_text_dropped": empty_text,
        "overflow_dropped": overflow,
        "id_repairs": repaired_count,
        "raw_ids_exact": raw_is_list and len(candidates) == 5 and raw_ids == expected_ids,
        "raw_ids_unique_nonempty": len(raw_ids) == len(set(raw_ids)) and all(raw_ids),
        "mappings": mappings,
    }
    return kept, audit


def normalize_row(row: dict[str, Any], model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = row.get("parsed")
    parsed_is_object = isinstance(parsed, dict)
    source = parsed if parsed_is_object else {}
    diagnoses, diagnosis_audit = normalize_claim_list(source.get("diagnosis_claims"), "diagnosis")
    prescriptions, prescription_audit = normalize_claim_list(source.get("prescription_claims"), "prescription")

    normalized_parsed = copy.deepcopy(source)
    normalized_parsed["diagnosis_claims"] = diagnoses
    normalized_parsed["prescription_claims"] = prescriptions
    normalized = copy.deepcopy(row)
    normalized["parsed"] = normalized_parsed
    require(
        normalized.get("response_text") == row.get("response_text"),
        "normalizer changed response_text",
    )

    targets = [str(item.get("target_diagnosis_id") or "") for item in prescriptions]
    raw_relevant_exact = (
        parsed_is_object
        and diagnosis_audit["raw_ids_exact"]
        and prescription_audit["raw_ids_exact"]
        and targets == [f"d{index}" for index in range(1, 6)]
        and diagnosis_audit["non_object_dropped"] == 0
        and diagnosis_audit["empty_text_dropped"] == 0
        and diagnosis_audit["overflow_dropped"] == 0
        and prescription_audit["non_object_dropped"] == 0
        and prescription_audit["empty_text_dropped"] == 0
        and prescription_audit["overflow_dropped"] == 0
    )
    audit = {
        "model": model,
        "recording_id": row_identity(row),
        "dataset": row.get("dataset"),
        "parsed_is_object": parsed_is_object,
        "diagnosis": diagnosis_audit,
        "prescription": prescription_audit,
        "raw_relevant_schema_exact": raw_relevant_exact,
        "response_text_unchanged": True,
        "non_id_content_mutations": 0,
        "content_claims_added": 0,
    }
    return normalized, audit


def _unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"Non-object {label} row")
        identity = row.get(key)
        require(isinstance(identity, str) and bool(identity), f"Missing {label} {key}")
        require(identity not in result, f"Duplicate {label} {key}: {identity}; retries must not overwrite first responses")
        result[identity] = row
    return result


def load_claim_references(reference_root: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Rebuild all expert blocks from public JSONL, joined only by recording IDs.

    Original VOCALIZATION labels are aliased to PHONATION at evaluation time.
    All issue claims, including the released inferred claims, remain included.
    """
    root = Path(reference_root)
    paths = [root / "annotations" / f"{name}.jsonl" for name in
             ("recordings", "top3_issue_annotations", "feedback_blocks", "atomic_claims")]
    for path in paths:
        require(sha256(path) == ANNOTATION_HASHES[path.name], f"Canonical reference content mismatch: annotations/{path.name}")
    paths += [root / "metadata/CANONICAL_EVALUATION.md", root / "metadata/top3_join_policy.json"]
    policy = paths[-2].read_text(encoding="utf-8")
    require("515" in policy and "strict_all_attempts" in policy, "Canonical evaluation policy mismatch")
    join_policy = json.loads(paths[-1].read_text(encoding="utf-8"))
    require(isinstance(join_policy, dict), "Invalid Top-3 join policy")
    for key, expected in {"status": "canonical", "recording_inventory_n": 515, "top3_annotation_n": 1064,
                          "top3_recording_n": 515, "dataset_recording_counts": {"A": 207, "B": 308},
                          "join_key": "physical audio SHA-256"}.items():
        require(join_policy.get(key) == expected, f"Canonical join policy mismatch: {key}")
    inventory = _unique(read_jsonl(paths[0]), "recording_id", "recording")
    require(len(inventory) == 515, "Expected 515 physical recordings")
    require(Counter(r.get("dataset") for r in inventory.values()) == {"A": 207, "B": 308}, "Dataset count mismatch")
    top3 = read_jsonl(paths[1])
    require(len(top3) == 1064 and {r["recording_id"] for r in top3} == set(inventory), "Canonical Top-3 coverage mismatch")
    blocks = _unique(read_jsonl(paths[2]), "block_uid", "feedback block")
    atomic = _unique(read_jsonl(paths[3]), "claim_uid", "atomic claim")
    grouped = defaultdict(list)
    for claim in atomic.values():
        block = blocks.get(claim.get("block_uid"))
        require(block is not None, f"Unknown feedback block: {claim.get('block_uid')}")
        require(claim.get("recording_id") == block.get("recording_id") in inventory, "Claim/block recording mismatch")
        require(claim.get("annotator_id") == block.get("annotator_id"), "Claim/block annotator mismatch")
        grouped[block["block_uid"]].append({"id": claim["local_claim_id"], "type": claim["claim_type"],
                                         "text": claim["text"], "targets": claim.get("target_local_claim_ids", [])})
    samples = {rid: {"sample_id": rid, "dataset": row["dataset"], "gt": {"claims": []}}
               for rid, row in inventory.items()}
    for block in blocks.values():
        rid = block.get("recording_id")
        require(rid in samples, f"Unknown block recording: {rid}")
        categories = ["PHONATION" if c == "VOCALIZATION" else c for c in block["categories"]]
        samples[rid]["gt"]["claims"].append({"id": block["block_uid"], "annotator_id": block["annotator_id"],
            "block_index": block["block_index"], "categories": categories, "claims": grouped[block["block_uid"]]})
    for sample in samples.values():
        gt = gt_claims(sample, "diagnosis")
        require(len(gt) == len({g["id"] for g in gt}) and all(g["category"] for g in gt), "Invalid expert claim identities/categories")
    counts = {dataset: sum(len(gt_claims(s, "diagnosis")) for s in samples.values() if s["dataset"] == dataset)
              for dataset in ("A", "B")}
    require(counts == {"A": 2090, "B": 4072}, f"Canonical diagnosis population drift: {counts}")
    return samples, {str(p.resolve()): sha256(p) for p in paths}


def validate_judgments(obj: Any, job: dict[str, Any]) -> list[str]:
    """Strict corrected515 validator; no omitted IDs or fallback judge labels."""
    if not isinstance(obj, dict) or not isinstance(obj.get("judgments"), list):
        return ["not_judgment_object"]
    judgments = obj["judgments"]
    if any(not isinstance(j, dict) for j in judgments):
        return ["non_object_judgment"]
    diagnosis = job["phase"] == "diagnosis"
    key = "gt_id" if diagnosis else "prescription_id"
    ids = [j.get(key) for j in judgments]
    if any(not isinstance(identity, str) for identity in ids):
        return ["non_string_id"]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate_ids")
    expected = job["gt"] if diagnosis else job["pred"]
    if set(ids) != {x["id"] for x in expected}:
        errors.append("missing_or_extra_ids")
    labels = DIAGNOSIS_LABELS if diagnosis else PRESCRIPTION_LABELS
    model_ids = {p["id"] for p in job["pred"]}
    for judgment in judgments:
        if judgment.get("label") not in labels:
            errors.append("invalid_label")
        if diagnosis:
            matches = judgment.get("matched_model_claim_ids")
            if not isinstance(matches, list) or any(not isinstance(x, str) or x not in model_ids for x in matches):
                errors.append("invalid_matched_ids")
            elif judgment.get("label") in HIT_LABELS and not matches:
                errors.append("hit_without_match")
    return sorted(set(errors))


def _messages(sample: dict[str, Any], phase: str) -> list[dict[str, str]]:
    if phase == "diagnosis":
        return build_diagnosis_prompt("diagnosis", gt_claims(sample, "diagnosis"), diagnosis_claims(sample, "diagnosis"))
    return build_prescription_prompt(model_prescription_items(sample), expert_issue_claims(sample))


def _validate_outcome(outcome: dict[str, Any], job: dict[str, Any]) -> None:
    require(outcome.get("job_sha256") == digest(job), f"Judgment/input hash mismatch: {job['id']}")
    require(outcome.get("valid") is True, f"Unresolved judge failure: {job['id']}")
    errors = validate_judgments({"judgments": outcome.get("judgments")}, job)
    require(not errors, f"Invalid judgment {job['id']}: {errors}")
    attempts = outcome.get("attempts")
    require(isinstance(attempts, list), f"Missing raw judge attempt audit: {job['id']}")
    if not job["pred"]:
        require(outcome.get("no_model_claims") is True and not attempts, "Empty claims must use the deterministic no-call outcome")
        require(all(j["label"] == "MISS" and j["matched_model_claim_ids"] == [] for j in outcome["judgments"]), "Empty predictions cannot earn diagnosis credit")
        return
    require(attempts and len(attempts) <= len(job["budgets"]), f"Invalid judge attempt count: {job['id']}")
    require(not outcome.get("no_model_claims"), "Nonempty prediction marked as empty")
    for index, attempt in enumerate(attempts):
        request = attempt.get("request")
        require(isinstance(request, dict) and request.get("messages") == job["messages"], "Judge attempt messages changed")
        require(attempt.get("request_sha256") == digest(request), "Judge request hash mismatch")
        token_key = "max_tokens" if job["phase"] == "diagnosis" else "max_completion_tokens"
        require(request.get(token_key) == job["budgets"][index], "Judge token-budget drift")
        if job["phase"] == "diagnosis":
            require(request.get("temperature") == 0 and request.get("response_format") == {"type": "json_object"}
                    and request.get("extra_body", {}).get("reasoning_effort") == "low", "Diagnosis judge request policy drift")
        else:
            require("temperature" not in request and request.get("reasoning_effort") == "low", "Prescription judge request policy drift")
        errors = attempt.get("validation_errors")
        require(isinstance(errors, list), "Missing judge validation audit")
        if index < len(attempts) - 1:
            require(bool(errors), "A valid earlier judge outcome was replaced")
        else:
            require(not errors, "Final judge attempt is unresolved")
            require(attempt.get("parsed", {}).get("judgments") == outcome["judgments"], "Outcome differs from raw successful judge attempt")
            response = attempt.get("response", {})
            choices = response.get("choices", [])
            require(len(choices) == 1 and choices[0].get("finish_reason") == "stop", "Incomplete raw judge response")
            parsed = parse_json_object(choices[0].get("message", {}).get("content") or "")
            require(parsed == attempt.get("parsed"), "Saved judge parse differs from raw response")


def _validate_sample_response(sample: dict[str, Any], model: str) -> None:
    """Tie evaluated claims to the original response or declared extraction."""
    if sample.get("generation_error"):
        require(sample["parsed"] == {}, "Generation errors must retain zero claims")
        return
    provenance = sample.get("claims_parse_provenance") or {}
    if provenance.get("parser") == "stored_parsed":
        require(provenance.get("response_sha256") == digest(sample["response_text"]), "Stored extraction response changed")
        require(provenance.get("parsed_sha256") == digest(sample["parsed"]), "Stored extracted Claims changed")
        return
    if sample.get("simple_llm_normalizer"):
        extraction = sample["simple_llm_normalizer"]
        require(isinstance(extraction.get("raw_text"), str), "Missing preserved Simple extraction response")
        original = parse_json_object(extraction["raw_text"]) or {}
        normalized, _ = normalize_row({"parsed": original}, model)
        # Historical Simple postprocessing canonicalizes category containers and
        # adds schema metadata. Evaluated claim IDs/text must remain unchanged.
        for kind in ("diagnosis", "prescription"):
            signature = lambda row: [(c["id"], c["text"]) for c in diagnosis_claims(row, kind)]
            require(signature(normalized) == signature(sample), "Simple extraction claim content/IDs changed")
        targets = lambda row: [(p["id"], p["target_diagnosis_id"]) for p in model_prescription_items(row)]
        require(targets(normalized) == targets(sample), "Simple extraction prescription targets changed")
        return
    original = parse_json_object(sample["response_text"]) or {}
    normalized, _ = normalize_row({"parsed": original}, model)
    expected = normalized["parsed"] if original else {}
    require(sample["parsed"] == expected, "Parsed Claims differ from original response; declare a preserved extraction explicitly")


def evaluate_claims(samples_path: str | Path, jobs_path: str | Path, outcomes_path: str | Path,
                    *, reference_root: str | Path, model: str | None = None) -> dict[str, Any]:
    """Replay one full canonical cohort without model or judge inference.

    All 515 sample rows and 1,030 completed phase outcomes are required. Missing
    model responses must have been retained as empty parsed samples. Missing or
    invalid judge outcomes stop evaluation rather than becoming model failures.
    """
    references, pins = load_claim_references(reference_root)
    samples = _unique(read_jsonl(samples_path), "sample_id", "sample")
    jobs = _unique(read_jsonl(jobs_path), "id", "judge job")
    outcomes = _unique(read_jsonl(outcomes_path), "id", "judge outcome")
    models = {j.get("model") for j in jobs.values()}
    require(len(models) == 1 and None not in models, "Replay expects exactly one model cohort")
    inferred_model = next(iter(models))
    require(model is None or model == inferred_model, "Requested model differs from job model")
    model = inferred_model
    require(set(samples) == set(references), "Samples must cover exactly the canonical 515 recording IDs")
    expected_jobs = {f"{model}:{sid}:{phase}" for sid in samples for phase in BUDGETS}
    require(set(jobs) == set(outcomes) == expected_jobs, "Incomplete or unexpected judge job/outcome population")
    per_sample = []
    for sid, sample in sorted(samples.items()):
        require(sample.get("dataset") == references[sid]["dataset"], f"Dataset identity mismatch: {sid}")
        require(sample.get("model_tag") == model, f"Sample model identity mismatch: {sid}")
        require(isinstance(sample.get("response_text"), str) and isinstance(sample.get("parsed"), dict), f"Missing preserved response/parse: {sid}")
        _validate_sample_response(sample, model)
        # Saved judge IDs are evaluator-local. Preserve their exact layout while
        # verifying duplicate-aware expert content against the public release.
        signature = lambda row: Counter((g["text"], tuple(g["category"])) for g in gt_claims(row, "diagnosis"))
        require(signature(sample) == signature(references[sid]), f"Released expert GT content/category mismatch: {sid}")
        gt = gt_claims(sample, "diagnosis")
        pred = diagnosis_claims(sample, "diagnosis")
        items = model_prescription_items(sample)
        require(len(gt) == len({g["id"] for g in gt}), f"Duplicate GT claim IDs: {sid}")
        for name, claims in (("diagnosis", pred), ("prescription", items)):
            require(len(claims) == len({c["id"] for c in claims}) <= 5, f"Invalid emitted {name} IDs/slots: {sid}")
        for phase, payload in (("diagnosis", pred), ("correction", items)):
            job = jobs[f"{model}:{sid}:{phase}"]
            require(job.get("sample_id") == sid and job.get("phase") == phase, "Conflicting judge job identity")
            require(job.get("gt") == gt and job.get("pred") == payload, f"Sample/judge input mismatch: {job['id']}")
            require(job.get("messages") == _messages(sample, phase), f"Frozen judge prompt mismatch: {job['id']}")
            require(job.get("budgets") == BUDGETS[phase], "Judge budget policy drift")
            _validate_outcome(outcomes[job["id"]], job)
        diagnosis = outcomes[f"{model}:{sid}:diagnosis"]["judgments"]
        correction = outcomes[f"{model}:{sid}:correction"]["judgments"]
        dc, rc = Counter(j["label"] for j in diagnosis), Counter(j["label"] for j in correction)
        support = {mid for j in diagnosis if j["label"] in HIT_LABELS for mid in j["matched_model_claim_ids"]}
        by_item = {item["id"]: item for item in items}
        e2e = sum(j["label"] == "VALID" and by_item[j["prescription_id"]]["target_diagnosis_id"] in support for j in correction)
        per_sample.append({"sample_id": sid, "dataset": sample["dataset"], "diagnosis_n": len(gt),
            "fixed_rx_slots": 5, "dx_strict": dc["STRICT_HIT"], "dx_hit": dc["STRICT_HIT"] + dc["COARSE_HIT"],
            "rx_valid": rc["VALID"], "e2e": e2e, "diagnosis_labels": dict(dc), "correction_labels": dict(rc),
            "dx_emitted": len(pred), "rx_emitted": len(items), "empty_parsed": not bool(sample["parsed"])})
    names = ("dx_strict", "dx_hit", "rx_valid", "e2e")
    counts = {key: sum(p[key] for p in per_sample) for key in names}
    require(sum(p["diagnosis_n"] for p in per_sample) == 6162, "Diagnosis denominator drift")
    emitted = sum(p["rx_emitted"] for p in per_sample)
    by_dataset = {dataset: {"recordings": sum(p["dataset"] == dataset for p in per_sample),
        **{key: sum(p[key] for p in per_sample if p["dataset"] == dataset)
           for key in ("diagnosis_n", "fixed_rx_slots", *names)}} for dataset in ("A", "B")}
    pins.update({str(Path(p).resolve()): sha256(p) for p in (samples_path, jobs_path, outcomes_path)})
    return {"status": "complete", "model": model, "dataset_version": DATASET_VERSION,
            "evaluation_policy": PROTOCOL, "recordings": 515, "diagnosis_n": 6162, "fixed_rx_slots": 2575,
            "points": {key: counts[key] / (6162 if key.startswith("dx_") else 2575) for key in names},
            "counts": counts, "dataset_counts": by_dataset, "rx_emitted": emitted, "rx_missing_slots": 2575 - emitted,
            "generation_parse_failures": sum(p["empty_parsed"] for p in per_sample),
            "strict_id_label_validation": True, "raw_judge_attempts_verified": True, "unresolved_judge_errors": 0,
            "source_sha256": pins, "per_sample": per_sample}


def prepare_claims_jobs(predictions_path: str | Path, reference_root: str | Path, output_dir: str | Path,
                        *, model: str, parser: str = "structured") -> dict[str, Any]:
    """Prepare canonical judge inputs offline; never extract new Simple claims.

    structured parses original JSON text; stored_parsed explicitly reuses a
    preserved extraction. Missing recordings and generation errors emit zero
    claims. Duplicate recording attempts are rejected in both modes.
    """
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", model), "Unsafe model identifier")
    require(parser in {"structured", "stored_parsed"}, "Use structured or explicitly stored_parsed Claims")
    references, pins = load_claim_references(reference_root)
    raw = read_jsonl(predictions_path)
    # Public prepared inputs use the dataset-prefix-free recording ID as
    # audio_id. Build its exact mapping from this canonical inventory only.
    # Never infer a recording from an audio filename.
    aliases = defaultdict(set)
    for sid in references:
        aliases[sid].add(sid)
        if sid.startswith(("A_", "B_")):
            aliases[sid[2:]].add(sid)
    rows = {}
    for row in raw:
        require(isinstance(row, dict), "Non-object prediction row")
        identities = [row[k] for k in ("sample_id", "recording_id", "audio_id") if row.get(k) is not None]
        require(identities and all(isinstance(x, str) and x for x in identities), "Prediction requires canonical recording IDs")
        matches = [aliases.get(identity, set()) for identity in identities]
        require(all(matches), f"Unknown canonical recording ID in {identities}; filenames are not identities")
        candidates = set.intersection(*matches)
        require(len(candidates) == 1, f"Missing or conflicting canonical recording ID: {identities}")
        sid = next(iter(candidates))
        require(sid not in rows, f"Duplicate prediction for {sid}; keep the original attempt and audit retries separately")
        rows[sid] = row
    samples, jobs, audits = [], [], []
    for sid, reference in sorted(references.items()):
        row = rows.get(sid, {"sample_id": sid, "response_text": "", "error": "missing_output"})
        text = row.get("response_text", row.get("response", row.get("text", "")))
        require(isinstance(text, str), f"Expected original response text: {sid}")
        generation_error = row.get("error") or (f"status:{row.get('status')}" if has_error(row) else None)
        if generation_error:
            parsed = {}
        elif parser == "stored_parsed":
            require(isinstance(row.get("parsed"), dict), f"Missing preserved extracted Claims object: {sid}")
            parsed = row["parsed"]
        else:
            parsed = parse_json_object(text) or {}
        sample = copy.deepcopy(reference)
        sample.update(id=f"{model}:{sid}", model_tag=model, response_text=text, parsed=copy.deepcopy(parsed))
        sample, audit = normalize_row(sample, model)
        if not parsed:
            sample["parsed"] = {}
        sample["generation_error"] = generation_error
        sample["claims_parse_provenance"] = {"parser": parser, "response_sha256": digest(text),
                                             "parsed_sha256": digest(sample["parsed"])}
        audit.update(sample_id=sid, raw_parse_failure=not bool(parsed), source_row_sha256=digest(row),
                     generation_error=generation_error, parser=parser)
        if parser == "stored_parsed":
            audit["stored_extraction_sha256"] = digest(parsed)
            if row.get("simple_llm_normalizer"):
                sample["simple_llm_normalizer"] = copy.deepcopy(row["simple_llm_normalizer"])
        samples.append(sample)
        audits.append(audit)
        for phase, payload in (("diagnosis", diagnosis_claims(sample, "diagnosis")), ("correction", model_prescription_items(sample))):
            jobs.append({"id": f"{model}:{sid}:{phase}", "model": model, "sample_id": sid, "phase": phase,
                         "gt": gt_claims(sample, "diagnosis"), "pred": payload,
                         "messages": _messages(sample, phase), "budgets": BUDGETS[phase]})
    jobs.sort(key=lambda j: (j["phase"] != "diagnosis", j["sample_id"]))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pins[str(Path(predictions_path).resolve())] = sha256(predictions_path)
    for name, payload in ((f"{model}_samples.jsonl", samples), ("jobs.jsonl", jobs), ("normalization_audit.jsonl", audits)):
        path = root / name
        encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload)
        require(not path.exists() or path.read_text(encoding="utf-8") == encoded, f"Refusing to overwrite changed artifact: {path}")
        path.write_text(encoded, encoding="utf-8")
        pins[str(path.resolve())] = sha256(path)
    manifest = {"status": "prepared", "model": model, "dataset_version": DATASET_VERSION, "protocol": PROTOCOL,
        "recordings": 515, "diagnosis_n": 6162, "fixed_rx_slots": 2575, "jobs": 1030, "parser": parser,
        "missing_outputs": sorted(set(references) - set(rows)), "generation_parse_failures": sum(a["raw_parse_failure"] for a in audits),
        "reference_layout": "Released block/annotator IDs; saved historical judgments retain their original IDs for replay.",
        "context_token_validation": "Required with the chosen local tokenizer before judge requests.", "source_sha256": pins}
    path = root / "manifest.json"
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    require(not path.exists() or path.read_text(encoding="utf-8") == encoded, f"Refusing to overwrite changed artifact: {path}")
    path.write_text(encoded, encoding="utf-8")
    return manifest
