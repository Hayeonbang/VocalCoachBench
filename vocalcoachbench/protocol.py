"""Release identity checks and provenance for the frozen public benchmark."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_VERSION = "canonical515_physical_audio_1064refs"
POLICY = "strict_all_attempts; first stored response; recording-balanced references"
ANNOTATION_HASHES = {
    "recordings.jsonl": "3f0dffa6ca5b71d7fa3ed54dabb279507b0e5e0d92cb7c6b23c4a924347b89b5",
    "top3_issue_annotations.jsonl": "72cf6dde146a7d8db11209ac164e3ab40f08afe91649768f849e2713cf87ffd5",
    "feedback_blocks.jsonl": "e2253bb98bd6d291412f39be3f533bf58e083643d514a54d848ef97b737f5c89",
    "atomic_claims.jsonl": "7b8d25acedd837e4e4a49f88a01825ea0984e7898edcdd80b96e4c53c88f93da",
    "segment_consensus_events.jsonl": "30bf9cc13478d49f2de26af77ec94141071e5426c6d07ae47d27a863f08c0c1b",
    "triplet_ranking_eval_units.jsonl": "343cb74b2bac7f2c7accc022a7c5729cf195c6a4fdc3c246dacf93b6e83a6d8e",
}
# These are deterministic scorer references prepared from the pinned release.
# A self-authored receipt must not be able to relabel a custom subset canonical.
PREPARED_REFERENCE_HASHES = {
    "top3_references.jsonl": "06458107d8381d5496d5759204206f48ca49bf5cf9232b2a0058ce0d1acf0d51",
    "segment_references.jsonl": "35ce4d5511aad4212652a320efa0bd792a47ee846aa1e858165bcbbc2d925652",
    "triplet_pairs.jsonl": "372e0e495838801804eadf098313c29ac5f670fef6fca71ff3681b58977ac324",
}
INVENTORY_ID_SHA256 = "ca5726f827c1ac17d5f85441cbdaf3c0818e052b9cc94c76afa6aa572fb0a71e"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_release(root: str | Path) -> dict:
    root = Path(root)
    policy = json.loads((root / "metadata/top3_join_policy.json").read_text())
    document = (root / "metadata/CANONICAL_EVALUATION.md").read_text()
    if (policy.get("recording_inventory_n"), policy.get("top3_annotation_n"), policy.get("top3_recording_n")) != (515, 1064, 515):
        raise ValueError("Canonical release requires 515 recordings and 1,064 Top-3 references")
    if "strict_all_attempts" not in document or policy.get("join_key") != "physical audio SHA-256":
        raise ValueError("Canonical evaluation policy is missing or incompatible")
    for name, expected in ANNOTATION_HASHES.items():
        if sha256(root / "annotations" / name) != expected:
            raise ValueError(f"Canonical reference content mismatch: annotations/{name}")
    from .io import read_jsonl
    recordings = read_jsonl(root / "annotations/recordings.jsonl")
    annotations = read_jsonl(root / "annotations/top3_issue_annotations.jsonl")
    ids = {r["recording_id"] for r in recordings}
    by_id = Counter(r["recording_id"] for r in annotations)
    if len(recordings) != len(ids) or ids != set(by_id):
        raise ValueError("Recording inventory/Top-3 identity mismatch")
    if Counter(r["dataset"] for r in recordings) != {"A": 207, "B": 308} or Counter(by_id.values()) != {1: 2, 2: 504, 6: 9}:
        raise ValueError("Canonical dataset counts/reference multiplicities differ")
    return {"dataset_version": DATASET_VERSION, "evaluation_policy": POLICY,
            "canonical_annotation_sha256": dict(ANNOTATION_HASHES),
            "recordings": 515, "top3_annotations": 1064,
            "segments": 262, "ranking_pairs": 567, "ranking_triplets": 189,
            "ranking_reference_attempts": 1134, "diagnosis_claims": 6162,
            "prescription_slots": 2575}


def scoring_provenance(reference: str | Path, prediction: str | Path, *, require_canonical: bool = False) -> dict:
    reference, prediction = Path(reference), Path(prediction)
    result = {"dataset_version": "custom_reference_set", "evaluation_policy": POLICY,
              "reference": {"path": str(reference), "sha256": sha256(reference)},
              "predictions": {"path": str(prediction), "sha256": sha256(prediction)}}
    receipt = reference.parent / "prepare_summary.json"
    if receipt.exists():
        metadata = json.loads(receipt.read_text())
        if metadata.get("dataset_version") == DATASET_VERSION:
            if metadata.get("canonical_annotation_sha256") != ANNOTATION_HASHES:
                raise ValueError("Preparation receipt does not identify the frozen canonical source")
            if metadata.get("prepared_file_sha256", {}).get(reference.name) != result["reference"]["sha256"]:
                raise ValueError("Prepared reference changed after its preparation receipt")
            if PREPARED_REFERENCE_HASHES.get(reference.name) != result["reference"]["sha256"]:
                raise ValueError("Prepared reference does not match the pinned canonical scorer input")
            if metadata.get("triplet_policy", "clean") != "clean" and reference.name == "triplet_pairs.jsonl":
                raise ValueError("All-triplet diagnostic references are not the canonical ranking set")
            result.update(dataset_version=DATASET_VERSION, preparation_receipt_sha256=sha256(receipt))
    if require_canonical and result["dataset_version"] != DATASET_VERSION:
        raise ValueError("Canonical scoring requires references and receipt from scripts/prepare_hf_release.py")
    return result


def canonical_audio_inventory(reference: str | Path) -> dict[str, str]:
    """Load all 515 audio/recording IDs, independently of local audio paths."""
    from .io import read_jsonl
    directory = Path(reference).parent
    path = directory / "audio_metadata.jsonl"
    receipt = json.loads((directory / "prepare_summary.json").read_text())
    if receipt.get("prepared_file_sha256", {}).get(path.name) != sha256(path):
        raise ValueError("Audio inventory changed after its preparation receipt")
    rows = read_jsonl(path)
    identities = sorted((str(row["audio_id"]), str(row["recording_id"])) for row in rows)
    digest = hashlib.sha256(json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if digest != INVENTORY_ID_SHA256:
        raise ValueError("Audio inventory does not match the canonical 515 recording identities")
    return dict(identities)


def canonical_recording_predictions(
    reference: str | Path, prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Resolve exact recording IDs to prepared audio IDs using the pinned map.

    Bare audio_id/sample_id values are the public scorer keys. Full recording_id
    values are accepted through this inventory mapping, never by filenames.
    """
    from .strict import row_identity
    inventory = canonical_audio_inventory(reference)
    aliases = {**{key: key for key in inventory}, **{value: key for key, value in inventory.items()}}
    output = []
    for row in prediction_rows:
        identity = row_identity(row, ("audio_id", "sample_id", "recording_id"))
        audio_id = aliases.get(identity)
        if audio_id is None:
            raise ValueError(f"Unknown prediction ID {identity!r}; not in the canonical 515 recording inventory")
        if row.get("recording_id") is not None and str(row["recording_id"]) != inventory[audio_id]:
            raise ValueError(f"recording_id does not match canonical audio_id {audio_id!r}")
        normalized = {**row, "audio_id": audio_id}
        if "sample_id" in normalized:
            normalized["sample_id"] = audio_id
        output.append(normalized)
    return output, set(inventory)
