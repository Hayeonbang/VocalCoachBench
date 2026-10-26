from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from .io import mean
from .strict import has_error, index_predictions, row_identity
from .taxonomy import CATEGORIES, normalize_category, normalize_category_list, parentize


def normalize_winner(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "AUDIO A"}:
        return "A"
    if text in {"B", "AUDIO B"}:
        return "B"
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_winner(row: dict[str, Any]) -> tuple[str | None, str]:
    if has_error(row):
        return None, "error"
    # An explicitly normalized null is authoritative; do not salvage raw text.
    if "winner" in row:
        winner = normalize_winner(row["winner"])
        return winner, "winner_field" if winner else "none"
    text = str(row.get("response_text") or row.get("text") or "").strip()
    parsed = parse_json_object(text)
    winner = normalize_winner((parsed or {}).get("winner"))
    if winner:
        return winner, "json"

    first_line = text.splitlines()[0].strip() if text else ""
    winner = normalize_winner(first_line)
    if winner:
        return winner, "first_line"

    match = re.search(r'["\']?winner["\']?\s*[:=]\s*["\']?(A|B|Audio A|Audio B)["\']?', text, flags=re.IGNORECASE)
    if match:
        winner = normalize_winner(match.group(1))
        if winner:
            return winner, "winner_regex"

    return None, "none"


def reference_prefers(order: list[str], left: str, right: str) -> str:
    return left if order.index(left) < order.index(right) else right


def pairwise_score_against_reference(
    model_pair_winners: dict[tuple[str, str], str | None],
    reference_order: list[str],
) -> tuple[float, float]:
    hits: list[float] = []
    signs: list[int] = []
    for left_index in range(len(reference_order)):
        for right_index in range(left_index + 1, len(reference_order)):
            better = reference_order[left_index]
            worse = reference_order[right_index]
            key = tuple(sorted((better, worse)))
            winner = model_pair_winners.get(key)
            if winner is None:
                hits.append(0.0)
                signs.append(0)
            elif winner == better:
                hits.append(1.0)
                signs.append(1)
            else:
                hits.append(0.0)
                signs.append(-1)
    return float(mean(hits) or 0.0), float(mean(signs) or 0.0)


def reconstruct_order(audio_ids: list[str], model_pair_winners: dict[tuple[str, str], str | None]) -> tuple[list[str], bool, bool]:
    wins = {audio_id: 0 for audio_id in audio_ids}
    missing = False
    for left_index in range(len(audio_ids)):
        for right_index in range(left_index + 1, len(audio_ids)):
            key = tuple(sorted((audio_ids[left_index], audio_ids[right_index])))
            winner = model_pair_winners.get(key)
            if winner is None:
                missing = True
                continue
            wins[winner] += 1
    has_cycle_or_tie = len(set(wins.values())) < len(wins)
    order = sorted(audio_ids, key=lambda audio_id: (-wins[audio_id], audio_id))
    return order, has_cycle_or_tie, missing


def evaluate_direct_triplets(
    pair_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_by_id = index_predictions(
        prediction_rows, id_keys=("pair_id",), expected_ids={str(row["pair_id"]) for row in pair_rows}
    )

    rows_by_triplet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_rows:
        key = str(pair.get("triplet_instance_id") or pair.get("triplet_id"))
        rows_by_triplet[key].append(pair)

    triplet_metrics: list[dict[str, Any]] = []
    parsed_pairs = 0
    total_pairs = 0
    for triplet_key, triplet_pairs in sorted(rows_by_triplet.items()):
        audio_ids = sorted(
            {str(row.get("audio_a_id")) for row in triplet_pairs}
            | {str(row.get("audio_b_id")) for row in triplet_pairs}
        )
        if len(audio_ids) != 3:
            raise ValueError(f"Triplet {triplet_key!r} must contain exactly three distinct audio IDs")

        model_pair_winners: dict[tuple[str, str], str | None] = {}
        pair_details: list[dict[str, Any]] = []
        for pair in triplet_pairs:
            total_pairs += 1
            prediction = predictions_by_id.get(str(pair.get("pair_id")), {})
            winner, parse_mode = extract_winner(prediction)
            # A/B refers to the presented order, which may reverse the canonical
            # pair. Validate identities before translating the chosen side.
            presented = [prediction.get("audio_a_id"), prediction.get("audio_b_id")]
            canonical = [str(pair.get("audio_a_id")), str(pair.get("audio_b_id"))]
            if any(item is not None for item in presented):
                if any(item is None for item in presented) or set(map(str, presented)) != set(canonical):
                    raise ValueError(f"Direct physical pair mismatch for {pair['pair_id']!r}")
                presented = list(map(str, presented))
            else:
                presented = canonical
            pred_audio_id = None
            if winner == "A":
                pred_audio_id = presented[0]
            elif winner == "B":
                pred_audio_id = presented[1]
            if pred_audio_id:
                parsed_pairs += 1
            key = tuple(sorted((str(pair.get("audio_a_id")), str(pair.get("audio_b_id")))))
            model_pair_winners[key] = pred_audio_id
            pair_details.append(
                {
                    "pair_id": pair.get("pair_id"),
                    "audio_a_id": pair.get("audio_a_id"),
                    "audio_b_id": pair.get("audio_b_id"),
                    "pred_winner": winner,
                    "pred_winner_audio_id": pred_audio_id,
                    "winner_parse_mode": parse_mode,
                }
            )

        first = triplet_pairs[0]
        reference_orders = [
            [str(item) for item in order]
            for order in (first.get("reference_orders") or [first.get("human_order") or []])
            if order
        ]
        model_order, has_cycle_or_tie, has_missing_pair = reconstruct_order(audio_ids, model_pair_winners)
        pair_scores: list[float] = []
        tau_scores: list[float] = []
        exact_scores: list[float] = []
        for reference_order in reference_orders:
            pair_acc, tau = pairwise_score_against_reference(model_pair_winners, reference_order)
            pair_scores.append(pair_acc)
            tau_scores.append(tau)
            exact_scores.append(float((not has_missing_pair) and (not has_cycle_or_tie) and model_order == reference_order))

        triplet_metrics.append(
            {
                "triplet_instance_id": triplet_key,
                "triplet_id": first.get("triplet_id"),
                "audio_ids": audio_ids,
                "reference_orders": reference_orders,
                "model_order": model_order,
                "has_cycle_or_tie": has_cycle_or_tie,
                "has_missing_pair": has_missing_pair,
                "direct_pairwise_accuracy": mean(pair_scores),
                "direct_kendall_tau": mean(tau_scores),
                "direct_exact_accuracy": mean(exact_scores),
                "pair_details": pair_details,
            }
        )

    summary = {
        "evaluation_mode": "strict_all_attempts",
        "triplet_n": len(triplet_metrics),
        "pair_n": total_pairs,
        "parsed_pair_rate": (parsed_pairs / total_pairs) if total_pairs else None,
        "direct_pairwise_accuracy": mean([row.get("direct_pairwise_accuracy") for row in triplet_metrics]),
        "direct_kendall_tau": mean([row.get("direct_kendall_tau") for row in triplet_metrics]),
        "direct_exact_accuracy": mean([row.get("direct_exact_accuracy") for row in triplet_metrics]),
        "cycle_or_tie_rate": mean([float(row.get("has_cycle_or_tie")) for row in triplet_metrics]),
        "missing_pair_rate": mean([float(row.get("has_missing_pair")) for row in triplet_metrics]),
    }
    return summary, triplet_metrics


def _extract_score(row: dict[str, Any]) -> float | None:
    if has_error(row):
        return None
    explicit_score = False
    for key in ("quality_score_0_5", "score", "quality_score"):
        explicit_score |= key in row
        value = row.get(key)
        if value is None:
            continue
        try:
            score = float(value)
            return score if not isinstance(value, bool) and math.isfinite(score) and 0 <= score <= 5 else None
        except (TypeError, ValueError, OverflowError):
            continue

    if explicit_score:
        return None

    text = str(row.get("response_text") or row.get("text") or "").strip()
    parsed = parse_json_object(text)
    if parsed:
        return _extract_score(parsed)
    try:
        score = float(text)
        return score if math.isfinite(score) and 0 <= score <= 5 else None
    except ValueError:
        return None


def evaluate_score_triplets(
    pair_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    *,
    expected_audio_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score triplets, optionally validating scores against a full inventory.

    Scores for recordings outside the triplet subset are allowed unless an
    explicit full inventory is provided. They are reported as unused_score_n.
    """
    indexed = index_predictions(score_rows, id_keys=("audio_id", "sample_id", "recording_id"), expected_ids=expected_audio_ids)
    scores_by_audio: dict[str, float] = {}
    for audio_id, row in indexed.items():
        score = _extract_score(row)
        if audio_id and score is not None:
            scores_by_audio[audio_id] = score

    rows_by_triplet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_rows:
        key = str(pair.get("triplet_instance_id") or pair.get("triplet_id"))
        rows_by_triplet[key].append(pair)

    triplet_metrics: list[dict[str, Any]] = []
    for triplet_key, triplet_pairs in sorted(rows_by_triplet.items()):
        audio_ids = sorted(
            {str(row.get("audio_a_id")) for row in triplet_pairs}
            | {str(row.get("audio_b_id")) for row in triplet_pairs}
        )
        if len(audio_ids) != 3:
            raise ValueError(f"Triplet {triplet_key!r} must contain exactly three distinct audio IDs")

        available_scores = {audio_id: scores_by_audio.get(audio_id) for audio_id in audio_ids}
        has_missing_score = any(score is None for score in available_scores.values())
        has_score_tie = len({score for score in available_scores.values() if score is not None}) < len(
            [score for score in available_scores.values() if score is not None]
        )
        model_order = sorted(
            audio_ids,
            key=lambda audio_id: (
                -(available_scores[audio_id] if available_scores[audio_id] is not None else float("-inf")),
                audio_id,
            ),
        )

        first = triplet_pairs[0]
        reference_orders = [
            [str(item) for item in order]
            for order in (first.get("reference_orders") or [first.get("human_order") or []])
            if order
        ]

        pair_scores: list[float] = []
        tau_scores: list[float] = []
        exact_scores: list[float] = []
        for reference_order in reference_orders:
            hits: list[float] = []
            signs: list[int] = []
            for left_index in range(len(reference_order)):
                for right_index in range(left_index + 1, len(reference_order)):
                    better = reference_order[left_index]
                    worse = reference_order[right_index]
                    better_score = available_scores.get(better)
                    worse_score = available_scores.get(worse)
                    if better_score is None or worse_score is None or better_score == worse_score:
                        hits.append(0.0)
                        signs.append(0)
                    elif better_score > worse_score:
                        hits.append(1.0)
                        signs.append(1)
                    else:
                        hits.append(0.0)
                        signs.append(-1)
            pair_scores.append(float(mean(hits) or 0.0))
            tau_scores.append(float(mean(signs) or 0.0))
            exact_scores.append(float((not has_missing_score) and (not has_score_tie) and model_order == reference_order))

        triplet_metrics.append(
            {
                "triplet_instance_id": triplet_key,
                "triplet_id": first.get("triplet_id"),
                "audio_ids": audio_ids,
                "reference_orders": reference_orders,
                "scores": available_scores,
                "model_order": model_order,
                "has_score_tie": has_score_tie,
                "has_missing_score": has_missing_score,
                "score_pairwise_accuracy": mean(pair_scores),
                "score_kendall_tau": mean(tau_scores),
                "score_exact_accuracy": mean(exact_scores),
            }
        )

    summary = {
        "evaluation_mode": "strict_all_attempts",
        "unused_score_n": len(set(indexed) - {str(row[key]) for row in pair_rows for key in ("audio_a_id", "audio_b_id")}),
        "triplet_n": len(triplet_metrics),
        "score_pairwise_accuracy": mean([row.get("score_pairwise_accuracy") for row in triplet_metrics]),
        "score_kendall_tau": mean([row.get("score_kendall_tau") for row in triplet_metrics]),
        "score_exact_accuracy": mean([row.get("score_exact_accuracy") for row in triplet_metrics]),
        "score_tie_rate": mean([float(row.get("has_score_tie")) for row in triplet_metrics]),
        "missing_score_rate": mean([float(row.get("has_missing_score")) for row in triplet_metrics]),
    }
    return summary, triplet_metrics


def f1_at_3(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    return len(set(pred[:3]) & set(ref[:3])) / 3.0


def top1_accuracy(pred: list[str], ref: list[str]) -> float:
    return float(bool(pred and ref and pred[0] == ref[0]))


def ndcg_at_3(pred: list[str], ref: list[str]) -> float:
    relevance = {category: 3 - index for index, category in enumerate(ref[:3])}
    dcg = sum(relevance.get(category, 0) / math.log2(index + 2) for index, category in enumerate(pred[:3]))
    idcg = sum(score / math.log2(index + 2) for index, score in enumerate([3, 2, 1]))
    return dcg / idcg if idcg else 0.0


def set_f1(pred: list[str], ref: list[str]) -> float:
    left, right = set(pred), set(ref)
    return 2.0 * len(left & right) / (len(left) + len(right)) if left and right else 0.0


def _top3_refs_by_audio(reference_rows: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    refs: dict[str, list[list[str]]] = defaultdict(list)
    for row in reference_rows:
        audio_id = row_identity(row, ("audio_id", "sample_id", "recording_id"))
        references = row.get("references") if "references" in row else [row.get("top3_issues") or row.get("labels")]
        if not isinstance(references, list) or not references:
            raise ValueError(f"Missing Top-3 references for {audio_id!r}")
        for ref in references:
            clean = normalize_category_list(ref, limit=3)
            if len(clean) != 3:
                raise ValueError(f"Top-3 reference for {audio_id!r} must contain three distinct valid categories")
            refs[audio_id].append(clean)
    return refs


def _class_prf(counts: dict[str, Any]) -> dict[str, Any]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # Deliberately keep absent classes in the seven-class mean, with F1=0.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_top3(reference_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonical seven-class F1, with equal total weight per recording.

    Each recording's experts share weight 1. Missing/error attempts predict no
    classes and remain in all denominators. Normalized partial predictions keep
    their partial credit, as in the frozen analyzer; valid_top3 records whether
    three distinct labels were produced.
    """
    refs_by_audio = _top3_refs_by_audio(reference_rows)
    indexed = index_predictions(prediction_rows, id_keys=("audio_id", "sample_id", "recording_id"), expected_ids=set(refs_by_audio))
    preds_by_audio = {
        audio_id: [] if has_error(row) else normalize_category_list(
            row.get("top3_issues") if "top3_issues" in row else row.get("prediction"), limit=3
        )
        for audio_id, row in indexed.items()
    }
    counts = {
        category: dict(tp=0.0, fp=0.0, fn=0.0, reference_mass=0.0, pred_recordings=0,
                       reference_recordings=0, reference_annotations=0)
        for category in CATEGORIES
    }
    metrics: list[dict[str, Any]] = []
    for audio_id, refs in sorted(refs_by_audio.items()):
        pred = preds_by_audio.get(audio_id, [])
        per_ref = []
        weight = 1.0 / len(refs)
        for ref in refs:
            per_ref.append(
                {
                    "f1_at_3": f1_at_3(pred, ref),
                    "top1": top1_accuracy(pred, ref),
                    "ndcg_at_3": ndcg_at_3(pred, ref),
                    "parent_f1": set_f1(parentize(pred), parentize(ref)),
                }
            )
            for category in CATEGORIES:
                predicted, positive = category in pred[:3], category in ref[:3]
                if predicted and positive:
                    counts[category]["tp"] += weight
                elif predicted:
                    counts[category]["fp"] += weight
                elif positive:
                    counts[category]["fn"] += weight
                counts[category]["reference_mass"] += weight * float(positive)
        for category in CATEGORIES:
            counts[category]["pred_recordings"] += int(category in pred[:3])
            counts[category]["reference_recordings"] += int(any(category in ref[:3] for ref in refs))
            counts[category]["reference_annotations"] += sum(category in ref[:3] for ref in refs)
        metrics.append(
            {
                "audio_id": audio_id,
                "prediction": pred,
                "references": refs,
                "reference_count": len(refs),
                "valid_top3": len(pred) == 3,
                "f1_at_3": mean([row["f1_at_3"] for row in per_ref]),
                "top1": mean([row["top1"] for row in per_ref]),
                "ndcg_at_3": mean([row["ndcg_at_3"] for row in per_ref]),
                "parent_f1": mean([row["parent_f1"] for row in per_ref]),
            }
        )
    class_metrics = []
    for category in CATEGORIES:
        count = counts[category]
        class_metrics.append({
            "category": category, **_class_prf(count),
            "reference_mass": count["reference_mass"],
            "reference_recordings": count["reference_recordings"],
            "reference_annotations": count["reference_annotations"],
            "gt_prevalence": count["reference_mass"] / len(metrics) if metrics else 0.0,
            "prediction_frequency": count["pred_recordings"] / len(metrics) if metrics else 0.0,
        })
    summary = {
        "evaluation_mode": "strict_all_attempts",
        "n": len(metrics),
        "audio_n": len(metrics),
        "f1_at_3": mean([row.get("f1_at_3") for row in metrics]),
        "top1": mean([row.get("top1") for row in metrics]),
        "ndcg_at_3": mean([row.get("ndcg_at_3") for row in metrics]),
        "parent_f1": mean([row.get("parent_f1") for row in metrics]),
        "macro_f1": mean([row["f1"] for row in class_metrics]),
        "class_metrics": class_metrics,
        "valid_top3_rate": mean([float(row["valid_top3"]) for row in metrics]),
    }
    return summary, metrics


def evaluate_segment(reference_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonical one-vs-rest F1 across all seven segment issue categories.

    Match accepts any reference label. Macro-F1 treats every accepted label as
    positive, so choosing one label can leave another accepted label as an FN.
    """
    refs_by_id = index_predictions(reference_rows, id_keys=("sample_id", "id"))
    indexed = index_predictions(prediction_rows, id_keys=("sample_id", "id"), expected_ids=set(refs_by_id))
    preds_by_id = {
        sample_id: None if has_error(row) else normalize_category(
            row.get("category") if "category" in row else row.get("segment_category") if "segment_category" in row else row.get("prediction")
        ) for sample_id, row in indexed.items()
    }
    counts = {category: dict(tp=0, fp=0, fn=0, gt=0, pred=0) for category in CATEGORIES}
    metrics: list[dict[str, Any]] = []
    for sample_id, row in sorted(refs_by_id.items()):
        labels = normalize_category_list(row.get("labels") or row.get("category_intersection") or row.get("categories") or row.get("consensus_issues"))
        if not labels:
            raise ValueError(f"Segment reference for {sample_id!r} must contain an accepted category")
        pred = preds_by_id.get(sample_id)
        correct = float(pred in labels) if pred else 0.0
        metrics.append(
            {
                "sample_id": sample_id,
                "prediction": pred,
                "labels": labels,
                "match": correct,
                "singleton": len(labels) == 1,
            }
        )
        for category in CATEGORIES:
            predicted, positive = pred == category, category in labels
            counts[category]["tp"] += int(predicted and positive)
            counts[category]["fp"] += int(predicted and not positive)
            counts[category]["fn"] += int(not predicted and positive)
            counts[category]["gt"] += int(positive)
            counts[category]["pred"] += int(predicted)
    class_metrics = []
    for category in CATEGORIES:
        count = counts[category]
        class_metrics.append({
            "category": category, **_class_prf(count),
            "reference_events": count["gt"], "predicted_events": count["pred"],
            "reference_prevalence": count["gt"] / len(metrics) if metrics else 0.0,
        })
    summary = {
        "evaluation_mode": "strict_all_attempts",
        "n": len(metrics),
        "segment_n": len(metrics),
        "match": mean([row.get("match") for row in metrics]),
        "macro_f1": mean([row["f1"] for row in class_metrics]),
        "class_metrics": class_metrics,
        "valid_prediction_rate": mean([float(row["prediction"] is not None) for row in metrics]),
        "singleton_n": sum(row["singleton"] for row in metrics),
    }
    return summary, metrics
