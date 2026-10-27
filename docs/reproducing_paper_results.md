# Reproducing the final leaderboard

Evaluator 0.2.0 targets dataset `canonical515_physical_audio_1064refs`, release
`v2026.10.26`, and the 2026-10-26 leaderboard. The pinned
[dataset release](https://huggingface.co/datasets/Hayeonbang/VocalCoachBench/tree/v2026.10.26)
includes `evaluation/leaderboard.json` with result definitions and coverage.
Follow the [evaluation workflow](submission.md) for preparation, inference,
postprocessing and scoring commands.

## Denominators and errors

Top-3 uses 515 recordings (A=207, B=308) and 1,064 expert references, with
references averaged within each recording. Segment uses the frozen 262-event
release. Direct and score ranking share 567 pairs from 189 triplets and 1,134
expert-reference attempts. Score ties and missing scores receive no pair
credit. Claims uses 6,162 diagnosis references and 2,575 fixed prescription
slots across all 515 recordings. Missing/invalid model outputs do not shrink
these denominators under `strict_all_attempts`.

Top-3 and Segment headline metrics are seven-class macro-F1. Top-3 F1@3 and
Segment any-match are separate metrics. Human expert-vs-expert Top-3 analysis
has 513 eligible recordings and is separate from model scoring.

Missing tasks appear in the combined report and must not be described as a full
benchmark result. Duplicate/unknown IDs and unresolved judge errors stop
evaluation. A normalized file alone cannot prove that an earlier failed
response was preserved; retain the original response logs.

## Reproduction scope

Saved predictions and judge decisions yield deterministic metrics. Historical
model outputs and judge outcomes are not bundled, so the reference dataset
alone cannot reproduce individual published model rows. Model inference
adapters remain user supplied. The [inference templates](prompts.md) record
requested settings, but adapters must enforce audio preprocessing, token
budgets, decoding and model revision.

The optional [Claims runner](claims.md) requires an explicit endpoint, judge
model and local tokenizer snapshot. New judge decisions may differ from saved
historical decisions. Historical expert IDs can differ from public claim UIDs;
replay preserves saved IDs and ordering and validates the exact expert
text/category multiset against the release.

See [release validation](release_validation.md) for scorer/parser regression
against stored outputs from all 13 final leaderboard models. That check used
no fresh model or judge inference.
