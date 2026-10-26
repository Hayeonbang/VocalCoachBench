# Submission guide

Prepare the pinned canonical dataset as described in
[dataset preparation](huggingface_dataset.md), then preserve one first-attempt
response for each expected ID. Record model revision, prompt hash, Main/Simple
protocol, audio preprocessing, decoding settings and failures. Do not replace
failed responses with successful retries in the same result artifact.

Use the formats in [data formats](data_format.md). If your adapter produces raw
text, normalize it with the matching [parser](postprocessing.md). Keep the raw
logs as provenance. Submit task-specific files:

- `top3_predictions.jsonl`: all 515 recording IDs; three ranked issue labels.
- `score_predictions.jsonl`: explicit quality scores in [0, 5], using the same
  recording inventory. Main can reuse the normalized Top-3/score output; Simple
  uses its separate score request.
- `segment_predictions.jsonl`: 262 segment IDs; one issue label each.
- `direct_pairwise_predictions.jsonl`: 567 pair IDs; A/B winner with presented
  audio identities. Missing or invalid winners receive no credit.

Run the structured scorer:

```bash
python scripts/evaluate_all.py --data-dir data --predictions-dir predictions --output-dir results --require-canonical --details
```

If Main scores are kept in the Top-3 file, additionally pass
`--scores predictions/top3_predictions.jsonl`.

For Claims, follow [Claims preparation and judging](claims.md), then add
`--claims-samples`, `--claims-jobs`, `--claims-outcomes`, and `--hf-root` to the
combined command, or use `vocalcoachbench evaluate-claims` separately. Missing
model claims retain the fixed denominators; unresolved judge outcomes prevent
a complete Claims result.

Include the output JSON reports, preparation receipt and original prediction
artifacts with a submission. State dataset version
`canonical515_physical_audio_1064refs`, protocol, denominator and
`strict_all_attempts`. The combined report explicitly lists unavailable tasks;
a partial task set is not a complete benchmark submission.
