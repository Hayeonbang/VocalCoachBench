# Reproducing the final leaderboard

Evaluator 0.2.0 targets dataset `canonical515_physical_audio_1064refs`, release
`v2026.10.26`, and the 2026-10-26 leaderboard. Download the pinned
[dataset release](https://huggingface.co/datasets/Hayeonbang/VocalCoachBench/tree/v2026.10.26).
The published `evaluation/leaderboard.json` records result definitions and coverage.

## Workflow

1. Prepare canonical references with `scripts/prepare_hf_release.py`. Keep
   `prepare_summary.json` beside the generated files.
2. Generate predictions using the appropriate Main or Simple protocol. Preserve
   the first response and its error state for each manifest ID. Use exact audio
   identities, not filename-only joins.
3. Normalize raw responses with `scripts/postprocess_predictions.py` and the
   matching `--protocol`. An error response must remain an error.
4. Score structured predictions with the canonical guard:

   ```bash
   python scripts/evaluate_all.py --data-dir data --predictions-dir predictions --output-dir results --require-canonical --details
   ```

5. For Claims, supply saved samples, jobs and outcomes through `evaluate-claims`
   or the corresponding `evaluate_all.py` flags. See [Claims](claims.md).
6. Report artifact paths and hashes, dataset version, denominator, Main/Simple
   protocol and `strict_all_attempts`, together with metrics.

## Denominators and errors

Top-3 uses 515 recordings and 1,064 expert references, with references averaged
within each recording. Segment uses the frozen 262-event release. Direct and
score ranking share 567 pairs from 189 triplets and 1,134 expert-reference
attempts. Score ties and missing scores receive no pair credit. Claims uses
6,162 diagnosis references and 2,575 fixed prescription slots across all 515
recordings. Missing/invalid model outputs do not shrink these denominators.

Top-3 and Segment headline metrics are seven-class macro-F1. Top-3 F1@3 and
Segment any-match are separate metrics. Human expert-vs-expert Top-3 analysis
has 513 eligible recordings and is a separate analysis from model scoring.

Missing tasks appear in the combined report and must not be described as a full
benchmark result. Duplicate/unknown IDs and unresolved judge errors stop
evaluation. The scorer verifies supplied artifacts; a normalized file alone
cannot prove that an earlier failed response was not removed upstream.

## Reproduction boundary

Saved predictions and judge decisions yield deterministic metrics. Historical
model outputs and judge outcomes are not bundled in this code repository, so
the reference dataset alone cannot reproduce individual published model rows.
Model inference adapters remain user supplied. The inference templates record
requested settings, but adapters must actually enforce audio preprocessing,
token budgets, decoding and model revision. See [prompts](prompts.md).

The optional Claims runner requires an explicit endpoint, judge model and local
tokenizer snapshot. Running it produces new decisions, which may differ from
saved historical decisions. No endpoint or credentials are embedded in this
repository. Historical internal claim IDs can differ from public claim UIDs;
replay validates the exact reference-text/category multiset and preserves the
saved IDs and ordering.

## Maintainer regression

The release was checked against stored Open9 and Closed4 outputs from the final
leaderboard, including its Main/Simple split. The validation record is described
in [release validation](release_validation.md). This is a scorer/parser replay;
it does not represent fresh model or judge inference.
