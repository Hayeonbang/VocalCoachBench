# VocalCoachBench

Evaluation toolkit for **VocalCoachBench: Benchmarking Audio-Language Models on
Expert Feedback for Singing**, accepted to **NeurIPS 2026, Evaluations & Datasets
Track**.

Hayeon Bang, Hounsu Kim, Wonil Kim, and Juhan Nam

[Paper](https://arxiv.org/abs/2609.04241) ·
[Dataset](https://huggingface.co/datasets/Hayeonbang/VocalCoachBench) ·
[Demo and leaderboard](https://hayeonbang.github.io/VocalCoachBench/)

Version **0.2.0** aligns the public scorer, raw-response parsers and frozen prompts
with the final 2026-10-26 leaderboard. It includes deterministic structured
metrics, offline replay of saved Claims judge outcomes, and an optional explicit
judge runner. Scoring saved outputs makes no model or judge API calls.

## Evaluation scope

Dataset version: `canonical515_physical_audio_1064refs`.

| Evaluation | Reference population | Main outputs |
|---|---|---|
| Top-3 | 515 recordings (A=207, B=308), 1,064 expert annotations | Seven-class macro-F1; F1@3 and nDCG@3 diagnostics |
| Segment | Frozen 262 consensus events | Seven-class macro-F1 and any-match accuracy |
| Ranking | 189 triplets, 567 pairs, 1,134 reference attempts | Direct and score-derived pairwise accuracy |
| Claims | 515 recordings, 6,162 diagnosis claims, 2,575 prescription slots | Strict hit, hit, valid and end-to-end rates |

The policy is `strict_all_attempts`: preserve the first stored model response,
keep failures and missing responses in the reference denominator, and reject
duplicate or unknown prediction IDs. Top-3 references receive equal total weight
per recording. The scorer cannot establish first-attempt provenance from an
already filtered prediction file; retain the original response logs.

Generation prompts use `PHONATION`; annotation/scorer label `VOCALIZATION` is its
equivalent alias. Main and Simple prompts/parsers are separate protocols.
The Segment release remains the frozen 262-event set used for the leaderboard.

## Install and prepare

From a source checkout, use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Download dataset revision `v2026.10.26` from the
[Hugging Face dataset](https://huggingface.co/datasets/Hayeonbang/VocalCoachBench/tree/v2026.10.26),
then prepare references:

```bash
python scripts/prepare_hf_release.py --hf-root /path/to/downloaded_dataset --out-dir data
```

Preparation verifies canonical annotation hashes and writes a receipt alongside
the references. Both a direct `annotations/` layout and the older nested
`VocalCoachBench_annotations/` layout are accepted. The release contains full
audio paths for 191 of 515 recordings and clips for 166 of 262 segments; resolve
the remaining audio from the original datasets before inference.

## Score predictions

For a small smoke test:

```bash
python scripts/evaluate_all.py --data-dir examples --predictions-dir examples --output-dir outputs/example_metrics
```

For canonical benchmark predictions:

```bash
python scripts/evaluate_all.py --data-dir data --predictions-dir predictions --output-dir results --require-canonical --details
```

Use `top3_predictions.jsonl`, `segment_predictions.jsonl`,
`score_predictions.jsonl`, and `direct_pairwise_predictions.jsonl`, or pass their
paths explicitly. The report identifies unavailable tasks. Missing rows within
a supplied task remain errors. Individual commands are also available through
`vocalcoachbench --help`.

Raw model responses require the matching parser, for example:

```bash
python scripts/postprocess_predictions.py --task top3_score --protocol main --input raw_outputs/top3.jsonl --out predictions/top3_predictions.jsonl
```

Keep `audio_id` values from the prepared manifest. Segment predictions use
`sample_id`; Direct predictions use `pair_id` and record presented audio order.
See [data formats](docs/data_format.md) and [postprocessing](docs/postprocessing.md).

## Claims

Replay saved judge outcomes offline:

```bash
vocalcoachbench evaluate-claims --hf-root /path/to/downloaded_dataset --samples claims/model_samples.jsonl --jobs claims/jobs.jsonl --outcomes claims/outcomes.jsonl --out results/claims.json
```

This validates references, IDs, request hashes and judge outcomes before applying
the fixed denominators. Unresolved judge errors stop scoring. See
[Claims preparation and judging](docs/claims.md) for new submissions and the
optional runner. Replaying stored decisions is deterministic; new judge
inference is a separate experiment and may produce different decisions.

## Documentation

- [Dataset preparation and audio access](docs/huggingface_dataset.md)
- [Frozen prompts and inference templates](docs/prompts.md)
- [Prediction formats](docs/data_format.md)
- [Submission workflow](docs/submission.md)
- [Reproduction scope and validation](docs/reproducing_paper_results.md)

The source checkout includes `scripts/`, `prompts/`, `examples/` and tests. The
Python package contains the scorers and Claims runner; it has no required
third-party runtime dependencies for offline scoring.
