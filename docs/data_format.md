# Data Format

All public files use UTF-8 JSONL: one JSON object per line.

The scorer consumes reference JSONL files and prediction JSONL files. It does
not run model inference. Audio files are used upstream to generate predictions;
the metric commands themselves do not load audio.

`raw_outputs/*.jsonl` files, when used, are optional logs of unprocessed model
text. They are not scored directly. Convert them to canonical prediction JSONL
with `scripts/postprocess_predictions.py`, or write canonical prediction JSONL
directly from your inference code.

## Audio Metadata

Audio metadata should identify the audio without assuming a fixed storage
backend.

```json
{"audio_id": "audio_001", "path": "audio/audio_001.wav", "song_id": "song_12"}
```

`path` can be local or relative to the benchmark root; adapters may additionally
support remote audio. Prepared metadata retains the full `recording_id` and
its scorer-facing `audio_id`. Keep their mapping from the inventory.
If `path` is `null`, the scorer can still use the reference row, but inference
code must resolve the audio from source metadata or a local audio store before
calling a model.

When using the Hugging Face dataset release, create these scorer-ready
metadata/reference files with `scripts/prepare_hf_release.py`. See
`docs/huggingface_dataset.md` for the expected downloaded layout.
For Top-3 and score inference, use the prepared `top3_audio_metadata.jsonl`
because it is keyed to `top3_references.jsonl`. Both metadata files contain the
same canonical 515 recording identities. Canonical CLI scoring resolves full
recording IDs through this inventory, without filename-only joins.

## Direct Pairwise Triplet Ranking

The benchmark releases a pair manifest. Each triplet produces three pair rows:
AB, AC, and BC. Models submit a winner for each row.

Manifest row:

```json
{
  "pair_id": "triplet_001_ab",
  "triplet_id": "triplet_001",
  "triplet_instance_id": "triplet_001_refset",
  "audio_a_id": "audio_001_a",
  "audio_b_id": "audio_001_b",
  "reference_orders": [
    ["audio_001_a", "audio_001_b", "audio_001_c"],
    ["audio_001_a", "audio_001_c", "audio_001_b"]
  ]
}
```

Prediction row:

```json
{"pair_id": "triplet_001_ab", "winner": "A", "confidence": 0.82}
```

When presented audio order differs from the manifest, include `audio_a_id` and
`audio_b_id` in the prediction. The winner is relative to this presented order;
the scorer checks the pair identities and converts to manifest order. If both
IDs are omitted, the prediction is interpreted in manifest order.

The evaluator computes pairwise accuracy, Kendall's tau, exact ranking accuracy,
cycle-or-tie rate, and missing-pair rate. If multiple expert reference orders
exist for a triplet, the model is compared against each reference order and the
scores are averaged.

The main triplet set excludes heavily conflicting triplet references. It keeps
triplets whose expert pair directions agree on at least two of the three implied
pairwise preferences.
In the Hugging Face release this policy is already encoded in
`annotations/triplet_ranking_eval_units.jsonl` and
`metadata/triplet_ranking_policy.json`; `scripts/prepare_hf_release.py` uses
those fields directly when preparing `triplet_pairs.jsonl`.

## Auxiliary Score-Derived Triplet Ranking

The auxiliary score-derived metric uses the same triplet manifest, but ranks
the three audios by single-audio scalar quality predictions.

Score prediction row:

```json
{"audio_id": "audio_001_a", "quality_score_0_5": 4.1}
```

Scores must be finite numbers in [0, 5]. Boolean, non-finite, out-of-range,
missing and explicit null values are errors. Tied scores earn no pair credit.
Main scores come from the Top-3 response; Simple scores use a separate prompt.
Score ranking is a separate surface from Direct pairwise ranking.

## Top-3 Issue Prediction

Reference row:

```json
{"audio_id": "audio_001", "references": [["PITCH", "BREATH", "VOCALIZATION"]]}
```

Prediction row:

```json
{"audio_id": "audio_001", "top3_issues": ["PITCH", "VOCALIZATION", "DICTION"]}
```

Metrics are averaged over reference lists within each recording, giving each
recording equal total weight. Headline `macro_f1` is the unweighted average of
the seven per-class F1 values; absent classes have F1 zero. `f1_at_3` is a separate
sample/reference-level overlap metric. Reference-ranked nDCG uses gains 3/2/1.

## Segment-Level Issue Classification

Reference row:

```json
{"sample_id": "segment_001", "labels": ["PITCH", "VOCALIZATION"]}
```

Prediction row:

```json
{"sample_id": "segment_001", "category": "PITCH"}
```

`match` credits any accepted expert label. `macro_f1` evaluates
all seven classes, with all accepted reference labels as positives and a single
predicted class. A missing prediction remains in both denominators.

## Categories

The seven issue categories are:

```text
PITCH, RHYTHM, DICTION, BREATH, VOCALIZATION, TECHNIQUE, EXPRESSION
```

Frozen generation prompts use `PHONATION`, which the scorer maps to the same
class as historical annotation label `VOCALIZATION`.

## Failures and provenance

Each prediction file must contain at most one row per expected ID. Duplicate,
unknown, missing or conflicting ID fields cause an input error; omitted expected
rows remain scoring errors. An `error` field or failure `status` prevents valid
looking output fields from reviving a failed attempt. Keep original responses
and request metadata. Raw JSONL normalization is described in
[postprocessing](postprocessing.md); Claims schemas are in [Claims](claims.md).

CLI reports include source paths and SHA-256 hashes. Use `--require-canonical`
with the complete prepared directory to enforce the pinned release. Example
fixtures are explicitly reported as `custom_reference_set`.
