# Hugging Face Dataset Usage

Download [Hayeonbang/VocalCoachBench, revision v2026.10.26](https://huggingface.co/datasets/Hayeonbang/VocalCoachBench/tree/v2026.10.26).
The public snapshot has `annotations/`, `metadata/`, `evaluation/` and `audio/`
at its root. The older nested `VocalCoachBench_annotations/` layout below is
also accepted:

```text
VocalCoachBench_annotations/
  README.md
  annotations/
    atomic_claims.jsonl
    feedback_blocks.jsonl
    recordings.jsonl
    segment_annotations.jsonl
    segment_consensus_events.jsonl
    source_datasets.jsonl
    triplet_ranking_eval_units.jsonl
    top3_issue_annotations.jsonl
    triplet_rankings.jsonl
  metadata/
    CANONICAL_EVALUATION.md
    top3_join_policy.json
    category_ontology.json
    triplet_ranking_policy.json
    release_stats.json
  audio/
    diverse_song/
    segment_clips_iou03/
```

Prepare structured references from the raw annotations. The script checks the
six pinned annotation hashes and canonical 515/1,064 coverage before writing:

```bash
python scripts/prepare_hf_release.py \
  --hf-root /path/to/downloaded_hf_release \
  --out-dir data
```

The script accepts either `/path/to/downloaded_hf_release` or
`/path/to/downloaded_hf_release/VocalCoachBench_annotations`.
By default it also creates `data/audio` as a symlink to the downloaded release's
`audio/` directory, so metadata paths point to files under `data/audio/...`.
Use `--no-link-audio` to disable this behavior.

After preparation, the evaluator checkout should contain:

```text
data/
  audio_metadata.jsonl
  top3_audio_metadata.jsonl
  triplet_pairs.jsonl
  top3_references.jsonl
  segment_references.jsonl
  segment_metadata.jsonl
  prepare_summary.json
  audio/
    ...
```

Keep all generated JSONL files and the receipt together for canonical scoring.
The audio inventory validates prediction IDs even when scoring saved responses;
audio waveforms themselves are required only for new model inference.

For model inference, use the paths in the prepared metadata files. Use
`data/top3_audio_metadata.jsonl` for Top-3 issue prediction and score-derived
triplet inference, and `data/segment_metadata.jsonl` for segment inference.
`data/audio_metadata.jsonl` mirrors the raw `recordings.jsonl` table for general
lookup. If a row has `path: null`, that audio is not directly runnable from the
prepared release; resolve it from the source dataset metadata or from your local
audio store before running inference.
In this release, `top3_audio_metadata.jsonl` has packaged paths for 191 of
515 Top-3/score audios, and `segment_metadata.jsonl` has packaged paths for 166
of 262 segment rows; the remaining rows are still valid benchmark references but
must be resolved from the original source datasets or a local audio store before
model inference.

## Prepared Files

`triplet_pairs.jsonl`
: Pair manifest and triplet references for direct pairwise triplet ranking.
  This is generated from `annotations/triplet_ranking_eval_units.jsonl` by
  keeping rows with `benchmark_included == true`.

`top3_references.jsonl`
: Expert Top-3 issue references.

`segment_references.jsonl`
: Accepted expert labels for segment-level issue classification.

`audio_metadata.jsonl`
: Recording-level audio identifiers and paths from `annotations/recordings.jsonl`.

`top3_audio_metadata.jsonl`
: Audio identifiers and paths aligned exactly to `top3_references.jsonl`.
  Use this file for Top-3 issue prediction and score-derived triplet inference.

`segment_metadata.jsonl`
: Segment identifiers and paths for segment inference.

`prepare_summary.json`
: Canonical dataset version, source hashes, prepared-file hashes, counts and
  conversion settings. `--require-canonical` validates this receipt and the
  pinned reference content rather than trusting a row count alone.

With the default clean triplet policy, `prepare_summary.json` should report:

```json
{
  "triplet_source_file": "triplet_ranking_eval_units.jsonl",
  "kept_triplet_count": 189,
  "pair_count": 567,
  "top3_audio_count": 515,
  "segment_count": 262,
  "audio_path_count": 191,
  "audio_path_missing_count": 324,
  "top3_audio_metadata_count": 515,
  "top3_audio_path_count": 191,
  "top3_audio_path_missing_count": 324,
  "segment_path_count": 166,
  "segment_path_missing_count": 96
}
```

The raw HF release preserves all 450 triplet ranking annotations in
`annotations/triplet_rankings.jsonl`. Benchmark inclusion is machine-readable:
raw annotation rows include `benchmark_included`, and unique-triplet rows in
`annotations/triplet_ranking_eval_units.jsonl` define the 219 triplets and the
189 triplets used for evaluation. The policy is recorded in
`metadata/triplet_ranking_policy.json`.

## Quick Checks

Check that the raw HF release has the expected annotation files:

```bash
ls /path/to/downloaded_hf_release/VocalCoachBench_annotations/annotations/triplet_rankings.jsonl \
   /path/to/downloaded_hf_release/VocalCoachBench_annotations/annotations/top3_issue_annotations.jsonl \
   /path/to/downloaded_hf_release/VocalCoachBench_annotations/annotations/segment_consensus_events.jsonl
```

Prepare scorer-ready files:

```bash
python scripts/prepare_hf_release.py \
  --hf-root /path/to/downloaded_hf_release \
  --out-dir data
```

Check that the prepared files exist:

```bash
ls data/triplet_pairs.jsonl \
   data/top3_references.jsonl \
   data/top3_audio_metadata.jsonl \
   data/segment_references.jsonl
```

Inspect the first few rows:

```bash
head -2 data/triplet_pairs.jsonl
head -2 data/top3_references.jsonl
head -2 data/segment_references.jsonl
```

Run a scorer once you have prediction files:

```bash
python scripts/evaluate_all.py \
  --data-dir data \
  --predictions-dir predictions \
  --output-dir results
```

If the dataset is downloaded into another directory, either symlink it:

```bash
ln -s /path/to/prepared_data data
```

or pass the path explicitly:

```bash
python scripts/evaluate_all.py \
  --data-dir /path/to/prepared_data \
  --predictions-dir predictions \
  --output-dir results
```

## Notes

The evaluator repository intentionally does not include the full audio dataset.
The `data/` directory is ignored by git.

Dataset version is `canonical515_physical_audio_1064refs`. Never resolve the
four colliding Dataset-B filename pairs by filename alone. Use the canonical
recording inventory and physical-audio identities. Segment remains the frozen
262-event set used for the final leaderboard; see `metadata/PUBLIC_RELEASE.md`.

Use `--require-canonical` on published benchmark scores. `--triplet-policy all`
prepares a diagnostic set and cannot be reported as canonical ranking.
