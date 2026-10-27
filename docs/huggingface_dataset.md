# Dataset preparation

Download [Hayeonbang/VocalCoachBench, revision v2026.10.26](https://huggingface.co/datasets/Hayeonbang/VocalCoachBench/tree/v2026.10.26).
The downloaded root contains `annotations/`, `metadata/`, `evaluation/` and
`audio/`. Prepare the references with:

```bash
python scripts/prepare_hf_release.py \
  --hf-root /path/to/downloaded_dataset \
  --out-dir data
```

Preparation verifies six pinned annotation hashes and canonical coverage of
515 recordings (A=207, B=308) and 1,064 Top-3 annotations. The older nested
`VocalCoachBench_annotations/` layout is also accepted. By default, the script
creates `data/audio` as a symlink to the downloaded `audio/` directory;
`--no-link-audio` disables this step.

## Prepared files

| File | Purpose |
|---|---|
| `top3_references.jsonl` | Expert Top-3 issue references |
| `segment_references.jsonl` | Accepted labels for the frozen 262 segments |
| `triplet_pairs.jsonl` | 567 pairs from the 189 benchmark triplets |
| `top3_audio_metadata.jsonl` | Recording IDs and paths for Top-3 and score inference |
| `segment_metadata.jsonl` | Segment IDs and paths for segment inference |
| `audio_metadata.jsonl` | General recording lookup from `annotations/recordings.jsonl` |
| `prepare_summary.json` | Dataset version, source/prepared-file hashes, counts and conversion settings |

Keep these files together. Scoring validates IDs through the prepared inventory;
it does not require audio waveforms. Use `--require-canonical` when reporting
benchmark scores to validate the receipt and pinned references. Prediction
schemas and scoring commands are in [data formats](data_format.md) and the
[evaluation workflow](submission.md).

## Audio access

Use `top3_audio_metadata.jsonl` for Top-3 and score inference, and
`segment_metadata.jsonl` for segment inference. Packaged audio paths cover
**191 of 515 recordings** and **166 of 262 segments**. A `path: null` row remains
a valid benchmark reference, but its audio must be resolved from the original
source dataset metadata or a local audio store before inference. The code
repository does not bundle the audio dataset; its `data/` directory is ignored
by git.

Preserve the mapping between full `recording_id` and scorer-facing `audio_id`
in the inventory. Resolve the four colliding Dataset-B filename pairs by
physical-audio identity, never by filename alone.

## Reference scope

Dataset version is `canonical515_physical_audio_1064refs`. Segment remains the
frozen 262-event set used for the final leaderboard; the dataset's
`metadata/PUBLIC_RELEASE.md` records this scope.

The raw release retains 450 ranking annotations over 219 unique triplets.
Preparation uses `annotations/triplet_ranking_eval_units.jsonl` and its
`benchmark_included` fields to select the 189 evaluation triplets, following
`metadata/triplet_ranking_policy.json`. `--triplet-policy all` prepares a
diagnostic set and cannot be reported as canonical ranking.
