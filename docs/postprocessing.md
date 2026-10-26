# Post-processing raw responses

Use `scripts/postprocess_predictions.py` to convert archived first-response text
to scorer fields. The default is the **Main** protocol. The pipeline is:

```text
raw response JSONL -> deterministic post-processing -> prediction JSONL -> metric commands
```

Keep both files. Post-processing preserves raw response text, identifiers,
presented Direct A/B identities, source receipts and error metadata. Duplicate
IDs raise an error; a later successful retry cannot replace an earlier failure.
Missing predictions remain errors in the scorer's reference denominator under
`strict_all_attempts`. No model or judge is called during post-processing.

```bash
python scripts/postprocess_predictions.py \
  --task top3_score --protocol main --input raw/main_top3.jsonl \
  --out predictions/main_top3.jsonl --print-summary

python scripts/postprocess_predictions.py \
  --task segment --protocol simple --input raw/simple_segment.jsonl \
  --out predictions/simple_segment.jsonl --print-summary

python scripts/postprocess_predictions.py \
  --task score --protocol simple --input raw/simple_score.jsonl \
  --out predictions/simple_score.jsonl --print-summary

python scripts/postprocess_predictions.py \
  --task direct_pairwise --protocol main --input raw/direct.jsonl \
  --out predictions/direct.jsonl --print-summary
```

Simple Top-3 and Simple score require separate responses. `--task top3_score
--protocol simple` extracts only categories; `--task score --protocol simple`
extracts the separate quality score. Main Top-3 extracts both.

## Parsing policy

Top-3 and Segment use the deterministic category parser used in final aligned
scoring: JSON fields are read first, with alias and free-text fallbacks. For
example, `BREATH, PHONATION, PITCH` preserves that order. Invalid JSON with
recognizable category names can still yield categories under this frozen
policy; JSON syntax failure alone is not the structured-task failure metric.
An explicit `error` always invalidates the row, including any otherwise
parseable categories, winner or score. An empty raw response is not rescued by
separately supplied convenience prediction fields.

Main quality scoring accepts only an explicit `quality_score_0_5`,
`qualityScore_0_5`, or `qualityScore` field in a JSON object, using the aligned
scorer's existing fence, substring, escape and singleton-array handling. Values
must be finite numbers in `[0, 5]`, numeric strings, or ratios with denominator
5. Booleans, out-of-range values, arbitrary prose numbers, `score` fields and
non-finite values fail; values are never clipped. A separate Simple score
response can additionally be an entire scalar such as `3.5` or `3.5/5`.

Direct winners refer to the **presented** Audio A/B order. The scorer maps them
back to the canonical pair using `audio_a_id` and `audio_b_id`; preserve these
fields even when the presented order is reversed. Unparsed winners and scores
are written as explicit `null`, which prevents a scorer from reparsing and
rescuing a failed prediction. The public parser fixes the historical bare
`Audio B` first-line bug. An audit of all 6,804 stored Direct responses from
12 published models found no affected non-error response, so this fix does not
change the final leaderboard.

## Source rows and compatibility

```json
{"audio_id":"recording_001","response_text":"{\"top3_issues\":[\"PITCH\",\"BREATH\",\"PHONATION\"],\"quality_score_0_5\":3.2}"}
{"sample_id":"segment_001","response_text":"PHONATION"}
{"pair_id":"pair_001","audio_a_id":"recording_a","audio_b_id":"recording_b","response_text":"{\"winner\":\"B\"}"}
```

`response`, `text` and `raw_output` are supported text aliases. Recording IDs
can be provided as `audio_id`, `recording_id` or `sample_id`; nested task metadata
is preserved. Resolve IDs against the canonical inventory before scoring; do
not resolve Dataset-B collisions by filename. Already structured prediction
rows without a raw-text field remain supported.

`--use-stored-parsed` is an explicit compatibility option for pinned historical
Top-3/Segment report artifacts with an unchanged `parsed` object. It reads
`parsed.top3_issues` or `parsed.segment_category`, while scores still use raw
text. Before enabling it, verify the report's source hash and raw-text match.
It is not a repair step or permission to generate a new model-assisted parse.
Claims use their separately documented extraction and judge pipeline.
