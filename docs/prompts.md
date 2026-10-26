# Frozen generation prompts

The public prompts match the 2026-10-26 final leaderboard release, dataset version
`canonical515_physical_audio_1064refs`. `prompts/manifest.json` records the SHA-256
of each exact file and of its stripped runtime text. Main and Simple are separate
protocols; select the matching parser and do not pool their responses.

| Surface | Main | Simple |
|---|---|---|
| Top-3 | `prompts/main/top3_score.txt` (includes quality score) | `prompts/simple/top3.txt` |
| Score | Included in Main Top-3 | `prompts/simple/score.txt` (separate request) |
| Segment | `prompts/main/segment.txt` | `prompts/simple/segment.txt` |
| Direct ranking | `prompts/main/direct_pairwise.txt` | Model-specific concatenated-audio wrappers are not bundled |
| Claims / overall | `prompts/main/claims.txt` | `prompts/simple/overall.txt` |

Generation prompts use **PHONATION**. Released annotations and the scorer's
seven-class vocabulary retain **VOCALIZATION** as the equivalent historical
label. These are one class; do not add an eighth class or rewrite original data.
The previous top-level `*_v5.txt` filenames remain byte-identical aliases of the
corresponding files above for existing commands.

The Main Direct prompt requires separately labeled Audio A and Audio B inputs.
Their identities and presented order must be saved with every response. Do not
use it unchanged with concatenated audio: the published Audio Flamingo 3 and
Music Flamingo Direct runs used additional interval instructions. Covo-Audio
has no published Direct score.

The inference templates pass greedy settings (`do_sample=false`, `temperature=0`,
`top_p=1`) and a first-180-second audio limit to the user-supplied adapter. They
record file/runtime prompt hashes and the requested configuration. The adapter
must enforce those settings and record its model revision, system prompt,
audio preprocessing and token budget. These templates make no network calls
until `call_model` is implemented. A dry run writes requests marked as errors,
so it cannot be mistaken for completed inference.

```bash
python scripts/infer_single_audio_template.py \
  --task top3_score --protocol main --inputs data/audio_metadata.jsonl \
  --out requests/main_top3.jsonl --dry-run

python scripts/infer_single_audio_template.py \
  --task score --protocol simple --inputs data/audio_metadata.jsonl \
  --out requests/simple_score.jsonl --dry-run
```

`--resume` skips every existing attempt, including failures; it rejects duplicate
IDs and changed request provenance. Use a separate artifact for retries or a
new protocol. A custom `--prompt` is recorded as a custom prompt and is not a
claim of frozen-protocol equivalence. Prompt identity alone also does not prove
that a historical model's system prompt, audio wrapper or decoding matched.

The templates persist a request marker before calling the adapter and save
response snapshots atomically. Preserve the adjacent `.attempts/` directory
with each output file. If execution stops during a request, resume rejects the
unresolved marker; recover the original response or record its failure rather
than deleting the marker and generating another answer.
