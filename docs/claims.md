# Claims evaluation

The scorer implements the final leaderboard protocol. It uses all
**515 recordings (A=207, B=308)**,
**6,162 expert diagnosis claims (A=2,090, B=4,072)** and **2,575 prescription
slots (515 × 5)**. Dataset version: `canonical515_physical_audio_1064refs`.

| Metric | Numerator | Fixed denominator |
| --- | --- | --- |
| `dx_strict` | Expert claims labeled `STRICT_HIT` | 6,162 |
| `dx_hit` | Expert claims labeled `STRICT_HIT` or `COARSE_HIT` | 6,162 |
| `rx_valid` | Emitted prescriptions labeled `VALID` | 2,575 |
| `e2e` | `VALID` prescriptions whose target model diagnosis supports at least one expert claim with either hit label | 2,575 |

These are expert-claim recall and fixed-slot prescription metrics. The
recording-balanced expert averaging used for Top-3 is a separate metric.
Claims are not averaged by recording, and missing prescription slots never
shrink the denominator. Strength claims are not scored.

## Replay saved judgments without inference

```bash
vocalcoachbench evaluate-claims \
  --hf-root /path/to/downloaded_dataset \
  --samples saved/model_samples.jsonl \
  --jobs saved/jobs.jsonl --outcomes saved/outcomes.jsonl \
  --out results/claims.json
```

Each replay contains one model cohort. Its name is inferred from the jobs;
optional `--model` checks that it matches the requested model.

The release root must contain `annotations/recordings.jsonl`,
`top3_issue_annotations.jsonl`, `feedback_blocks.jsonl`, `atomic_claims.jsonl`,
and the canonical policy files under `metadata/`. The scorer rebuilds the
expert issue population by recording ID and block UID. It never joins raw
Dataset-B submissions by filename. `VOCALIZATION` is aliased to `PHONATION`;
the released inferred issue claims remain included in the 6,162 denominator.
The four annotation files must match the toolkit's frozen SHA-256 pins; equal
row counts do not make a modified reference set canonical.

The three replay files use these schemas:

- **Samples:** one row for every canonical recording, with `sample_id`,
  `dataset`, `model_tag`, original `response_text`, normalized `parsed`, and
  `gt.claims` containing expert blocks. Empty responses remain present.
- **Jobs:** two per sample, with `id` (`model:sample_id:phase`), `model`,
  `sample_id`, `phase` (`diagnosis` or `correction`), exact `gt`, exact `pred`,
  full `messages`, and the frozen `budgets`.
- **Outcomes:** exactly one per job, with `id`, `job_sha256`, boolean `valid`,
  `judgments`, and complete raw `attempts`. Each successful attempt includes
  its request and hash, raw response, parsed object, and validation errors.
  Empty model claims use `no_model_claims: true` with no judge attempts.

Replay checks all recording, claim, job and outcome identities; rejects
duplicates; compares expert text/category multisets with the released data;
rebuilds the exact judge prompts; checks job/request hashes; and verifies
successful judgments against the raw judge response. It also checks Main
claims against the preserved response, and historical Simple claim IDs/text
against the preserved extraction response. A missing or invalid **judge**
outcome stops evaluation; it is not counted as a model failure.

Historical evaluator-local expert IDs differ from public release UIDs. Replay
preserves the saved IDs and order, verifies all expert content with
multiplicity against the release, and requires exact sample-to-job identity.
Results contain the artifact paths and SHA-256 hashes, denominators, policy,
raw integer counts, per-recording counts, and A/B subtotals.

Historical model/judge archives are not bundled. Exact leaderboard replay
requires saved samples, jobs and outcomes.

## Prepare a new evaluation offline

```bash
vocalcoachbench prepare-claims --predictions predictions.jsonl --hf-root /path/to/downloaded_dataset --output-dir judge_inputs --model my-model
```

Each prediction row supplies a canonical `recording_id` (or `sample_id`) and
the original `response_text`. A prepared input's bare `audio_id` is resolved
through the canonical inventory and may accompany its full `recording_id`.
Conflicting identities are rejected. Optional `error`, or a `status` of
`error`, `failed`, or `failure`, marks a generation failure.
Preparation retains missing recordings and generation errors as zero claims;
it rejects repeated recording IDs rather than selecting a successful retry.
Do not submit a partial running generation as a completed benchmark result.

The frozen parser accepts JSON objects, including fenced or surrounded JSON.
The frozen normalizer retains the first five nonempty claim objects
of each evaluated type and repairs only missing or duplicate local IDs. It
never adds or rewrites claim text, categories, or target links. Malformed or
empty responses emit zero claims. `normalization_audit.jsonl` records dropped
objects, omitted slots, ID repairs and raw parsing failures.

For Simple free-form outputs, use `--parser stored_parsed` only when each row
also contains the original preserved `parsed` extraction. Preparation records
the source and extraction hashes. It does not generate a new extraction or
treat free-form prose as a structured V5 object. Retain the extraction model,
prompt and raw extraction response with the experiment artifacts.

Prescription targets preserve the frozen implementation: use an explicit
target when it names an emitted diagnosis; otherwise `pN` may resolve to `dN`
when that diagnosis exists. This existing fallback affects E2E and is not a
newly generated diagnosis. A valid prescription can earn validity credit
without an expert-supported target, but cannot then earn E2E credit.

New inputs use public block/annotator UIDs and release order, which can differ
from historical judge message IDs/order despite preserving expert content.

## Optional judge calls

Install the optional runtime packages before calling a judge:

```bash
python -m pip install openai transformers
python -m vocalcoachbench.claims_judge \
  --jobs judge_inputs/jobs.jsonl \
  --outcomes judge_inputs/outcomes.jsonl \
  --endpoint http://127.0.0.1:8019/v1 \
  --model openai/gpt-oss-120b \
  --tokenizer /path/to/local/gpt-oss-120b-snapshot
```

This command makes inference requests. Preparation and replay are offline.
The server and tokenizer must already exist; the toolkit neither starts a
server nor downloads model assets. The endpoint, served judge model, local
tokenizer path, and tokenizer file hashes
are recorded. Remote endpoints require the environment variable named by
`--api-key-env` (default `OPENAI_API_KEY`); credentials are not saved.

The historical judge was `openai/gpt-oss-120b`, snapshot
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, with a 16,384-token context and low
reasoning effort. Diagnosis requests use temperature 0, JSON-object mode and
budgets 2,048 / 4,096 / 4,096. Prescription requests omit temperature and use
completion-token budgets 4,096 / 8,192 / 8,192. Prompt builders are included
in `vocalcoachbench.claims`; the prescription prompt uses the first 20 expert
issue claims as context.

The optional runner checks actual tokenizer input IDs before requests and
persists a request-start marker before each network call, then preserves every
full-context attempt. An empty diagnosis deterministically
receives `MISS` for every expert claim; an empty prescription list yields no
judgments. Neither case makes a judge call. Unresolved judge responses remain
errors without fallback labels. A resumed invocation only handles jobs never
started; unresolved outcomes or interrupted attempt files require preserving
that run and using a new output path.

Fresh judge results depend on the selected model/runtime. Matching prompts
and settings does not guarantee the historical judge's exact responses.

See [release validation](release_validation.md) for regression coverage and
the optional runner's testing limits.
