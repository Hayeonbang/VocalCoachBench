# Evaluator 0.2.0 validation

The public evaluator was checked against the final **2026-10-26 Open9 + Closed4
leaderboard**, dataset `canonical515_physical_audio_1064refs`, using
`strict_all_attempts` and the stored first responses. No new model or judge
inference was performed.

| Check | Result |
|---|---|
| Raw responses → public parser → structured metrics | 13 models, 51 surfaces, 456 numeric comparisons; zero differences at absolute tolerance 1e-12 |
| Saved Claims samples/jobs/outcomes → public scorer | 13 models, 52/52 ratios exactly equal; 13,390 judge jobs validated |
| Unit and integration regression suite | 52 tests passed |
| Example commands | Passed |
| Frozen original protocol preflight | Passed; original experiment implementations retained |
| Wheel installation in a clean virtual environment | Offline install, CLI and canonical Claims reference loading passed |

Structured coverage is 515 recordings / 1,064 Top-3 references, 262 segments,
and 189 triplets / 567 pairs. Claims retains 6,162 diagnosis references and
2,575 prescription slots per model. Covo-Audio has no published Direct result;
the check does not invent one.

Tests cover recording-balanced macro-F1, partial/failed/missing predictions,
score ties and invalid numbers, duplicate and unknown IDs, presented A/B order,
reference/inventory tampering, frozen prompts, fixed Claims denominators,
judge response and target-link validation, and interrupted inference resumption.
A legacy parser error on a standalone `Audio B` response was corrected. The
stored Direct population (6,804 rows across 12 models) contains zero affected
non-error responses, so the correction changes no final leaderboard result.

[Machine-readable validation summary](release_validation.json) records the
source hashes, reference denominators, leaderboard hash and private replay
receipt hashes. Historical raw prediction/judge archives are not bundled in
this repository; exact historical replay requires those artifacts. The optional
network judge runner was checked using offline mocks and real job preflight,
not by calling a live judge server.
