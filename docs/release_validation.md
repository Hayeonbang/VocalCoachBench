# Evaluator 0.2.0 validation

The evaluator was checked against the **2026-10-26 leaderboard** using dataset
`canonical515_physical_audio_1064refs`, `strict_all_attempts` and stored first
responses. No new model or judge inference was performed.

| Check | Result |
|---|---|
| Raw responses → public parser → structured metrics | 13 models, 51 surfaces, 456 numeric comparisons; zero differences at absolute tolerance 1e-12 |
| Saved Claims samples/jobs/outcomes → public scorer | 13 models, 52/52 ratios exactly equal; 13,390 judge jobs validated |
| Unit and integration suite | 52 tests passed |
| Example commands and clean wheel installation | Passed |

Coverage was 515 recordings / 1,064 Top-3 references, 262 segments, and 189
triplets / 567 pairs. Claims retained 6,162 diagnosis references and 2,575
prescription slots per model. Covo-Audio has no published Direct result.

Historical prediction/judge archives are not bundled; exact replay requires
those artifacts. The optional network judge runner was tested with offline
mocks and real job preflight, without calling a live judge server.
