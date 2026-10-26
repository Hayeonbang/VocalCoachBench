# VocalCoachBench demo page

Static, dependency-free project page: open `index.html` directly, or serve it:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Then open <http://localhost:8000/demo_site/> (serve from the repository root so the
"Browse the full dataset" link to `hf_release/VocalCoachBench_annotations/demo/index.html` resolves).
Keep this preview bound to localhost. Do not expose the repository root publicly;
it also contains private experiment and credential files.

Files

- `index.html`, `style.css`, `app.js`: the page (hero, 01 Expert annotations, 02 Model feedback vs. expert claims, 03 Leaderboard, 04 Source datasets). Fonts (Space Grotesk, Inter, JetBrains Mono) load from Google Fonts and fall back to system fonts offline.
- `data.js`: showcase annotations, model outputs, judge labels, waveform peaks, leaderboard numbers, source-dataset table.
- `audio/`: four Dataset-B recordings copied unchanged from `hf_release/VocalCoachBench_annotations/audio/diverse_song/`.
- `assets/vcb_overview.png`: rendered from `neurips2026/figs/VCB_final.pdf`.
- `scripts/build_data.py`: regenerates `data.js` and `../project_page/leaderboard.js` from the public release, historical v5 showcase artifacts, and `rebuttal_experiments/20261026_final_leaderboard/leaderboard.json`. The manuscript is not a numeric source.
- `scripts/verify_leaderboard.py`: independently checks both public snapshots against structured result files, judge counts and fixed denominators, verifies linked receipt hashes, and recomputes the majority Top-3 macro-F1 from canonical annotations.

Refresh the canonical export first, then run from the repository root:

```bash
python rebuttal_experiments/20261026_final_leaderboard/build_leaderboard.py
python demo_site/scripts/build_data.py
python demo_site/scripts/verify_leaderboard.py
```

The leaderboard contains 13 models (9 open-weight, 4 closed API) and one
no-audio reference baseline. All metrics are stored as unrounded ratios and
displayed as percentages. Top-3 uses recording-balanced seven-class macro-F1
over 515 recordings / 1,064 references, with `strict_all_attempts`. Prescription
validity and diagnosis-to-prescription E2E use 2,575 fixed slots, not only the
emitted prescriptions. Main/Simple protocols, access type, concatenated-audio
Direct variants and unavailable values are explicit.

The sample feedback comparison is intentionally marked **historical**: its
outputs and judge labels are illustrative v5 artifacts, not the current aligned
leaderboard cohort. Their source paths remain in `showcase_provenance`.

`project_page/` is independently static: publish its `index.html`, `results.js`,
generated `leaderboard.js`, and existing `img/` and `audio/` assets together.
For public hosting, stage only the intended static assets and the referenced
`rebuttal_experiments/20261026_final_leaderboard/leaderboard.json` at the matching
relative path; do not publish the repository root. No remote hosting or
deployment is performed by these build scripts.

Live site: https://hayeonbang.github.io/VocalCoachBench/ (branch `gh-pages` of github.com/Hayeonbang/VocalCoachBench; to update, copy this folder minus `scripts/` to that branch and swap the dataset-browser link to the Hugging Face URL).

Links: paper arXiv:2609.04241, dataset https://huggingface.co/datasets/Hayeonbang/VocalCoachBench, code https://github.com/Hayeonbang/VocalCoachBench.
