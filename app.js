/* VocalCoachBench demo page. Plain script, no build step; data comes from data.js. */
(function () {
  "use strict";
  const D = window.VCB_DATA;

  const ISSUE = {
    PITCH: { c: "var(--pitch)", ko: "음정" },
    RHYTHM: { c: "var(--rhythm)", ko: "리듬" },
    DICTION: { c: "var(--diction)", ko: "발음" },
    BREATH: { c: "var(--breath)", ko: "호흡" },
    VOCALIZATION: { c: "var(--vocalization)", ko: "발성" },
    PHONATION: { c: "var(--vocalization)", ko: "발성" },
    TECHNIQUE: { c: "var(--technique)", ko: "테크닉" },
    EXPRESSION: { c: "var(--expression)", ko: "표현" },
  };
  const PAIR = ["var(--pair-1)", "var(--pair-2)", "var(--pair-3)", "var(--pair-4)", "var(--pair-5)", "var(--pair-6)"];
  const JUDGE = { STRICT_HIT: ["strict", "strict hit"], COARSE_HIT: ["coarse", "coarse hit"], MISS: ["miss", "miss"], CONTRADICT: ["contradict", "contradict"] };
  const FIX = { VALID: ["valid", "valid"], WEAK: ["weak", "weak"], INVALID: ["invalid", "invalid"], VOCALLY_INAPPROPRIATE: ["invalid", "unsafe"] };
  const PROTO = { Ref: "var(--miss)", Main: "var(--accent)", Closed: "var(--rhythm)", Simple: "var(--vocalization)" };
  const PROTO_TITLE = {
    Ref: "Reference baseline: never sees audio",
    Main: "Main protocol: structured JSON prompts",
    Closed: "Closed-source model, main protocol via public API",
    Simple: "Simple protocol: short natural-language prompts; not directly comparable",
  };

  const state = { lang: "en", expanded: new Set(), claimsOpen: new Set(), compareIdx: 0, allCols: false, sort: null };

  // ---- tiny DOM helper -------------------------------------------------
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "onclick") el.addEventListener("click", v);
        else if (k === "checked") el.checked = !!v;
        else el.setAttribute(k, v === true ? "" : v);
      }
    }
    const add = (c) => {
      if (c === null || c === undefined || c === false) return;
      if (Array.isArray(c)) c.forEach(add);
      else el.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    };
    children.forEach(add);
    return el;
  }
  const txt = (o) => (state.lang === "ko" && o.text_ko ? o.text_ko : o.text);
  const title = (s) => s.charAt(0) + s.slice(1).toLowerCase();
  const cats = (x) => [].concat(x || []);
  function fmtDur(s) {
    if (s === null || s === undefined || isNaN(s)) return "–:––";
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return m + ":" + String(r).padStart(2, "0");
  }
  function issueChip(label) {
    const i = ISSUE[label] || { c: "var(--muted)" };
    return h("span", { class: "chip", style: "--c:" + i.c, title: label }, state.lang === "ko" && i.ko ? i.ko : label);
  }
  function rankBadge(n, label) {
    const i = ISSUE[label] || { c: "var(--muted)" };
    return h("span", { class: "rank", style: "--c:" + i.c }, h("i", null, n), state.lang === "ko" && i.ko ? i.ko : label);
  }
  function badge(map, label, reason) {
    const m = map[label];
    if (!m) return h("span", { class: "badge none", title: reason || "not judged" }, h("i", { class: "g" }), h("b", null, "–"));
    return h("span", { class: "badge " + m[0], title: reason || "" }, h("i", { class: "g" }), h("b", null, m[1]));
  }

  // ---- audio player with a precomputed waveform (no fetch, works from file://) ----
  const players = {};
  function player(e) {
    if (players[e.id]) return players[e.id];
    const peaks = e.peaks || [], N = peaks.length || 1;
    const svg = () => '<svg viewBox="0 0 ' + N + ' 100" preserveAspectRatio="none" aria-hidden="true">' +
      peaks.map((v, i) => { const hh = Math.max(3, v * 96); return '<rect x="' + (i + 0.2) + '" width="0.6" y="' + (50 - hh / 2) + '" height="' + hh + '"/>'; }).join("") + "</svg>";
    const audio = h("audio", { preload: "metadata", src: e.audio });
    const wave = h("div", { class: "wave", role: "slider", "aria-label": "Seek" });
    wave.innerHTML = svg() + '<div class="fill">' + svg() + "</div>";
    const fill = wave.querySelector(".fill");
    const time = h("span", { class: "time" }, "0:00 / " + fmtDur(e.duration));
    const btn = h("button", { class: "play", "aria-label": "Play" });
    const root = h("div", { class: "player" }, btn, wave, time, audio);
    const setPct = (pct) => { fill.style.clipPath = "inset(0 " + (100 - Math.max(0, Math.min(1, pct)) * 100) + "% 0 0)"; };
    setPct(0);
    btn.addEventListener("click", () => {
      if (audio.paused) { document.querySelectorAll("audio").forEach((a) => { if (a !== audio) a.pause(); }); audio.play(); }
      else audio.pause();
    });
    audio.addEventListener("play", () => root.classList.add("playing"));
    audio.addEventListener("pause", () => root.classList.remove("playing"));
    audio.addEventListener("ended", () => { root.classList.remove("playing"); setPct(0); });
    audio.addEventListener("timeupdate", () => {
      const d = audio.duration || e.duration || 1;
      setPct(audio.currentTime / d);
      time.textContent = fmtDur(audio.currentTime) + " / " + fmtDur(d);
    });
    wave.addEventListener("click", (ev) => {
      const r = wave.getBoundingClientRect(), pct = (ev.clientX - r.left) / r.width, d = audio.duration || e.duration;
      if (d) { audio.currentTime = pct * d; setPct(pct); }
    });
    players[e.id] = root;
    return root;
  }

  // ---- section 2: listen & read ---------------------------------------
  function noteLine(e) {
    const first = e.top3[e.annotators[0]][0];
    return e.top1_differs
      ? h("span", { class: "note-line" }, h("i", { class: "g" }), "Top-1 issue differs between annotators")
      : h("span", { class: "note-line agree" }, h("i", { class: "g" }), "Both annotators rank " + first + " first");
  }
  const CLAIM_CUTOFF = 5, CLAIM_SHOWN = 4;
  function renderClaims(claims, blockUid) {
    const colorOf = {};
    claims.filter((c) => c.claim_type === "issue").forEach((c, i) => { colorOf[c.claim_uid] = PAIR[i % PAIR.length]; });
    const open = state.claimsOpen.has(blockUid);
    const shown = claims.length > CLAIM_CUTOFF && !open ? claims.slice(0, CLAIM_SHOWN) : claims;
    const hidden = claims.length - shown.length;
    return h("div", { class: "claims" }, shown.map((c) => {
      let color = "var(--muted)", tag = c.claim_type, dots = null;
      if (c.claim_type === "issue") color = colorOf[c.claim_uid];
      else if (c.claim_type === "correction") {
        const t = (c.target_claim_uids || []).filter((u) => colorOf[u]);
        if (t.length) color = colorOf[t[0]];
        if (t.length > 1) dots = h("span", { class: "dots", title: "also addresses " + (t.length - 1) + " more issue(s)" },
          t.slice(1).map((u) => h("i", { style: "background:" + colorOf[u] })));
      } else color = "var(--strict)";
      return h("div", { class: "claim " + c.claim_type, style: "--c:" + color }, h("span", { class: "tag" }, tag), h("span", null, txt(c), dots));
    }), hidden > 0 || (open && claims.length > CLAIM_CUTOFF)
      ? h("button", { class: "more small", onclick: () => { open ? state.claimsOpen.delete(blockUid) : state.claimsOpen.add(blockUid); renderShowcase(); } },
        open ? "Fewer" : "+" + hidden + " more")
      : null);
  }
  function note(e, b) {
    const claims = e.claims.filter((c) => c.block_uid === b.block_uid);
    return h("div", { class: "note" },
      h("div", { class: "cats" }, b.categories.map(issueChip)),
      renderClaims(claims, b.block_uid),
      h("details", { class: "orig" }, h("summary", null, "Original note"), h("p", null, txt(b))));
  }
  function coachColumn(e, a) {
    const blocks = e.blocks.filter((b) => b.annotator_id === a);
    const top1 = e.top3[a][0];
    const key = e.id + "|" + a;
    const open = state.expanded.has(key);
    const preferred = blocks.find((b) => b.categories.includes(top1)) || blocks[0];
    const shown = open ? blocks : blocks.filter((b) => b === preferred);
    return h("div", { class: "coach" },
      h("div", { class: "who" }, h("span", { class: "name" }, a), h("span", { class: "hint" }, "Top-3")),
      h("div", { class: "ranks" }, e.top3[a].map((l, i) => rankBadge(i + 1, l))),
      shown.map((b) => note(e, b)),
      blocks.length > 1 ? h("button", { class: "more", onclick: () => { open ? state.expanded.delete(key) : state.expanded.add(key); renderShowcase(); } },
        open ? "Fewer notes" : "All " + blocks.length + " notes") : null);
  }
  function renderShowcase() {
    const root = document.getElementById("showcase");
    root.replaceChildren(...D.showcase.map((e) => h("article", { class: "card", id: "rec-" + e.id },
      h("p", { class: "headline" }, e.title),
      h("div", { class: "meta" },
        h("span", null, "Diverse-song subset"),
        h("span", null, e.source_dataset),
        h("span", null, fmtDur(e.duration)),
        h("span", { class: "rid" }, e.id)),
      noteLine(e),
      player(e),
      h("div", { class: "coaches" }, e.annotators.map((a) => coachColumn(e, a))))));
  }

  // ---- section 3: models vs experts -----------------------------------
  function renderTabs() {
    const tabs = document.getElementById("compare-tabs");
    tabs.replaceChildren(...D.showcase.map((e, i) => h("button", { class: i === state.compareIdx ? "on" : "", role: "tab", onclick: () => { state.compareIdx = i; renderTabs(); renderCompare(); } }, e.title)));
  }
  function renderCompare() {
    const e = D.showcase[state.compareIdx];
    const tags = Object.keys(e.models);
    const J = {};
    tags.forEach((t) => { J[t] = {}; e.models[t].diag_judge.forEach((j) => { if (j.gt_uid) J[t][j.gt_uid] = j; }); });
    const blockCats = {};
    e.blocks.forEach((b) => { blockCats[b.block_uid] = b.categories; });
    const rows = [];
    const tally = {};
    tags.forEach((t) => { tally[t] = { STRICT_HIT: 0, COARSE_HIT: 0, MISS: 0, CONTRADICT: 0 }; });
    e.annotators.forEach((a) => {
      const top1 = e.top3[a][0];
      rows.push(h("tr", { class: "group" }, h("td", { colspan: 1 + tags.length }, a, " ", h("span", { class: "first" }, "· Top-1 "), rankBadge(1, top1))));
      e.claims.filter((c) => c.claim_type === "issue" && c.annotator_id === a).forEach((c) => {
        const isTop1 = (blockCats[c.block_uid] || []).includes(top1);
        rows.push(h("tr", { class: isTop1 ? "top1" : "", style: isTop1 ? "--c:" + ISSUE[top1].c : null, title: isTop1 ? "From a note about the coach's #1 issue" : null },
          h("td", { class: "ctext" }, txt(c)),
          tags.map((t) => { const j = J[t][c.claim_uid]; if (j) tally[t][j.label] = (tally[t][j.label] || 0) + 1;
            const refs = j ? (j.matched || []).map((id) => String(id).replace(/\D/g, "")).filter(Boolean) : [];
            return h("td", { class: "mcell" }, badge(JUDGE, j && j.label, j && j.reason), refs.length ? h("span", { class: "mref", title: "matched model diagnosis " + refs.join(", ") }, refs.join(",")) : null); })));
      });
    });
    const matrix = h("div", { class: "matrix-card" },
      h("h3", null, "Did the model catch each expert diagnosis?"),
      h("p", { class: "sub" }, "Number = the model diagnosis it matched (numbered in the cards below). Coloured edge = claim from a block tagged with the annotator's Top-1 issue."),
      h("div", { class: "table-wrap" }, h("table", { class: "matrix" },
        h("thead", null, h("tr", null, h("th", null, "Expert diagnosis"), tags.map((t) => h("th", null, e.models[t].label)))),
        h("tbody", null, rows))));

    const cards = h("div", { class: "models" }, tags.map((t) => modelCard(e, t, tally[t])));
    const cardsHead = h("div", { class: "models-head" }, h("h3", null, "What each model said"), h("p", { class: "sub" }, "Numbers match the matrix above. Corrections point at the diagnosis they address."));
    document.getElementById("compare-body").replaceChildren(matrix, cardsHead, cards);
  }
  function modelCard(e, t, k) {
    const m = e.models[t], p = m.parsed || {};
    const diags = p.diagnosis_claims || [], fixes = p.prescription_claims || [], strengths = p.strength_claims || [];
    const best = {};
    m.diag_judge.forEach((j) => (j.matched || []).forEach((id) => {
      const rank = { STRICT_HIT: 3, COARSE_HIT: 2, CONTRADICT: 1 }[j.label] || 0;
      if (rank > (best[id] ? best[id].rank : 0)) best[id] = { rank, label: j.label, reason: j.reason };
    }));
    const fixJ = {};
    m.presc_judge.forEach((j) => { fixJ[j.pid] = j; });
    const empty = !diags.length && !fixes.length;
    return h("div", { class: "model-card" },
      h("h3", null, m.label, h("span", { class: "chip neutral proto", title: PROTO_TITLE[m.protocol === "simple" ? "Simple" : "Main"] }, m.protocol === "simple" ? "simple protocol" : "main protocol")),
      k ? h("p", { class: "tally-line" }, k.STRICT_HIT + " strict · " + k.COARSE_HIT + " coarse · " + k.MISS + " miss" + (k.CONTRADICT ? " · " + k.CONTRADICT + " contradict" : "") + " of " + (k.STRICT_HIT + k.COARSE_HIT + k.MISS + k.CONTRADICT) + " expert diagnoses") : null,
      empty ? h("p", { class: "sub" }, "No claims could be extracted from this output.") : null,
      diags.length ? h("p", { class: "h" }, "Diagnoses") : null,
      h("ul", { class: "list" }, diags.map((d) => { const b = best[d.id]; return h("li", null, h("span", { class: "num" }, String(d.id).replace(/\D/g, "")),
        h("span", null, cats(d.category).map(issueChip), " ", d.text, h("span", { class: "li-badges" }, b ? badge(JUDGE, b.label, b.reason) : h("span", { class: "badge none", title: "No expert claim matched this diagnosis" }, h("i", { class: "g" }), h("b", null, "no match"))))); })),
      fixes.length ? h("p", { class: "h" }, "Corrections") : null,
      h("ul", { class: "list" }, fixes.map((f) => { const j = fixJ[f.id]; const tgt = f.target_diagnosis_id ? String(f.target_diagnosis_id).replace(/\D/g, "") : null;
        return h("li", { class: "fix" }, h("span", { class: "num", title: tgt ? "addresses diagnosis " + tgt : "no target diagnosis" }, tgt ? "→" + tgt : "→?"),
          h("span", null, f.text, h("span", { class: "li-badges" }, badge(FIX, j && j.label, j && j.reason)))); })),
      strengths.length ? h("p", { class: "h" }, "Strengths") : null,
      strengths.length ? h("ul", { class: "list" }, strengths.map((s) => h("li", null, h("span", { class: "num" }, "+"), h("span", null, s.text)))) : null,
      h("details", null, h("summary", null, "Raw model output"), h("pre", null, m.raw || "(empty)")));
  }

  // ---- section 4: leaderboard -----------------------------------------
  const LB = D.leaderboard;
  const COLS = [
    ["top3_macro_f1", "Top-3 macro-F1", "Top-3"],
    ["segment_macro_f1", "Segment macro-F1", "Segment"],
    ["segment_match", "Segment Match", "Segment"],
    ["score_rank", "Score pair acc.", "Ranking"],
    ["direct", "Direct pair acc.", "Ranking"],
    ["dx_strict", "Strict hit", "Diagnosis"],
    ["dx_hit", "Hit", "Diagnosis"],
    ["rx_valid", "Valid / fixed slots", "Correction"],
    ["e2e", "E2E", "Correction"],
  ].map(([key, label, group]) => ({ key, label, group, get: (r) => r.metrics[key] }));
  const isReference = (r) => LB.reference_rows.some((ref) => ref.model_id === r.model_id);
  const protocolLabel = (r) => isReference(r) ? "Ref" : r.protocol === "simple" ? "Simple" : "Main";
  const formatMetric = (v) => v === null || v === undefined ? "N/A" : (v * 100).toFixed(2);
  const CORE_ORDER = ["top3_macro_f1", "direct", "dx_strict", "dx_hit", "rx_valid", "e2e"];

  function visibleCols() {
    return state.allCols ? COLS : CORE_ORDER.map((key) => COLS.find((c) => c.key === key));
  }
  function sortedRows() {
    const rows = [...LB.rows, ...LB.reference_rows].map((r, i) => ({ r, i }));
    if (!state.sort) return rows.map((x) => x.r);
    const { key, dir } = state.sort;
    const col = COLS.find((c) => c.key === key);
    rows.sort((a, b) => {
      if (["model", "protocol", "access"].includes(key)) {
        const va = a.r[key], vb = b.r[key];
        return (va < vb ? -1 : va > vb ? 1 : a.i - b.i) * dir;
      }
      const va = parseFloat(col.get(a.r)), vb = parseFloat(col.get(b.r));
      if (isNaN(va) && isNaN(vb)) return a.i - b.i;
      if (isNaN(va)) return 1;
      if (isNaN(vb)) return -1;
      return (vb - va) * dir || a.i - b.i;
    });
    return rows.map((x) => x.r);
  }
  function renderLeaderboard() {
    const cols = visibleCols();
    const th = (key, label, extra) => {
      const on = state.sort && state.sort.key === key;
      return h("th", Object.assign({ class: on ? "sorted" : "", onclick: () => {
        state.sort = on ? (state.sort.dir === 1 ? { key, dir: -1 } : null) : { key, dir: 1 };
        renderLeaderboard();
      } }, extra || {}), label, on ? h("span", { class: "arrow" }, state.sort.dir === 1 ? " \u25BE" : " \u25B4") : null);
    };
    const head = [];
    if (state.allCols) {
      const groups = [];
      cols.forEach((c) => { const g = groups[groups.length - 1]; if (g && g.name === c.group) g.n++; else groups.push({ name: c.group, n: 1 }); });
      head.push(h("tr", null, h("th", { colspan: 3 }), groups.map((g) => h("th", { colspan: g.n, style: "text-align:center" }, g.name))));
    }
    head.push(h("tr", null, th("protocol", "Protocol"), th("access", "Access"), th("model", "Model"),
      cols.map((c) => th(c.key, c.label, { title: LB.metric_definitions[c.key].definition }))));
    const body = sortedRows().map((r) => {
      const proto = protocolLabel(r);
      return h("tr", { class: isReference(r) ? "ref" : "", "data-model-id": r.model_id },
        h("td", null, h("span", { class: "chip", style: "--c:" + PROTO[proto], title: PROTO_TITLE[proto] }, proto)),
        h("td", null, isReference(r) ? "No audio" : r.access === "closed" ? "Closed" : "Open"),
        h("td", { class: "model", title: (r.notes || []).join("\n") }, r.model, r.notes && r.notes.length ? " *" : ""),
        cols.map((c) => { const v = c.get(r);
          return h("td", { class: v === null || v === undefined ? "empty" : "", "data-metric": c.key }, formatMetric(v)); }));
    });
    document.getElementById("lb").replaceChildren(h("thead", null, head), h("tbody", null, body));
  }

  function renderLeaderboardProvenance() {
    document.getElementById("model-count").textContent = LB.rows.length;
    document.getElementById("leaderboard-provenance").replaceChildren(
      h("p", null, "Dataset version: ", h("code", null, LB.dataset_version)),
      h("p", null, "Source artifact: ", h("a", { href: "../" + LB.publication_source.path }, LB.publication_source.path)),
      h("p", null, "Source SHA-256: ", h("code", null, LB.publication_source.sha256)));
    document.getElementById("leaderboard-notes").replaceChildren(
      h("h3", null, "Per-model provenance and caveats"),
      h("ul", null, LB.rows.filter((r) => r.notes && r.notes.length).map((r) => h("li", null, h("b", null, r.model + ": "), r.notes.join(" ")))),
      h("h3", null, "Metric definitions"),
      h("ul", null, COLS.map((c) => h("li", null, h("b", null, c.label + ": "), LB.metric_definitions[c.key].definition))));
  }

  // ---- footer: sources ------------------------------------------------
  function renderSources() {
    const rows = D.sources.map((s) => h("tr", null,
      h("td", null, h("a", { href: s.source_url, target: "_blank", rel: "noopener" }, s.source_dataset)),
      h("td", { class: "n" }, s.recordings_in_vocalcoachbench),
      h("td", null, s.audio_redistributed ? h("span", { class: "chip", style: "--c:var(--strict)" }, "Included")
        : h("span", { class: "chip neutral", title: s.audio_redistribution_status }, s.audio_redistribution_status === "excluded" ? "Not included" : "Not included · " + s.audio_redistribution_status.replace(/_/g, " "))),
      h("td", null, s.license)));
    document.getElementById("sources").replaceChildren(
      h("thead", null, h("tr", null, h("th", null, "Source dataset"), h("th", { style: "text-align:right" }, "Recordings"), h("th", null, "Audio in release"), h("th", null, "Licence / terms"))),
      h("tbody", null, rows));
  }

  // ---- wiring ---------------------------------------------------------
  document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => {
    state.lang = b.dataset.lang;
    document.querySelectorAll("[data-lang]").forEach((x) => x.classList.toggle("on", x === b));
    renderShowcase(); renderCompare();
  }));
  document.getElementById("allcols").addEventListener("change", (ev) => { state.allCols = ev.target.checked; renderLeaderboard(); });
  document.querySelectorAll("[data-todo]").forEach((a) => a.addEventListener("click", (ev) => ev.preventDefault()));

  renderShowcase();
  renderTabs();
  renderCompare();
  renderLeaderboard();
  renderLeaderboardProvenance();
  renderSources();
})();
