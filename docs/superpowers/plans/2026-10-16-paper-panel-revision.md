# Paper Panel-Review Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise `paper/main.tex` so every Required item (R1–R21) and every should-fix item (S1–S28, S30–S38) of the 2026-10-16 panel roadmap is addressed, with new numbers only from cached results.

**Architecture:** Three waves. Wave 1 (parallel, main tree, distinct files): a tested analysis module that writes one new results JSON, verified BibTeX additions, and a facts file. Wave 2 (parallel, one git worktree per writer, disjoint section ownership): three writers edit `main.tex`. Wave 3 (main tree): front-matter rewrite after merge, build, three read-only verifiers, fixes, commit.

**Tech Stack:** LaTeX built with `tectonic`; Python 3.12 project venv (`venv/bin/python`), numpy, pytest; git worktrees.

**Spec:** `paper/analysis/panel-review-2026-10-16/synthesis.md` (Part 2 roadmap: item text, Requirement, Acceptance criteria). Reviewer cards in the same folder.

## User decisions (2026-10-16, binding)

- R1: **two research questions, sweeps stay in the body** — sweep methods move out of Results into Methods/Protocol; sweep results get their own Results section; a sweep Discussion subsection is added; title adjusted.
- Analyses: **cheap re-analyses from cached predictions/result files only**. No new inference. R4 takes route (b): round five restated at artifact level. R20 takes the scoping route (no new range-aware test).
- S11: paper states the bundled weights derive from CC BY-NC 4.0 data and are released **for non-commercial research use only**. No LICENSE file is added.
- Git: branch `docs/paper-panel-revision`, local commits, **no push**. Consider items S29 and S39 are optional.

## Global Constraints

- Research prototype, not a medical device: no wording may imply clinical validity or treatment guidance (`CLAUDE.md`).
- Never modify `GlucoBench/`, frozen protocol files, or any existing file in `paper/results/`; new results are new files only. No CGM readings or participant IDs in committed files (anonymize as P1…Pn).
- Every new number in the paper traces to a committed JSON in `paper/results/` (existing files or `paper/results/calibration-coverage/panel_revision_r1.json`).
- Every new citation is verified by DOI or web search before entering `paper/references.bib`; no unverified entry is cited.
- Abstract stays ≤ 1,920 plain-text characters (arXiv metadata limit).
- Build: `cd paper && tectonic main.tex` must succeed with no undefined references or citations. No new LaTeX packages.
- Pinned replacements: "toolchain-default" → "contiguous first-$N$ (dataset-order) selection" at first use, "contiguous first-$N$ selection" after (the arm label "sequential" stays); "clinical thresholds" → "trend-arrow rate bands ($\pm1$ and $\pm2$~\mgdl/min)"; "clinical-shape classifier" → "directional classifier"; "label-noise ceiling"/"headroom" → "label-definition stability check"; "never-read" → "never-evaluated"; abstract/conclusion "ships" → "is bundled in the research applications".
- "calibration" means quantization calibration; first use says it is neither CGM sensor calibration nor probability calibration.
- Round five is stated at artifact level: "the seed-4 flatten-head artifact (cosine schedule, validation-chosen epoch)", with the 1.1-point five-seed spread stated beside every citation of +0.20.
- Title: `GlucoEdge: Quantization-Calibration Coverage and Staged Model Selection in On-Device Glucose Trend Classification`.
- New labels (exact): `app:history`, `app:cohorts`, `sec:sweepprotocol`, `sec:sweepresults`, `sec:sweepdiscussion`, `tab:protocolmap`, `tab:cohorts`, `tab:variants`, `tab:readledger`, `tab:shippederrors`. Keep every existing label.
- Agents in the main tree never run `git add/commit`; the orchestrator commits. Writers commit only on their own worktree branch.
- Commit messages end with the session attribution lines.
- S31 applies to every writer: expand each acronym at its first use inside the writer's own region (INT8, PTQ, LiteRT, TCN, CORN, QDQ, T1D, …) and map recall to sensitivity / precision to positive predictive value where clinical readers meet them.

## Review Focus

1. Merge of three writer branches drops or duplicates an edit → each writer's acceptance items are re-verified after merge (Task 8 verifiers V1/V2).
2. A number edited in one place but not its twins (abstract, intro, tables, conclusion) → V3 numeric-consistency audit over the whole file.
3. LaTeX breaks after merge (labels, table widths, appendix after bibliography) → Task 7 build gate, zero undefined refs/cites.
4. Abstract grows past 1,920 characters with the two-RQ framing → Task 7 character count check.
5. A reviewer request is factually wrong (e.g. S10: `recency_collision.json` *is* versioned under `paper/analysis/accuracy-review-2026-10-13/`) → writers verify before editing and record REVIEWER_DISAGREE in the response table instead of complying.

---

### Task 0: Branch and review package (done)

- [x] `git checkout -b docs/paper-panel-revision`; commit `paper/analysis/panel-review-2026-10-16/` (f746bf4).

### Task 1: Analysis module and results JSON (Wave 1, agent "analysis")

**Files:**
- Create: `experiments/panel_revision.py`, `tests/test_panel_revision.py`
- Create (outputs): `paper/results/calibration-coverage/panel_revision_r1.json`, `paper/results/calibration-coverage/artifact_hashes_r1.json`

**Interfaces (produces, pure functions tested on synthetic arrays):**
- `participant_contributions(y_true, participant_ids, pred_a, pred_b, classes=MINORITY) -> dict[str, float]` — additive per-participant contributions (recall points ×100) to the pooled minority-macro-recall difference A−B, using pooled class supports; values sum to the pooled difference.
- `mean_contributions(y_true, participant_ids, pred_pairs) -> dict[str, float]` — average of the above over (seed × draw) pairs.
- `sign_flip_test(contribs: list[float]) -> dict` — exact two-sided test over all 2^G sign patterns (G ≤ 20): keys `observed`, `p_two_sided`, `n_patterns`.
- `cluster_bootstrap_stat(y_true, participant_ids, stat_fn, n_boot=2000, seed=ANALYSIS_SEED) -> dict` — same draw/percentile/undefined-fraction semantics as `analyze_calibration_coverage.paired_cluster_bootstrap`, for arbitrary statistics (interaction contrasts).
- `censor_shares(windows, labels, endpoint=None, codes=(39.0, 401.0)) -> dict` — window fractions with any input (and, if given, endpoint) at a code, overall and by true class.
- `confusion_summary(y_true, y_pred) -> dict` — 5×5 matrix, per-class recall (sensitivity), precision (PPV), adjacent-class error share, direction-reversal shares (true falling/falling_fast predicted rising/rising_fast and mirror).
- `level_stratified_recall(y_true, y_pred, anchor_mgdl, edges=(70, 180, 250)) -> dict` — directional recall and support per band `<70, 70–180, 180–250, >250`.
- Reuse (consume) `confusion_counts`, `recalls`, `minority_macro_recall`, `macro_f1`, `resample_rows`, `ANALYSIS_SEED` from `experiments/analyze_calibration_coverage.py`.

- [ ] **Step 1: Write failing tests** in `tests/test_panel_revision.py`: `test_contributions_sum_to_pooled_difference` (random 3-participant data; `sum(contribs) == 100*(mmr(a)-mmr(b))` to 1e-9); `test_sign_flip_all_positive_six` (`[1,2,3,4,5,6]` → `p_two_sided == 2/64`, `n_patterns == 64`); `test_sign_flip_symmetric` (`[1,-1]` → `p_two_sided == 1.0`); `test_cluster_bootstrap_reproducible` (same seed → identical ci95; ci95 contains the point); `test_censor_shares_counts` (2 of 4 windows contain 401 → 0.5); `test_confusion_summary_reversal` (one true falling_fast predicted rising_fast → reversal share 1/support); `test_level_bands` (anchors 60/100/200/300 land in the four bands).
- [ ] **Step 2:** `venv/bin/python -m pytest tests/test_panel_revision.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement the functions; then a `main()` runner (not unit-tested; needs local `results/`) that writes `panel_revision_r1.json` with these top-level keys, each entry carrying its source file paths:
  - `weinstock.seed_draw_variation` — per seed: float directional recall; uniform200−sequential200 effect; within-seed SD over the three uniform draws (held_out, temporal).
  - `weinstock.interaction` — (a) `uniform_vs_sequential`: (U1000−S1000)−(U200−S200); (b) `randomized_pairs`: budget-effect differences for range−balanced, range−uniform, balanced−uniform. Each with point, ci95, undefined_fraction from `cluster_bootstrap_stat` (n_boot 2000, ANALYSIS_SEED). Note that `protocol.json` lists the interaction without a formula.
  - `weinstock.sequential_1000_vs_200` — held_out, temporal, same bootstrap.
  - `weinstock.censoring` — value counts at 38–41 and 399–402 mg/dL in eligible inputs (evidence for 39/401 being LOW/HIGH codes); `censor_shares` for held_out and temporal (inputs; endpoints if they can be rebuilt exactly with the study loader, else `null` with a reason); uniform200−sequential200 held-out contrast re-estimated excluding windows with a coded anchor (and endpoint when available).
  - `weinstock.facts` — max |histogram−minmax| observer difference (points), randomized-200 participant range across all draws, widest single-tier interval per component from `sensitivity_tiers_r1.json`.
  - `confirmation.{dubosson,hall}.{temporal,held_out}` — anonymized per-participant mean contributions (P1…Pn), `sign_flip_test`, per-class support, undefined_fraction (copied from `confirmation_*.json`), and clip window fractions (overall and falling_fast) for sequential-200 and uniform-200 from the cached `*_meta.json` (mean over seeds; note if identical). Include Weinstock's held-out clip values from `r1` meta for comparison.
  - `shipped_model.{held_out,rtcgm}` — `confusion_summary` of the round-five winner INT8 (seed 4) from `results/calibration-coverage/sweep-r5/eval/*/winner_int8_preds.npy`; `level_stratified_recall` on held_out (anchor rebuilt from the study's held-out windows, aligned by key); RT-CGM level stratification only if anchors can be rebuilt in under ~15 minutes, else `null` with a reason.
  - `round_one_f1.{held_out,temporal}` — per-class F1 and precision, winner INT8 vs champion INT8, from `results/calibration-coverage/sweep-r1/eval/`.
  - `pedap_per_class` — per-class recall, round-five winner INT8 vs champion INT8.
  - `artifacts` — INT8 artifact counts (study 110, of which 8 device-timed; dubosson, hall, delta), byte sizes and operator inventory of the confirmation and delta INT8 artifacts, and the tensor dtype map (which ops/tensors are float vs int8) of one study INT8 artifact.
  - `sweep_settings` — training batch size per sweep round (smoke records), INT8 calibration pool per round.
  - Separately write `artifact_hashes_r1.json` (artifact name → SHA-256) from `results/calibration-coverage/r1/artifacts_manifest.json`.
- [ ] **Step 4:** tests PASS; run `venv/bin/python -m experiments.panel_revision` → JSON written; `venv/bin/python -m pytest -q` (whole suite) PASS.
- [ ] **Step 5:** report back; orchestrator commits `feat(analysis): panel-revision re-analyses from cached results`.

### Task 2: Verified citations (Wave 1, agent "citations")

**Files:** Modify `paper/references.bib` (append only). Create `paper/analysis/panel-review-2026-10-16/citation_verification.md` (key, DOI/URL, how verified, one-line content check).

- [ ] Verify and append BibTeX (keys `firstauthorYEARword`, DOI where one exists) for: Williams & Aletras ACL 2024; Yu, Yang & Shi CVPRW 2021; Cai et al. ZeroQ CVPR 2020; Krishnamoorthi 2018 (arXiv:1806.08342); Nagel et al. 2021 (arXiv:2106.08295); Wu et al. 2020 integer-quantization evaluation (arXiv); Zhu et al. IEEE IoT J (IoMT real-time glucose prediction with edge computing); Marling & Bunescu OhioT1DM 2020; Kovatchev et al. Diabetes Care 2004 (CG-EGA); Zanderigo et al. JDST 2007; Facchinetti et al. IEEE TBME 2014 (sensor error model); Lin et al. ICCV 2017 (focal loss); Bai, Kolter & Koltun 2018 (TCN); Loshchilov & Hutter ICLR 2017 (SGDR); Shi, Cao & Raschka (CORN); Wortsman et al. ICML 2022 (model soups); Frankle et al. ICML 2020 (linear mode connectivity); Yang et al. ICML 2021 (deep imbalanced regression); Cameron, Gelbach & Miller REStat 2008; Mitchell et al. FAT* 2019 (model cards); Battelino et al. Diabetes Care 2019 (time-in-range consensus); cohort sources: JDRF CGM Study Group NEJM 2008, Wadwa et al. NEJM 2023 (PEDAP), Brown et al. NEJM 2019 (iDCL/DCLP3), Aleppo et al. Diabetes Care 2017 (REPLACE-BG), Laffel et al. JAMA 2020 (CITY), Pratley et al. JAMA 2020 (WISDM), the SENCE trial primary paper, Dubosson et al. 2018 (D1NAMO), Hall et al. PLoS Biol 2018 (glucotypes) — skip any key already present in `references.bib`.
- [ ] Any work that cannot be verified is listed as UNVERIFIED in the log and NOT added.
- [ ] Orchestrator commits `docs(paper): add verified references for panel revision`.

### Task 3: Facts file (Wave 1, agent "facts", read-only except its output)

**Files:** Create `/private/tmp/claude-501/-Users-mohamednabil-Documents-vibe-GlucoEdge/65b3560f-ad33-4d3f-8d3d-c28b2c7242d3/scratchpad/revision/facts.md`.

- [ ] Record, each with source and exact quote/value: TF `RepresentativeDataset` and LiteRT quantization guidance wording (cited pages); torchao/LiteRT Torch fixed-qparams API (`FixedQParamsQuantizationSpec`) availability; Dexcom G4-era reportable range and LOW/HIGH display; Weinstock et al. 2016 design (case-control on severe hypoglycemia, n, blinded CGM device); RT-CGM in DiaData (trial, devices, why 96 of 451 admitted subjects have no scored windows, meaning of the r1 collinear fraction 0.305); one-line descriptions of CITY, DCLP3, REPLACE-BG, SENCE, WISDM, PEDAP from `paper/results/diadata_audit_r1.json` and `../DiaData/README.md`; Kistkins et al. 2023 exact recalls, n=11, 15-min data; iOS benchmark CPU backend/thread settings from `ios/` source; the sweep-r1 champion definition per round (from `accuracy_sweep_r*.json`).

### Task 4: Writer A — calibration body (Wave 2, worktree `../GlucoEdge-wA`, branch `rev/writer-a`)

**Owns (edit nothing else):** §4 Data and Methods (incl. Figure 1), §5, the existing paragraphs of §6, §7.1–§7.8, §7.14–§7.15 and their tables/figures, §8.1–§8.2, Appendix `app:history`.

- [ ] Items (Requirement + Acceptance criteria from the spec): R5 (§5.2, §7.8, §8.1; quote the guidance in §8.1), R6 (§7.7 rewrite per frozen predictions), R11, R12, R13 (§8.2 part), R14, R15 (Table 11: drop percentile CIs for the six-participant components; add per-participant contributions and exact sign-flip p), R17 (§8.1 part), R20 (§7.3 and §8.1 parts), S1 (Figure 1 and §4.3 parts: describe the model family, not only the 2,909-parameter CNN), S5 (§7.8), S7 (§6 sentence), S8 parts (move history from §4.3/§5.1/§5.3/§6 and all of §7.14 + Table 20 into `app:history`; Figure 3 legend off the bars; "Macro recall" column labels; restore type sizes in owned tables), S9, S12 (`tab:protocolmap` listing H1–H4 from `protocol.json`, primary, every secondary with status and location; seed/draw variation; sequential 1,000 vs 200 interval), S14 (one sentence + pointer to the JSON), S15, S20 (§4.1 codes + a sensitivity sentence in §7.6), S21 (§7.15), S25, S26, S27, S32, S36, S37 (§7.3 part), S38 (§7.2 participant range).
- [ ] Rename `\section{Results}` to `\section{Results: Calibration Coverage}`; move §7.15 (Deployment costs) to directly after §7.8.
- [ ] `cd paper && tectonic main.tex` succeeds; commit on `rev/writer-a`; report item → location list.

### Task 5: Writer B — sweeps (Wave 2, worktree `../GlucoEdge-wB`, branch `rev/writer-b`)

**Owns:** new `\subsection{Accuracy sweep protocol}\label{sec:sweepprotocol}` at the marker before `\section{Results}`; new `\section{Results: Staged Accuracy Sweeps}\label{sec:sweepresults}` replacing the §7.9–§7.13 region; new `\subsection{What the sweeps show}\label{sec:sweepdiscussion}` at the marker before `\section{Limitations}`; Appendix `app:cohorts` (`tab:cohorts`, `tab:variants`, `tab:readledger`).

- [ ] Move sweep methods (selection rules, gates, endpoints, stable floor, one-read discipline, mixtures, batch size 512, INT8 calibration pool) from §7.9–§7.13 into `sec:sweepprotocol`.
- [ ] Items: R4 (artifact-level wording + seed spread), R7 (`tab:cohorts` + RT-CGM description in the round-three text), R9 (§7.12 part), R13 (§7.12 part), R16, R18, R19 (§7.12 part), R21 (§7.13 part), S1 (`tab:variants`), S2, S8 parts (merge the sweep confirmatory tables into one; reference-model naming), S13 (`tab:readledger`: date, analysis, split, decision — main study through round five), S16, S22, S23, S24, S28 (`tab:shippederrors` + §`sec:sweepdiscussion` paragraph on what a displayed class means), S35, S38 (§7.13 part), optional S39.
- [ ] `tectonic main.tex` succeeds; commit on `rev/writer-b`; report item → location list.

### Task 6: Writer C1 — related work and back matter (Wave 2, worktree `../GlucoEdge-wC`, branch `rev/writer-c`)

**Owns:** §2 (incl. Table 1), §3, §9 Limitations, §10, new file `MODEL_CARD.md`.

- [ ] Items: R2 (§2.4 calibration-data paragraph), R3 (on-device health/glucose paragraph; Zhu et al. row in Table 1; OhioT1DM citation), R6/R18/R19/R20 Limitations parts, S1 (Table 1 row), S10 (cite the existing path; REVIEWER_DISAGREE on "missing"), S11 (non-commercial statement + conventional ethics sentence), S17 (hashes JSON pointer, batch size, local-artifact limitation), S18, S19, S30, S33 (`MODEL_CARD.md` per Mitchell et al.: intended use, out-of-scope uses, inputs, known failure modes, populations not represented; summary sentence in §10.2), S34 (Limitations part), S7 (Limitations count).
- [ ] `tectonic main.tex` succeeds; commit on `rev/writer-c`; report item → location list.

### Task 7: Merge, front matter, build (Wave 3, orchestrator + agent "front")

- [ ] Merge `rev/writer-a`, `rev/writer-b`, `rev/writer-c` into `docs/paper-panel-revision`; resolve conflicts by keeping both owners' edits.
- [ ] Agent "front" rewrites Title, Abstract (S3 length, S4, S5, S13 reuse disclosure, S31 terms, S33 "bundled", S38 "never-evaluated"), §1 (two research questions; contribution list per R1, R2, R4, R5, R8, R9, R18, S4, S5, S37; history trimmed to one sentence pointing to `app:history`), §11 Conclusion (R4, R6, R9, R10, R16, R17, R18, R19, R20, R21, S6, S34), and fixes cross-references to moved sections.
- [ ] Gate: `tectonic main.tex` clean (no undefined refs/cites); abstract ≤ 1,920 plain characters; `grep -n -i "toolchain-default\|clinical threshold\|clinical-shape\|label-noise ceiling\|headroom\|never-read"` returns nothing in `main.tex`.

### Task 8: Verification (Wave 3, three read-only agents in parallel)

- [ ] V1: each R1–R21 acceptance criterion → PASS/FAIL with quoted evidence.
- [ ] V2: each S1–S38 item → PASS/FAIL/REVIEWER_DISAGREE with evidence.
- [ ] V3: every changed number matches its JSON; twins agree across abstract/intro/tables/conclusion; every `\cite` key exists and was verified in Task 2's log.
- [ ] Orchestrator fixes FAILs, rebuilds `main.pdf`, runs `venv/bin/python -m pytest -q`, writes `paper/analysis/panel-review-2026-10-16/response_to_reviewers.md` (item → change → location, or REVIEWER_DISAGREE with evidence), commits.
