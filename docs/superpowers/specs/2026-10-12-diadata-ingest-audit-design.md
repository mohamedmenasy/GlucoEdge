# DiaData Ingest + Audit — Design

Date: 2026-10-12
Status: approved (sub-project 1 of 2; sub-project 2 = accuracy sweep r2,
spec'd only after this audit's output exists)

## Intent

The accuracy sweep (sweep-r1, PR #18) showed the classifier is
data-starved, not capacity-starved: its weakest numbers are the minority
fast classes and OOD participants, and weinstock's 200 subjects are the
largest cohort GlucoBench offers. DiaData v3 (Zenodo record 17285631,
CC BY-NC 4.0) integrates 13 T1D studies — 1,720 subjects — into one
open-download collection. This sub-project brings it in as an external,
never-committed data source and audits what actually survives our
correctness filters, so the r2 sweep spec can be written from measured
counts instead of hopes.

Phase goal (user decision, 2026-10-12): a better shipped model. r2's
endpoint will be directional recall with a stable-recall floor guard;
primary confirmatory evaluation stays the existing weinstock
held_out/temporal (paired vs the bundled champion, same machinery as
r1), with held-out DiaData cohorts as a secondary generalization
report. Those are r2 decisions recorded here for continuity; nothing in
this sub-project touches evaluation.

Decision rule this audit feeds: **if the included cohorts' fast-class
window count (length-12 sliding windows over our recomputed segments
whose `label_trend` is `falling_fast` or `rising_fast`) is < 2x
weinstock's own count under the same rule, stop and reassess before
writing any r2 spec.** The extra data must materially address class starvation
or r2 is not worth its compute.

## Constraints (project-wide, binding)

- Not a medical device: no code, comment, or output string implies
  clinical validity or treatment guidance.
- DiaData is read from, never modified, never committed — same rules as
  GlucoBench. No subject-level rows, IDs, or timestamps ever enter the
  repo; committed artifacts carry aggregates only.
- CC BY-NC 4.0 is compatible with this portfolio/research use; the data
  is never redistributed. Attribution lands in the paper's
  data-availability subsection when r2 publishes.
- Test suite must keep running without DiaData or GlucoBench cloned
  (lazy imports; synthetic frames in tests).

## Section 1: Acquisition & layout

- DiaData v3 lives at `../DiaData/` — a sibling directory like
  `GlucoBench/`. Only the 5-minute-sampled archive is extracted.
- `.gitignore` gets `DiaData` (no trailing slash — symlink-safe in git
  worktrees, same as the GlucoBench entry).
- CLAUDE.md's commands block documents the one-time fetch: curl from
  Zenodo record 17285631, unzip, expected top-level layout. The zip's
  sha256 is recorded in the audit output for provenance.
- README's data-sources line gains DiaData next to GlucoBench, with the
  license named.
- The loader is read-only; nothing ever writes into `../DiaData/`.

## Section 2: Loader + windowing

- New module `training/diadata.py` with one public function:

  ```python
  load_diadata(root: Path, include_cohorts: Sequence[str]) -> pd.DataFrame
  # columns: id, id_segment, time, gl
  ```

  This is the exact frame shape `GlucoseTrendDataset` consumes, so all
  windowing/labeling code is reused untouched. No new Dataset class.
- `id = "{source_dataset}:{subject_id}"` — keeps cohorts separable for
  r2 mixtures, held-out-cohort evaluation, and overlap exclusion.
- **Native-5-min filter is cohort-level, by documented sensor.**
  DiaData's 5-min archive resamples every cohort to 5 minutes, so
  timestamp deltas cannot reveal interpolation. The audit module holds a
  hand-written, citation-backed table mapping each of the 13 source
  studies to its sensor and native interval; only native-5-min cohorts
  (Dexcom-class) are eligible. Libre/15-min studies (e.g., HUPA-UCM,
  ShanghaiT1DM) are excluded outright — interpolated rates fake the
  rate-of-change labels, deleting exactly the fast classes we need.
- Cross-check heuristic: per cohort, the fraction of consecutive reading
  triples that are exactly collinear (interpolated runs are perfectly
  linear). A nominally-5-min cohort scoring anomalously high is flagged
  and excluded (`excluded_flagged`).
- `id_segment` is recomputed by us: a new segment starts whenever the
  delta between consecutive readings exceeds 5 minutes + 30 seconds
  tolerance. DiaData's own gap handling is ignored.
- **Overlap exclusion is cohort-level and decided in the audit.**
  DiaData's sources include "Severe Hypoglycemia in Older Adults with
  T1D" — plausibly the same study as GlucoBench's weinstock2016 — and
  "RT-CGM". Any source study that matches a GlucoBench cohort is
  dropped entirely: training on held-out participants is leakage and
  would invalidate the r2 confirmatory read. The audit prints a
  per-source verdict with evidence (participant counts, study-period
  date ranges, citations).

## Section 3: Audit CLI + published output

- `experiments/audit_diadata.py`, read-only, one command:
  `python -m experiments.audit_diadata`. Writes:
  - `results/diadata_audit/audit_r1.json` — full local record
    (gitignored).
  - `paper/results/diadata_audit_r1.json` — committed; aggregates only.
- Per source cohort: sensor + documented native interval (with
  citation), subject count, hours, segment count after our gap rule,
  collinear-triple fraction, verdict
  (`included` / `excluded_interval` / `excluded_overlap` /
  `excluded_flagged`), and — for included cohorts — the 5-class
  `label_trend` distribution (counts + fractions) and fast-class totals
  alongside weinstock's own for comparison.
- Overlap section: per suspicious pair (DiaData source vs GlucoBench
  cohort), the evidence and the verdict. Never subject IDs.
- Top-level summary: total included subjects / hours / windows and the
  fast-class multiplier vs weinstock alone — the number the decision
  rule reads.
- Provenance: DiaData version and zip sha256.

## Section 4: Testing

- Loader tests on synthetic frames (no DiaData needed): segment splits
  at the gap rule, cohort include-filtering, id format, column contract.
- Audit pure-function tests: collinear-triple fraction on constructed
  sequences, verdict assignment, label-distribution aggregation.
- The audit run itself is validated by reading its output against the
  DiaData paper's published counts (like train.py runs — not unit
  tested).

## Deliverables

- `training/diadata.py` + tests.
- `experiments/audit_diadata.py` + tests.
- `.gitignore`, CLAUDE.md, README lines.
- `paper/results/diadata_audit_r1.json` (committed aggregates).
- A short decision note appended to this spec (or a dated sibling note)
  recording the audit verdicts and whether the 2x decision rule passed.

## Out of scope

- Any training or sweep arm; any r2 protocol or gate values.
- Any paper prose beyond the data-sources line.
- Any app (android/ios) change.
- Downloading/handling any archive other than the 5-minute one.
- Per-subject interpolation detection (impossible post-resample;
  cohort-level is the honest unit).

## Open questions deferred to r2 spec

- Exact mixture arms and ratios (depend on audit counts).
- Directional-recall gate values and the stable-recall floor margin.
- Whether (96,144) capacity re-wins under the new mixture.

## Audit decision note (2026-10-12, run diadata-audit-r1)

Included cohorts: CITY, DLCP3, PEDAP, RBG (ReplaceBG), SENCE, WISDM —
all JAEB Dexcom native-5-min studies. Included subjects: 1,001; hours:
4,720,389; total windows: 51.6M.
Fast-class windows: 3,911,376 vs weinstock 40,374 — multiplier 96.9x.
(The first run's weinstock baseline of 38,998 counted GlucoBench's
duplicated context rows; the final review caught it, the concat is now
deduped on (id, id_segment, time), and the audit was rerun. The same
study through the DiaData pipeline (SHD) gives 31,091 — the rule
passes under every baseline, 96.9x to 125.8x.)
Decision rule (>= 2x): PASSED by two orders of magnitude. Proceed to the
r2 spec.
Excluded: D1NAMO (overlap: GlucoBench dubosson2018), SHD (overlap:
GlucoBench weinstock2016 — exactly 200 subjects, confirming the match),
RT-CGM (mixed sensors, per-subject interval not recoverable), HUPA-UCM +
ShanghaiT1D (Libre 15-min native), DDATSHR + T1GDUJA (sensor
undocumented). Collinear-triple fractions for included cohorts ran
0.125-0.229 (integer-quantized Dexcom baseline), all far below the 0.5
flag threshold; no cohort was excluded_flagged.
Raw (non-imputed) variant used throughout; preprocessed variant never
extracted. Zip sha256
2271dd6b2806f2952916b5a99ced09afae689346053e261173691d40c3b543c9.
