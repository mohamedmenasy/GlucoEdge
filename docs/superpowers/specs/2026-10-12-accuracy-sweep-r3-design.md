# Accuracy Sweep r3 — RT-CGM reopen, time-of-day channel, TCN arm

**Date:** 2026-10-12
**Status:** approved design (brainstorm 2026-10-12, user chose: full ship
intent incl. client changes; 5 seeds per arm; staged greedy structure)
**Builds on:** `2026-10-12-accuracy-sweep-r2-design.md` (merged, winner
shipped as the bundled pair), `2026-10-12-diadata-ingest-audit-design.md`
(`diadata_audit_r1.json`)

## 1. Intent and success criteria

r2 proved the data lever: DiaData-only training beat the Weinstock-trained
champion on the pre-registered directional endpoint (+0.60pp held-out,
+1.9pp PEDAP zero-shot). r3 pulls the three remaining levers ranked by
expected value: more admitted data (RT-CGM), a circadian input feature
(time-of-day), and one architecture alternative (dilated TCN). Ship
intent: if the winner passes the gates, it replaces the bundled pair —
including client-side encoder changes if a time-of-day arm wins.

Success = a pre-registered confirmatory read against the shipped r2
champion with the same three gates; an honest null is an acceptable
outcome and is published.

Seeds: **5 per arm** at selection (r2 used 3; its seed noise band ±1.4pp
exceeded the shipped +0.6pp effect, so 3-seed means cannot resolve the
effect sizes r3 hunts).

## 2. Stage 0 — RT-CGM reopen audit

RT-CGM (JDRF CGM RCT, NEJM 2008): 451 subjects, 1.51M hours, excluded in
audit r1 as `excluded_interval` — sensors mixed (DexCom SEVEN / Paradigm
/ FreeStyle Navigator), per-subject sensor not recoverable, Navigator
samples at 1 min.

Reopen rule (cadence, not brand, is what the 5-minute window geometry
needs): per subject, over consecutive timestamp differences within
segments,

- **admit** iff the modal interval is 300s AND ≥95% of that subject's
  intervals lie in [270s, 330s];
- reject otherwise.

Output `paper/results/diadata_audit_r2.json`: admitted/rejected subject
counts, aggregate interval histogram (binned counts only), label-window
counts for the admitted pool, and the rule stated verbatim. No
participant IDs, timestamps, or per-subject rows leave gitignored
results/. Admitted subjects form pseudo-cohort **`RTCGM5`**, appended to
`training/mixtures.py` `TRAIN_COHORTS` for the D stage only. RT-CGM
rejected subjects and PEDAP never enter training; PEDAP remains the
zero-shot probe.

Expected (not binding): Navigator users rejected; roughly 200–350 of 451
admitted.

## 3. Sweep structure — staged greedy, 5 seeds, 15M cap

All selection on weinstock validation eligible windows (unchanged
denominator from r1/r2); stable-recall disqualifier identical to r2
(mean val stable < champion val stable − 5pp → arm disqualified; all
arms disqualified → loud failure, no winner). Whole-subject sampling,
`SAMPLE_SEED` fixed, cap **15,000,000 windows per arm** (unchanged for
comparability; overshoot by at most one subject, as r2, stated as
"near 15M").

- **Stage D (data):** `D_base` = shipped recipe exactly (DiaData-5
  cohorts, (96,144), weighted CE, delta-anchor, window 12) vs `D_rtcgm`
  = DiaData-5 + RTCGM5, proportional whole-subject sampling under the
  same cap. 2 arms.
- **Stage F (features):** Stage D winner ± time-of-day channels (§4).
  2 arms (the winner cell reuses Stage D checkpoints; only the ToD cell
  trains).
- **Stage A (architecture):** Stage F winner on TrendCNN vs dilated TCN
  (§5). TCN cell trains; CNN cell reuses.
- **Stage S:** best single seed of the 5 by val directional recall
  (stable floor still binding); no ensembles (failed in r1).

Winner ties resolve by r1 `cost_key` then list order (deviation from
"windows trained on" tie-break carried from r2; no tie within 0.1pp has
ever occurred).

Tensor caches are keyed by **mixture + feature flag + sample seed**
(`{mixture}_{feat}_s{seed}.npz`), fixing r2's deferred minor (arm-name
keys could silently reuse stale data and duplicated 76MB caches).

Training: MPS, frozen batch from a fresh smoke stage (r2 recipe: accept
largest batch within 1pp of batch-64 mean, verify same-seed
reproducibility, write smoke.json), 20 epochs, weighted CE on the
sampled labels. Per-checkpoint provenance sidecars (added late in r2)
are written from the first training this round.

## 4. Time-of-day input channels

Input tensor grows [1,12] → **[3,12]**:

- channel 0: delta-anchor encoding, unchanged (11 consecutive
  differences + last reading / 16, in the same positions as today);
- channel 1: sin(2π·h/24) per timestep;
- channel 2: cos(2π·h/24) per timestep;

where h = fractional hour-of-day of each reading's timestamp. The
sin/cos pair is continuous across midnight. `TrendCNN` (and the TCN)
take an `in_channels` constructor parameter, default 1 — all existing
call sites and tests unchanged.

**Stated assumption (spec + paper):** timestamps in GlucoBench and
DiaData are device-local clock time; no timezone normalization is
attempted or possible. Circadian signal survives this assumption;
cross-timezone noise is accepted and disclosed.

Ship implication (accepted at brainstorm): if a ToD arm wins the gates,
both clients' `DeltaEncoding` add the two channels (Kotlin + Swift),
golden-vector schema v2 carries per-window timestamps, parity suites
re-pin, and the pair is device re-verified before merge.

## 5. TCN arm

`training/model.py` gains `TrendTCN` beside `TrendCNN`: causal dilated
1D convolutions, 4 residual blocks, dilations (1, 2, 4, 8), kernel 3,
96 channels, ReLU, global-average-pool head → 5 logits,
`in_channels` parameter as §4. Receptive field 31 > 12 — the whole
window with temporal hierarchy. Parameter count within 2× of the
(96,144) TrendCNN (~43k) so the arm tests architecture, not capacity;
the test suite pins this bound. LiteRT conversion of a TCN checkpoint is
smoke-tested in unit tests (dilated Conv1d is standard ops).

## 6. Endpoint, gates, protocol (unchanged discipline)

- Primary: held-out directional recall (minority-macro recall over the
  four non-stable classes), winner INT8 paired vs **champion = the
  shipped r2 pair** (float sha `96eef43c…`, INT8 sha `a887ea4e…`),
  participant-cluster bootstrap, 2,000 draws, **pre-registered seed
  20261013**.
- Gate 1: delta > 0 AND CI lower bound > −0.5pp.
- Gate 2: winner INT8 held-out stable recall ≥ champion INT8 − 2.0pp.
- Gate 3: winner INT8 within 0.5pp of its own float (directional).
- Secondary: temporal split same recipe; macro-F1 both splits; PEDAP
  zero-shot descriptive (never trained).
- Calibration: `sequential_200_d0` + `uniform_random_200_d101` as r2;
  uniform INT8 is the ship candidate.
- `protocol_sweep_r3.json` committed clean before any held-out /
  temporal / PEDAP label read; `stage_evaluate` refuses otherwise; the
  protocol carries the full selection trail, smoke record, champion
  sha256s, prediction, endpoints, gates, and the pairing rule (windows
  are identical by construction; dropped-window counts reported — r2
  deferred minor closed).
- The bootstrap seed constant lives in one place and the protocol test
  asserts the code uses it (r2's seed slip cannot recur silently).

## 7. Publication and ship

- `paper/results/calibration-coverage/accuracy_sweep_r3.json`: aggregates
  only (per-arm train-window counts included from the start), protocol
  pointer, deviations block if any; participant-leak walker before
  commit.
- Paper: r3 subsection after r2's, one table, page budget 15.
- Ship decision stops for the user in all cases. Contract-unchanged
  winner → r2-style ship (assets + goldens + shas + both suites +
  device verification). ToD winner → extended ship (§4). TCN winner →
  assets-only ship (graph change invisible to clients). No gates passed
  → MODELS.md one-liner, pair retained.

## 8. Testing requirements

- Interval-inference admit/reject: synthetic per-subject timestamp
  patterns (clean 5-min; 1-min Navigator-like; 5-min with >5% jitter
  outside tolerance; empty subject) — hand-computed verdicts.
- ToD channels: hand-computed sin/cos at known hours incl. midnight
  wrap; channel alignment with delta positions.
- `TrendTCN`: output shape, param bound vs TrendCNN, receptive field,
  `in_channels` plumbing; LiteRT conversion smoke.
- Cache key: distinct paths across mixture / feature flag / sample seed.
- Stage logic: reuse-vs-train cells per stage; disqualifier; loud
  all-disqualified failure (ports of r2 tests where signatures change).
- Protocol guard refusal pre-freeze; bootstrap-seed assertion.
- Audit JSON leak walker on both published files.
- Suite runs via `venv/bin/python -m pytest`; no CGM data needed for any
  unit test.

## 9. Out of scope

Personalization / on-device adaptation (separate brainstorm); window
lengths other than 12 (settled r1); losses other than weighted CE
(settled r1/r2); ensembles (settled r1); capacity changes (settled r2);
cohorts rejected by both audits; any clinical claim.
