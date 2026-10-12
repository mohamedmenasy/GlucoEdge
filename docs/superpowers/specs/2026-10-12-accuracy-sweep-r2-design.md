# Accuracy Sweep r2 — Design

Date: 2026-10-12
Status: approved (sub-project 2 of 2; builds on
2026-10-12-diadata-ingest-audit-design.md and its audit decision note)

## Intent

Sweep r1 (PR #18) won on macro-F1 and lost on directional recall, so it
did not ship; its post-mortem fixed the endpoint. The DiaData audit
(PR #19) cleared 1,001 new T1D subjects with 96.9x weinstock's
fast-class windows. r2 asks the corrected question: does training on
more real data (and how it is mixed) produce a model that beats the
bundled champion on the metric the product exists for — directional
recall — without sacrificing stable detection?

Ship intent (user decision, 2026-10-12): if the frozen gates pass, the
winner replaces the bundled pair via a separate, explicitly-approved
ship PR.

Carried-in decisions (2026-10-12 brainstorm):
- Endpoint: directional recall (macro recall over the four non-stable
  classes), with a stable-recall floor guard.
- Primary confirmatory eval: existing weinstock held_out/temporal,
  paired vs the bundled champion — same machinery as r1.
- Secondary: zero-shot PEDAP (never trained on), descriptive.
- Scope: data mixture x capacity only. Compute: user's Mac GPU (MPS);
  benchmark measured 26k win/s at batch 64 and 236k win/s at batch 1024
  for TrendCNN (96,144); budget cap 15M windows per arm; no cloud.

## Constraints (project-wide, binding)

- Not a medical device: no code, comment, or output string implies
  clinical validity or treatment guidance.
- GlucoBench and DiaData are read-only externals; nothing subject-level
  is committed. Participant IDs, timestamps, window keys, and
  per-window predictions stay in gitignored results/.
- held_out and temporal labels are read exactly once, after the
  protocol file is committed; selection uses val only.
- Test suite runs without GlucoBench or DiaData present.
- Run id: sweep-r2. Branch: feat/accuracy-sweep-r2.

## Section 1: Recipe + smoke gate (frozen before any sweep arm)

- Loss: class-weighted CE (the champion's recipe). r1 settled the loss
  question: distance-weighted CE optimizes against directional recall.
- Fixed study-wide: window length 12, delta-anchor encoding
  (delta_windows(w, anchor=True), ANCHOR_SCALE=16), 5-logit softmax
  head, Adam lr 1e-3, 20 epochs, device MPS
  (torch.manual_seed + np.random.seed per run).
- Batch size: decided by one pre-sweep smoke run — weinstock-only,
  batch 64 vs 512, 2 seeds each. If 512 costs more than 1.0pp mean val
  directional recall vs 64, step down 256 -> 128 -> 64 until within
  tolerance. The passing batch is frozen for every arm and recorded in
  the run directory. The same smoke run records whether MPS runs are
  seed-reproducible (two identical-seed runs compared); if not,
  per-seed checkpoints remain the artifact of record, as in r1.
- Input pipeline: each arm's training windows are pre-materialized once
  into cached tensors (gitignored results/sweep-r2/tensors/), ~720MB at
  the 15M cap, so the GPU is not starved by per-item Python windowing.

## Section 2: Selection metric (all stages)

- Primary: mean val directional recall over 3 seeds (0,1,2), computed
  on the SAME fixed eligible weinstock val set as r1 (length-12
  windows; eligibility rules unchanged). Mixtures change training data
  only; val never moves.
- Tie rule: within 0.1pp, the cheaper arm wins
  (param_count, then windows trained on, then arm index).
- Disqualifier: an arm whose mean val stable recall is more than 5pp
  below the champion's val stable recall cannot win its stage,
  whatever its directional number. (The champion's val numbers are
  computed once from the bundled INT8 on the same val set; val is not
  a protected split.)

## Section 3: Stages

- **Stage M — mixtures.** 5 arms x 3 seeds at channels (96,144), each
  capped at 15M training windows via seeded subject-level sampling
  (whole subjects drawn per cohort; a subject's windows are never
  split across the in/out boundary):
  - M0: weinstock eligible train only (control; all ~330k windows).
  - M1: weinstock + the 5 training cohorts (CITY, DCLP3, ReplaceBG,
    SENCE, WISDM), natural proportions up to the cap.
  - M2: weinstock + the 5 cohorts, equal window budget per cohort
    (cap/6 each).
  - M3: the 5 DiaData cohorts only, no weinstock (does weinstock in
    training matter when the eval is weinstock?).
  - M4: M1 proportions + fast-class oversampling — windows labeled
    falling_fast/rising_fast are duplicated until each reaches the
    mean count of the other three classes (the class-imbalance arm).
- **Stage C — capacity.** Winning mixture at channels (24,36) and
  (192,288), 3 seeds each, bracketing (96,144).
- **Stage S — seeds.** 2 extra seeds (3,4) of the overall winner;
  final candidate = best single val seed of the 5. No ensembling (r1
  answered it: no gain).
- **PEDAP appears in no training mixture.** It is the untouched
  secondary cohort.
- Budget: ~27 trainings; at the measured MPS rates each stage completes
  in hours.

## Section 4: Calibration, conversion, protocol freeze

- Winner seeds export float + INT8 TFLite with the r1 calibration
  selections rebuilt at length 12: sequential_200_d0 and
  uniform_random_200_d101. The candidate INT8 is the uniform one; the
  sequential one is reported.
- `paper/results/calibration-coverage/protocol_sweep_r2.json` is
  committed BEFORE any held_out/temporal label is read. Same guard as
  r1: evaluate refuses unless the protocol is git-tracked and clean,
  and the refusal is verified live pre-freeze. The protocol carries:
  run id, spec path, freeze date and commit, the full selection trail,
  the winner, the bundled champion's sha256s, endpoints, gates,
  bootstrap recipe (participant-cluster, 2,000 draws, seed 20261012),
  and the pairing rule (intersection on (participant, anchor_time);
  dropped-window counts reported for BOTH sides).

## Section 5: Confirmatory read + ship gates

- Primary endpoint: held_out paired directional recall, winner INT8
  (uniform) vs champion INT8.
- Gates (all must pass to ship):
  1. Directional-recall delta > 0 AND bootstrap CI lower bound
     > -0.5pp.
  2. Stable floor: winner INT8 held_out stable recall >= champion INT8
     stable recall - 2.0pp.
  3. Winner INT8 within 0.5pp of its own float on held_out directional
     recall.
- Secondary, descriptive, no gates: temporal split (same recipe);
  macro-F1 on both splits (connects r1 and r2 tables); zero-shot PEDAP
  directional recall and per-class recall (its own windows, no champion
  pairing — the champion never saw PEDAP either, so its numbers are
  reported too).
- Outcomes: gates fail -> MODELS.md one-liner + paper reports the
  result; gates pass -> STOP and surface to the user before any ship
  work. The ship PR (separate branch, after the research PR merges)
  regenerates assets + goldens + sha pins together, both apps, as
  always.

## Section 6: Publishing

- `paper/results/calibration-coverage/accuracy_sweep_r2.json`:
  aggregates only (stage tables, contrasts, gate booleans, PEDAP
  numbers); verified free of subject-level data before commit.
- Paper: one subsection after the r1 sweep subsection — stage val
  table, confirmatory contrast with CI, gate outcomes, PEDAP zero-shot
  row; states val-only selection and the single pre-registered read;
  page budget 15 (trim elsewhere if it spills).

## Section 7: Infra (reuse, don't rewrite)

- New `experiments/accuracy_sweep_r2.py` importing the reusable pieces
  of `experiments/accuracy_sweep.py` (protocol guard, pairing,
  bootstrap, TFLite logits, conversion drive) — extracted into a shared
  module if importing is awkward, with r1's entry points left working.
- Data: `training/diadata.py::load_diadata` + `GlucoseTrendDataset`
  feed the mixtures; the mixture sampler and oversampler are new, small
  and pure.
- `training/train.py::train_model` gains device/batch knobs with
  backwards-compatible defaults (existing callers unchanged).

## Section 8: Testing

- Mixture sampler: subject-level draws, seeded determinism, cap
  respected, PEDAP never present, a subject never split.
- Oversampler: hand-computed duplication counts per class.
- Directional recall + stable floor + gate functions: hand-computed
  confusion-matrix cases, boundary values at the gate edges.
- Protocol guard: reuse r1's temp-git test pattern against the r2
  protocol path.
- All tests run without GlucoBench/DiaData.

## Out of scope

- New losses, window lengths, pooling variants beyond the five
  mixtures, ensembling, QAT, RT-CGM reopening, CORN/CORAL heads.
- Any android/ios change before an explicitly-approved ship PR.
- A second confirmatory read of held_out/temporal under any
  circumstances.

## Open items the smoke run settles

- Frozen batch size (64/128/256/512).
- MPS seed reproducibility (recorded either way).
