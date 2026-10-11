# Accuracy Sweep: Staged Loss / Capacity / Window / Pooling / Ensemble Study

**Date:** 2026-10-11
**Status:** Approved (design sections approved in-session 2026-10-11)
**Run id:** `sweep-r1`

## Intent

Raise held-out macro-F1 of the on-device 5-class trend classifier above the
current champion (the bundled delta-anchor pair from the D12 study:
float 0.5088 held-out macro-F1, INT8 within gate). Five interventions are
tested in one staged study — ordinal-aware losses, model capacity, input
window length, pooled multi-cohort training, and seed ensembling — under the
repo's locked-evaluation discipline. Research first; the winner ships only if
it passes frozen gates.

**Who it is for:** the same portfolio narrative as the calibration-coverage
and D12 studies — a defensible, pre-registered accuracy improvement with the
bundled models updated only on measured evidence.

**Success:** a confirmatory held-out read, protocol frozen first, that either
upgrades the bundled pair or documents a null result honestly.

## Non-negotiable discipline

- The weinstock `held_out` split (GlucoBench `test_idx_ood`) and `test`
  (temporal) split are read **once**, at the end, for the final winner vs
  champion contrast. Every selection decision uses weinstock **val** only.
- `protocol.json` for the confirmatory read is committed **before** any
  scored held-out/temporal label is read, naming the final winner
  configuration, the champion artifacts, the prediction (winner ≥ champion),
  the gates, and the bootstrap recipe.
- All stages log their val tables to the run directory as they complete;
  published JSON carries the full selection trail so the greedy path is
  auditable.
- GlucoBench is read, never modified. Participant ids, timestamps, window
  keys, per-window predictions stay in gitignored `results/`.

## Fixed study-wide choices

- Inputs: delta-anchor encoding (`training/deltas.py`, `anchor=True`,
  `ANCHOR_SCALE=16`), applied to absolute windows exactly as in D12.
- 5 classes, horizon 3 rows / 15 minutes, eligibility = scored +
  no-gap-fill windows per `training/splits.py` (unchanged).
- Selection metric: **mean val macro-F1 over seeds 0,1,2** on eligible
  weinstock val windows (sklearn `classification_report` macro avg,
  `zero_division=0`). Final winner retrains/extends to seeds 0–4.
- Training recipe otherwise frozen to the study baseline: Adam lr=1e-3,
  batch 64, 20 epochs, CPU, `torch.manual_seed + np.random.seed`.
- Ties (within 0.1pp mean val macro-F1): the cheaper arm wins (fewer
  params, shorter window, no pooling, no ensemble, simpler loss — in that
  order of preference).

## Stage A — loss (12 trainings)

Four arms, current architecture (channels 24/36), window 12. All keep the
5-logit softmax head — no CORN/CORAL head, zero app decode change. `w` below
is the balanced class-weight vector already computed by
`training.train.class_weights`.

1. **ce** (baseline, current): weighted cross-entropy.
2. **ordinal-smooth**: cross-entropy against a smeared target — true class
   gets `1 − ε`, each ordinal neighbor gets `ε/2` (a boundary class gives
   the full `ε` to its single neighbor), `ε = 0.15`. Sample loss multiplied
   by `w[y]`.
3. **distance-ce**: weighted cross-entropy plus expected ordinal distance,
   `L = CE_w + λ · Σ_j p_j · |j − y|`, `λ = 0.5`.
4. **focal**: `L = −w[y] · (1 − p_y)^γ · log p_y`, `γ = 2.0`. This arm is
   the class-imbalance intervention; no separate oversampling arm.

Winner by selection metric carries into Stage B. Losses live in
`training/losses.py` with unit tests (hand-computed small-tensor cases).

## Stage B — capacity × window (15–24 trainings)

Grid: `TrendCNN` channels `(24,36)` / `(48,72)` / `(96,144)` × input length
`{12, 24}`, Stage A's winning loss. `TrendCNN.__init__` gains a
`channels=(24, 36)` parameter; default preserves the current architecture
byte-for-byte. Input length is `GlucoseTrendDataset(input_length=...)`;
delta input length = `input_length − 1 + 1` (anchor).

Before training, compute eligible train/val window counts at length 24 and
36. Length 36 joins the grid only if its eligible train count is ≥ 80% of
length 12's; length 24 runs regardless but the same count is recorded.
The arm `(24,36) × 12` is Stage A's winner re-used, not retrained.

Winner carries forward. If the winning window length ≠ 12, the ship cost
(replay buffer + `DeltaEncoding` length change in both apps) is accepted —
noted, not a gate.

## Stage C — pooling (3 trainings)

Winner config trained on weinstock-only (already have it — reuse) vs pooled
eligible train windows from weinstock + dubosson + hall (each cohort's own
`load_cohort_context` provenance rules; delta encoding makes levels
comparable across cohorts). Class weights computed on the pooled train set.
Selection on **weinstock val only** — pooling wins only if it does not hurt
the primary cohort.

## Stage D — ensemble (2 trainings)

Winner config extended to seeds 0–4 (seeds 3 and 4 are the new trainings). Compare on val: mean-logit ensemble of
all 5 seeds vs the best single seed. Ensemble carries into the final only if
its val macro-F1 gain over the best single seed is **≥ +0.5pp**.

## Final confirmatory read (the one held-out touch)

1. Convert the final winner to LiteRT per the D12 recipe
   (`conversion/`, PT2E INT8): float + INT8 with the shipped calibration arm
   `uniform_random_200` (draw seed d101) and the diagnostic
   `sequential_200`, calibration inputs delta-transformed and length-matched.
   Ensemble = one artifact pair per seed.
2. Commit `protocol.json`: winner config, champion artifact sha256s (bundled
   `trend_float.tflite` `c30560f7…`, `trend_int8.tflite` `8590a3d2…`),
   prediction, gates, bootstrap recipe (paired participant-cluster
   bootstrap, 2,000 draws, seed 20261010), primary endpoint = held_out
   macro-F1, secondary = temporal.
3. Evaluate winner and champion on eligible held_out and temporal windows.
   If window lengths differ, the paired contrast uses the intersection of
   eligible windows keyed by `(participant, anchor_time)`; windows only one
   model can score are excluded from the pairing and their count reported.
   Ensemble inference = mean of per-seed logits (float) / dequantized
   logits (INT8).

## Ship gates (frozen in protocol.json before the read)

1. Winner INT8 held-out macro-F1 − champion INT8 > 0, **and** the paired
   bootstrap CI lower bound > −0.5pp.
2. Winner INT8 within 0.5pp of the winner's own float (D12 gate).
3. Ensemble ships only if it passed Stage D's margin **and** beats the
   champion by ≥ +1.0pp on gate 1's point estimate; otherwise the best
   single-seed model ships even if the ensemble was the research winner.

Any gate fails → research-only: paper section + published JSON, bundled
models untouched.

## If it ships

- Separate ship PR from the research branch (D12 pattern): new `.tflite`
  pair (or K pairs under gate 3) + golden vectors regenerated together,
  sha256s re-pinned in both apps' parity tests, MODELS.md provenance.
- Window ≠ 12: replay buffer size and `DeltaEncoding` expected length change
  in Kotlin + Swift, with unit tests updated RED→GREEN.
- Ensemble: logit averaging in both `TrendClassifier`s, parity pins every
  bundled artifact.
- Device verification (iPhone parity run) when the user attaches the phone;
  ship PR does not wait on it, the verification is recorded after, as with
  D12.

## Deliverables

- `training/losses.py` + tests; `TrendCNN(channels=...)` + test that the
  default matches the current architecture's parameter count (2,909).
- `experiments/accuracy_sweep.py`: stages `precheck` (window-length
  eligibility counts), `train` (stage-aware, resumable by run-id +
  arm name, same skip-if-exists pattern as `calibration_coverage`),
  `select` (writes each stage's val table + winner), `convert`, `evaluate`
  (refuses to run unless protocol.json is committed), `analyze`.
- Published `paper/results/calibration-coverage/accuracy_sweep_r1.json`:
  per-stage val tables, selection trail, confirmatory contrast, gate
  outcomes. No participant-level data.
- Paper: one subsection + one table (stage winners' val numbers + the
  confirmatory held-out/temporal contrast with CI).
- Memory/project_status update at the end.

## Out of scope

- CORN/CORAL or any output-head change; QAT; architecture families other
  than TrendCNN with wider channels; time-of-day or any non-glucose input
  features; changes to labeling thresholds; a second confirmatory read of
  any kind.
