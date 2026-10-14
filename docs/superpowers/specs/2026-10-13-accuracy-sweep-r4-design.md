# Accuracy Sweep r4 — Noise Ceiling, Objective Heads, Robustness, Weight Averaging

**Date:** 2026-10-13
**Status:** approved design (brainstorm 2026-10-13, user chose: full ship
intent incl. client decode changes; Experiment 0 as a binding abort gate;
three stages H/R/W; 5 seeds per arm)
**Builds on:** `2026-10-12-accuracy-sweep-r3-design.md` (null result, r2
pair retained), r3 error analysis (2026-10-13, this session)

## 1. Intent and evidence

r3's held-out confusion structure, measured from the published run's saved
predictions (champion INT8, 46,376 eligible held-out windows):

- 72% of all errors are adjacent-class;
- slow-class errors go to the same-direction fast class (47% falling,
  46% rising), not to stable (25% each) — the model over-estimates rate
  magnitude, consistent with balanced inverse-frequency class weights
  inflating rare fast-class logits;
- stable recall 0.459 — nearly half of stable windows get a spurious
  direction;
- per-seed held-out directional spread 1.1pp; the 5-seed mean-logit
  ensemble (0.5304) ≈ best single seed (0.5306), confirming r1: ensembling
  buys nothing, variance reduction might.

r4 therefore attacks the **objective** (Stage H), **target/input noise**
(Stage R), and **seed variance** (Stage W), after first measuring whether
label noise leaves any headroom to win (Experiment 0).

**Scope reversal, stated:** r1 settled "losses other than weighted CE" —
but under the old macro-F1 selection endpoint. The endpoint changed to
directional recall in r2. r4 deliberately reopens objective design under
the current endpoint; this is a recorded decision, not drift.

Error-structure provenance caveat: the confusion analysis above is a
descriptive post-hoc read of already-saved r3 eval artifacts. It informs
arm *choice*; all r4 *selection* remains val-only, and the single
confirmatory held-out read happens once, at the end, under the frozen
protocol. The analysis is reported in the paper as motivating evidence.

## 2. Experiment 0 — label-noise ceiling (binding abort gate)

Current labels are endpoint differences: `rate = (g[t+3] − g[t]) / 15`
(`training/labeling.py::label_trend`). Exp0 measures how much of the task
is sensor noise at the class boundaries:

- **Smoothed label:** least-squares slope over the four horizon readings
  `g[t..t+3]` (same clinical thresholds, same asymmetric boundary rules:
  falling side `≤`, rising side `<`).
- **Ceiling estimate:** on weinstock val eligible windows, treat the
  smoothed label as a predictor of the endpoint label and compute
  directional recall. This approximates the best any rate-faithful
  predictor can score against endpoint-labeled eval.
- **Descriptive companion:** the same agreement under Gaussian
  perturbation of the levels at MARD scale (σ ∈ {1, 2} mg/dL), reported,
  not gated.

**Abort rule (pre-registered in the protocol before any arm trains):**
`ceiling − 0.5330` (champion val directional) `< 2.0pp` → r4 stops;
Experiment 0 is published as the round's result (paper subsection +
aggregates JSON), and no training arm runs. Otherwise the sweep proceeds
and the ceiling is reported alongside the stage tables.

Exp0 reads **val only** (val is not a protected split; the protected
held-out/temporal splits are untouched).

## 3. Sweep structure — staged greedy H → R → W, 5 seeds, 15M cap

Carried unchanged from r3: selection on weinstock val eligible windows
(same denominator as r1–r3); per-stage stable-recall disqualifier (mean
val stable < champion val stable − 5pp); whole-subject sampling,
`SAMPLE_SEED` fixed, 15,000,000-window cap per arm ("near 15M" overshoot
rule); tensor caches keyed `{mixture}_{feat}_s{seed}.npz`; MPS training,
frozen batch from a fresh smoke stage, 20 epochs, Adam lr 1e-3;
per-checkpoint provenance sidecars; tie rule: within 0.1pp, `cost_key`
then list order — **applied at every stage including seed selection**
(fixes r3's Stage S deviation).

Data recipe for every arm: the shipped r2 champion mixture (DiaData-5
cohorts, channels (96,144), delta-anchor encoding, window 12). PEDAP
never trains; it stays the zero-shot probe. RT-CGM is not used (r3: null).

### Stage H — objective (4 arms)

- **H_ce** — control: shipped recipe exactly (weighted CE). Checkpoints
  reuse r3 `D_base` (= r2 M3) seeds 0–4 under the bit-identical smoke
  reproduction rule; provenance note carried as in r3.
- **H_wtemp** — class weights `w_c = (1/freq_c)^0.5`, normalized the same
  way as the current balanced weights (which are ∝ 1/freq_c); weighted CE
  otherwise unchanged. Output contract unchanged (5 logits).
- **H_reg** — regression head: `num_outputs=1`, target = true rate in
  mg/dL/min, Huber loss (δ = 1.0). Decode: thresholds at −2/−1/1/2 with
  the labeler's boundary rules. Selection metrics computed on decoded
  classes. Output contract: 1 scalar.
- **H_corn** — CORN ordinal head: 4 sigmoid outputs, target = ordinal
  level (0–4), CORN conditional training loss; class weights folded in by
  sample weighting `w[y]`. Decode: count of outputs > 0.5. Output
  contract: 4 logits.

### Stage R — robustness on the H winner (3–4 arms)

- **R_base** — H winner reused (no retraining).
- **R_labels** — training labels from the LSQ slope over `g[t..t+3]`
  (Exp0's estimator); **eval labels unchanged everywhere** (val,
  held-out, temporal, PEDAP all stay endpoint-labeled, preserving champion
  comparability).
- **R_noise** — train-time Gaussian noise σ = 2 mg/dL added to levels
  before delta encoding, fresh draw per epoch, seeded; eval inputs clean.
- **R_mixup** — mixup α = 0.4 on (window, rate) pairs; **this arm exists
  only if H_reg won Stage H** (mixing is linear in the regression target;
  with discrete heads it is not run).

### Stage W — weight averaging on the R winner (3 cells, no new configs)

- **W_best** — best single seed by val directional (control; tie rule as
  above).
- **W_soup** — uniform weight average of the 5 seed checkpoints (TrendCNN
  has no BatchNorm — plain parameter mean, no stat recompute).
- **W_swa** — per seed, average the last 5 epoch checkpoints; then best
  seed. Per-epoch checkpoint retention is enabled for every arm r4
  itself trains (H and R stages). If the R winner is a reused-checkpoint
  cell with no epoch history (H_ce control carried through R_base), the
  W_swa cell is skipped and reported as not-applicable — W_best and
  W_soup still run.

Stage W selects on val like any stage; the winner cell's artifact is the
confirmatory candidate.

## 4. Endpoint, gates, protocol (carried discipline)

- Primary: held-out directional recall, winner INT8 paired vs **champion
  = shipped r2 pair** (float sha `96eef43c…`, INT8 sha `a887ea4e…`),
  participant-cluster bootstrap, 2,000 draws, **pre-registered seed
  20261014**, seed constant in one place with the protocol test asserting
  its use.
- Gate 1: delta > 0 AND CI lower bound > −0.5pp.
- Gate 2: winner INT8 held-out stable recall ≥ champion INT8 − 2.0pp.
- Gate 3: winner INT8 within 0.5pp of its own float (directional).
- Secondary: temporal split same recipe; macro-F1 both splits; PEDAP
  zero-shot descriptive.
- Calibration: `sequential_200_d0` + `uniform_random_200_d101`; uniform
  INT8 is the ship candidate. For H_reg/H_corn winners the calibration
  pipeline feeds the same delta-transformed inputs; conversion handles
  the new output shapes.
- `protocol_sweep_r4.json` committed clean before any held-out / temporal
  / PEDAP label read; `stage_evaluate` refuses otherwise. The protocol
  carries the Exp0 result and abort verdict, full selection trail, smoke
  record, champion sha256s, prediction, endpoints, gates, pairing rule,
  dropped-window counts.

## 5. Publication and ship

- `paper/results/calibration-coverage/accuracy_sweep_r4.json`: Exp0
  numbers, aggregates only, per-arm train-window counts, protocol
  pointer, deviations block; participant-leak walker before commit.
  Aborted round publishes Exp0 alone, same file.
- Paper: r4 subsection after r3's (error-structure evidence, Exp0,
  stage tables, confirmatory contrast or abort statement), one table,
  page budget 15.
- Ship decision stops for the user in all cases. H_ce/H_wtemp-lineage
  winner → assets-only ship (r2 pattern). H_reg/H_corn-lineage winner →
  extended ship: both clients' decode (Kotlin + Swift), golden-vector
  schema v2 (output shape + decode rule), parity suites re-pinned, device
  re-verification before merge. No gates passed → MODELS.md one-liner,
  pair retained.

## 6. Testing requirements

- Exp0: LSQ slope hand-computed on small arrays; agreement math on a
  hand-built confusion; abort threshold boundary cases (exactly 2.0pp).
- Heads: CORN encode/decode round-trip incl. all-zeros/all-ones logits;
  regression decode at exact boundary rates (−2, −1, 1, 2) matching
  `label_trend`'s ≤/< rules; sqrt-weight computation vs hand values.
- R stage: LSQ labeler vs endpoint labeler divergence cases; noise
  augmentation determinism under seed and absence at eval; mixup target
  linearity and gating-by-H-winner logic.
- W stage: soup of identical checkpoints equals the checkpoint; SWA
  window off-by-one; checkpoint-retention plumbing.
- Conversion: LiteRT smoke for 1-output and 4-output heads (float +
  INT8), golden-vector generation for whichever contract wins.
- Protocol guard refusal pre-freeze; bootstrap-seed assertion; leak
  walker on published JSON. Suite runs without GlucoBench/DiaData.

## 7. Out of scope

Kalman/causal-filter client front-end (own round — ship surface);
self-supervised pretraining on the 57M-window pool (own round — compute);
personalization; window lengths, capacity, mixtures, logit ensembles,
RT-CGM (settled r1–r3); QAT; any change to the clinical thresholds or to
eval labels; LR-schedule/EMA sweeps; any clinical claim.
