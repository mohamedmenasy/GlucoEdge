# Accuracy sweep r5: training recipe, position-preserving head, fresh confirmation cohort

Date: 2026-10-14. Status: approved design (brainstormed 2026-10-13 in a
parallel session as an r4 candidate; renumbered r5 after that round
merged first).
Builds on: `2026-10-12-accuracy-sweep-r3-design.md` (rejected TCN),
`2026-10-13-accuracy-sweep-r4-design.md` (identity outcome: objective
heads, robustness transforms and weight averaging all lost to the
weighted-CE control; no confirmatory labels spent), the r3 final review
(selection-floor/Gate-2 mismatch, strict-argmax tie deviation), and the
external diagnostic (Codex, 2026-10-13,
`analysis/accuracy-review-2026-10-13/`, committed 3755d55) showing that
two synthetic histories with the same current glucose but a rise 30
versus 20 minutes earlier produce bit-identical outputs in the shipped
checkpoint, float and INT8 models: global average pooling erases
within-window timing.

## 1. Question

Does (a) a better training recipe and (b) a position-preserving
classification head beat the shipped champion on directional recall
without breaking the stable floor — confirmed on a cohort whose labels
have never been evaluated?

## 2. What r3 and r4 taught (constraints carried in)

- r4's Experiment 0 measured the label-noise ceiling on weinstock val:
  0.954 directional against the champion's 0.533 — 42pp of headroom.
  The task is not noise-capped; the ceiling is not re-measured in r5.
- r4 exhausted the objective axis (sqrt weights, CORN, regression
  decode), the robustness axis (level-noise, mixup) and weight
  averaging (soup/SWA): every arm lost to the weighted-CE control on
  validation. The head axis — how the [144, 12] feature map reaches the
  classifier — is the one major axis no round has tested; both
  TrendCNN and TrendTCN end in `AdaptiveAvgPool1d`.
- Seed noise on weinstock val directional is wider than most effects;
  every arm runs 5 seeds (`ARM_SEEDS_R5 = (0,1,2,3,4)`).
- The r3 selection floor (champion val stable − 5pp) admitted a seed
  that then failed Gate 2 (− 2pp) — the failure was visible on val
  before the read. r5 uses the SAME margin at selection and gate:
  champion val stable − 2pp.
- r3's Stage S used strict argmax while seeds 1 and 2 sat 0.008pp
  apart; disclosed as a deviation. r5 applies the tie rule at every
  selection including Stage S: winners within 0.1pp (TIE_BAND) resolve
  by cost_key (fewer parameters first), then list order.
- Weinstock held-out and temporal have carried three confirmatory
  reads (r1–r3) plus exploratory threshold analyses; they can no
  longer serve as fresh confirmation. r4 spent no further reads. PEDAP
  has been read repeatedly as a descriptive probe.

## 3. Arms and staged selection

Training data for every arm: the r2/r3 `D5` mixture tensors (five
DiaData cohorts, PEDAP excluded, ~15.14M windows, SAMPLE_SEED
20261012) — already cached; no tensor rebuild. Input contract
unchanged: `[1, 12]` delta/anchor encoding.

### Stage R — training recipe

- `R_base`: existing `D5_none_cnn` checkpoints seeds 0–4 (reused from
  r3's checkpoint directory; zero training). Recipe: 20 epochs, final
  epoch kept, constant LR.
- `R_val`: same TrendCNN(96,144), same data, 5 seeds retrained with:
  - cosine LR decay from the current LR to zero over the same 20 epochs;
  - per-epoch validation; the kept checkpoint is the epoch maximizing
    val directional recall subject to val stable recall ≥ champion val
    stable − 2pp (epochs failing the floor are ineligible; if every
    epoch fails, the seed keeps the final epoch and is flagged).

Selection: five-seed mean val directional recall, disqualifier floor
champion val stable − 2pp, tie band 0.1pp → cost_key → list order.

### Stage H — classification head

- `H_gap`: Stage R winner (reused checkpoints).
- `H_flat`: same conv stack (96,144), AdaptiveAvgPool1d removed,
  flatten the [144, 12] feature map and classify with
  `fc(1728 → 5)`. ~50,645 parameters (cap: 85,450 = 2× champion).
  Trained with the Stage R winning recipe, 5 seeds.

Selection: same rule as Stage R.

### Stage S — seed

Best seed of the Stage H winner by val directional recall, per-seed
floor champion val stable − 2pp, tie band 0.1pp → cost_key → list
order. SystemExit if all seeds fail the floor.

Checkpoint format: the r3 meta-dict (`state_dict`, `arch`,
`in_channels`, `channels`) plus a new `head` field ("gap" | "flat");
`load_checkpoint` grows a head-aware branch with full backward compat
(absent `head` means "gap"). Checkpoints keyed
`{mixture}_{features}_{arch}_{head}_{recipe}_seed{N}.pt`; `R_base`
reuses r3 files via an explicit alias, ledgered.

## 4. Confirmation: RT-CGM zero-shot primary

- Primary cohort: DiaData RT-CGM — 451 cadence-admitted subjects,
  ~12.7M eligible windows, labels NEVER evaluated in any round.
  Disclosure carried in the protocol: r3's rejected `D_rtcgm` arm
  trained on this cohort, but neither the shipped champion nor any r5
  arm trains on it, and no model under comparison has seen its labels
  at training or evaluation time.
- Endpoint: winner uniform INT8 vs shipped champion INT8
  (`trend_float.tflite` sha `96eef43c…`, `trend_int8.tflite` sha
  `a887ea4e…` at freeze, recomputed from the live assets), paired on
  identical eligible windows, participant-cluster bootstrap, 2,000
  draws, pre-registered seed `BOOT_SEED_R5 = 20261015` (single-sourced,
  test-asserted).
- Gates (all on RT-CGM):
  1. directional delta > 0 AND CI lower bound > −0.5pp;
  2. winner INT8 stable recall ≥ champion INT8 stable − 2.0pp;
  3. winner INT8 within 0.5pp of its own float (directional).
- Secondary, disclosed as reused: weinstock held-out and temporal
  (same paired recipe, descriptive CIs), macro-F1 on all splits,
  PEDAP zero-shot descriptive. None gate.
- Calibration: `sequential_200_d0` + `uniform_random_200_d101` as
  before; uniform INT8 is the ship candidate. Calibration windows come
  from weinstock train as in r2/r3 — never from RT-CGM or PEDAP.
- `protocol_sweep_r5.json` committed clean before any RT-CGM label
  read; `stage_evaluate` refuses otherwise. It carries the full
  selection trail, smoke record, champion sha256s, prediction,
  endpoints, gates, pairing rule, the RT-CGM training-exposure
  disclosure, and the selection-rule changes relative to r3.
- Identity guard (r4 lesson): if the surviving candidate's converted
  artifacts are sha256-identical to the shipped champion, the
  confirmatory read is skipped and the round publishes an identity
  outcome — fresh labels are not spent on a delta that is zero by
  construction.

## 5. Ship intent

Gates pass → full r2-style ship: winner float + uniform INT8 become
the bundled pair; goldens regenerated and re-pinned (same `[1,12]`
contract, no client code changes); Android JVM + iOS simulator suites;
device verification on the iPhone 18 Pro Max. Gates fail → MODELS.md
note, bundled pair retained, full record published.

Either way the paper gains: the r5 subsection with one table
(`tab:sweepfive`) and the published `accuracy_sweep_r5.json` aggregate
(leak-walked; deviations block if any accrue). The GAP timing
diagnostic is already on main (3755d55); the r5 subsection cites it as
the motivating evidence for the head arm.

## 6. Non-goals

- No new architectures (TCN closed in r3) and no objective, robustness
  or averaging arms (closed in r4).
- No time-of-day or other input-contract changes (closed in r3).
- No RT-CGM training in any arm.
- No threshold tuning of existing models (the Codex exploratory
  adjustment stays exploratory).
- No re-measurement of the label-noise ceiling (r4's Experiment 0
  stands).

## 7. Infrastructure

`experiments/accuracy_sweep_r5.py`, structured like r3 (stages:
smoke, train, select, calib, convert, evaluate, analyze), reusing r3/r4
helpers where signatures fit (champion eval via live assets, LiteRT
conversion, leak walker). RT-CGM windows come from the existing hours
cache (`results/diadata_cache/hours/RT_CGM.npz`) filtered to the
cadence-admitted list; `stage_evaluate` materializes eval tensors
without ever writing RT-CGM labels outside gitignored `results/`.
Smoke: the r2/r3 bit-reproduction check (batch 512, repro delta 0.0)
before any training. Tests cover: head-aware checkpoint load +
backward compat, flatten-head shape and parameter bound, epoch-select
floor logic, tie-band selection incl. Stage S, BOOT_SEED_R5 single
source, evaluate refusal pre-freeze, RT-CGM never in any arm's
training cohorts, and the identity guard (sha-equal candidate skips
the read).
