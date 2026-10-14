# Accuracy sweep r4: training recipe, position-preserving head, fresh confirmation cohort

Date: 2026-10-13. Status: approved design.
Builds on: `2026-10-12-accuracy-sweep-r3-design.md` (rejected TCN candidate),
the r3 final review (selection-floor/Gate-2 mismatch, strict-argmax tie
deviation), and an external diagnostic (Codex, 2026-10-13,
`analysis/accuracy-review-2026-10-13/`) showing that two synthetic
histories with the same current glucose but a rise 30 versus 20 minutes
earlier produce bit-identical outputs in the shipped checkpoint, float
and INT8 models: global average pooling erases within-window timing.

## 1. Question

Does (a) a better training recipe and (b) a position-preserving
classification head beat the shipped champion on directional recall
without breaking the stable floor — confirmed on a cohort whose labels
have never been evaluated?

## 2. What r3 taught (constraints carried in)

- Seed noise on weinstock val directional is wider than most effects;
  every arm runs 5 seeds (`ARM_SEEDS_R4 = (0,1,2,3,4)`).
- The r3 selection floor (champion val stable − 5pp) admitted a seed
  that then failed Gate 2 (− 2pp) — the failure was visible on val
  before the read. r4 uses the SAME margin at selection and gate:
  champion val stable − 2pp.
- r3's Stage S used strict argmax while seeds 1 and 2 sat 0.008pp
  apart; disclosed as a deviation. r4 applies the r1 tie rule at every
  selection including Stage S: winners within 0.1pp (TIE_BAND) resolve
  by cost_key (fewer parameters first), then list order.
- Weinstock held-out and temporal have carried three confirmatory
  reads (r1, r2, r3) plus exploratory threshold analyses; they can no
  longer serve as fresh confirmation. PEDAP has been read repeatedly
  as a descriptive probe.

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
  trained on this cohort, but neither the shipped champion nor any r4
  arm trains on it, and no model under comparison has seen its labels
  at training or evaluation time.
- Endpoint: winner uniform INT8 vs shipped champion INT8
  (`trend_float.tflite` sha `96eef43c…`, `trend_int8.tflite` sha
  `a887ea4e…` at freeze, recomputed from the live assets), paired on
  identical eligible windows, participant-cluster bootstrap, 2,000
  draws, pre-registered seed `BOOT_SEED_R4 = 20261014` (single-sourced,
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
- `protocol_sweep_r4.json` committed clean before any RT-CGM label
  read; `stage_evaluate` refuses otherwise. It carries the full
  selection trail, smoke record, champion sha256s, prediction,
  endpoints, gates, pairing rule, the RT-CGM training-exposure
  disclosure, and the selection-rule changes relative to r3.

## 5. Ship intent

Gates pass → full r2-style ship: winner float + uniform INT8 become
the bundled pair; goldens regenerated and re-pinned (same `[1,12]`
contract, no client code changes); Android JVM + iOS simulator suites;
device verification on the iPhone 18 Pro Max. Gates fail → MODELS.md
note, bundled pair retained, full record published.

Either way the paper gains: the GAP timing diagnostic (identical
outputs for time-shifted synthetic histories), the r4 subsection with
one table (`tab:sweepfour`), and the published
`accuracy_sweep_r4.json` aggregate (leak-walked; deviations block if
any accrue).

## 6. Non-goals

- No new architectures (the TCN question is closed; r3).
- No time-of-day or other input-contract changes (closed; r3).
- No RT-CGM training in any arm.
- No threshold tuning of existing models (Codex's exploratory
  adjustment stays exploratory).
- No CORN/ordinal head this round — one new modeling idea per round.

## 7. Infrastructure

`experiments/accuracy_sweep_r4.py`, structured like r3 (stages:
smoke, train, select, calib, convert, evaluate, analyze), reusing r3
helpers where signatures fit (champion eval via live assets, LiteRT
conversion, leak walker). RT-CGM windows come from the existing hours
cache (`results/diadata_cache/hours/RT_CGM.npz`) filtered to the
cadence-admitted list; `stage_evaluate` materializes eval tensors
without ever writing RT-CGM labels outside gitignored `results/`.
Smoke: the r2/r3 bit-reproduction check (batch 512, repro delta 0.0)
before any training. Tests cover: head-aware checkpoint load +
backward compat, flatten-head shape and parameter bound, epoch-select
floor logic, tie-band selection incl. Stage S, BOOT_SEED_R4 single
source, evaluate refusal pre-freeze, RT-CGM never in any arm's
training cohorts.
