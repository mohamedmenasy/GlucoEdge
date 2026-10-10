# Second-cohort confirmation and delta-encoded inputs — design

**Date:** 2026-10-10
**Status:** approved in chat (both sections), spec for implementation planning
**Prerequisite state:** calibration-coverage study complete (PRs #10–#15 merged);
paper at 14 pp with tab:sensitivity and tab:devicebench; bundled models are the
study's seed-0 uniform-200 pair. Both sub-projects below land in the paper
BEFORE arXiv submission (user decision).

Not a medical device. No clinical claims anywhere in outputs or prose.

## Why

The paper's headline — the historical INT8 collapse was entirely the
toolchain-default sequential calibration, and any randomized 200-window set
fixes it for free — rests on one cohort (weinstock) and one input
representation (absolute mg/dL). This phase tests both edges:

- **A. Second-cohort confirmation:** does the effect reproduce where the
  mechanism predicts it should, and vanish where it predicts it should not?
- **B. Delta-encoded inputs (D12):** does re-encoding the input so its range
  is ~±30 mg/dL make INT8 quantization insensitive to calibration choice
  altogether?

## Sub-project A — two-cohort confirmation (dubosson + hall)

### Prespecified paired prediction

Frozen in each cohort's protocol JSON before any held-out/test label of that
cohort is read:

- **dubosson** (T1D, 9 participants, glucose up to 400, 15.9% of readings
  above 240): uniform−sequential @200 directional recall difference is
  **positive**.
- **hall** (57 participants, max 303, ≈0% above 240): the same difference is
  **near zero** — sequential selection cannot under-cover a range the
  population barely uses.

Both-or-neither outcomes falsify the mechanism story, not merely replication.

### Matrix (per cohort)

- 5 training seeds (0–4), same TrendCNN, same training settings as the main
  study (inverse-frequency weighted CE, Adam 1e-3, batch 64, 20 epochs).
- Calibration arms: sequential d0 + uniform random draws 101–103, budget 200
  only. 20 INT8 + 5 float artifacts per cohort. No balanced/range-coverage
  arms, no 1000 budget, no observer arm — those questions are answered on
  weinstock; this is confirmation.
- If a cohort's eligible calibration pool is smaller than 200 windows, the
  budget drops to the largest feasible round number and the protocol JSON
  records it before freezing.

### Machinery changes

- `experiments/data_availability.py`: generalize `load_weinstock_context()`
  to `load_cohort_context(name)` (weinstock stays the default; existing
  callers unchanged in behavior). Same row-owner rule, same no-gap-fill mask
  (7.5-minute threshold), same disjointness assert.
- `experiments/calibration_coverage.py` stages accept `--dataset` and key the
  run directory as `results/calibration-coverage/<run-id>/` with the dataset
  recorded in the protocol; run ids `conf-dubosson`, `conf-hall`.
- Reading-interval check per cohort before freezing (dubosson and hall are
  5-minute CGM in GlucoBench; verify against the formatter output, and if an
  interval differs the 7.5-minute threshold scales to 1.5× the interval,
  recorded in the protocol).
- Evaluation components: held-out participants where the GlucoBench config
  defines an OOD split; temporal otherwise. Pinned per cohort by a
  labels-untouched split audit (counts only) BEFORE the protocol freezes.
  With 9 participants, dubosson's intervals will be wide; the paper says so
  rather than hiding it.

### Locked-evaluation rule

Per cohort: protocol JSON (prediction, matrix, mask, components, bootstrap
seed) is frozen and committed before any scored label of that cohort's
evaluation components is read. Class support is revealed only at the
evaluate stage, as in the main study. Analysis = same paired
participant-cluster bootstrap (2,000 draws, seed 20261010, same
undefined-endpoint rules).

### Deliverables

- `paper/results/calibration-coverage/confirmation_{dubosson,hall}.json`
  (protocol + aggregate; no participant IDs or timestamps).
- Paper: one Results subsection + one table — two cohorts × {float,
  sequential-200, uniform-200 directional recall, Δ with 95% CI}; Limitations
  updated (single-cohort claim softened to "confirmed on a second T1D cohort
  with a prespecified narrow-range control").

## Sub-project B — delta-encoded inputs (D12)

### The quantization crux

LiteRT input quantization here is per-tensor: mixing a 0–400 anchor with
±30-range deltas in one tensor restores the coarse grid and defeats the
purpose. Hence two variants, and no two-input model (more app surface, no
clear win):

- **(a) deltas-only:** input = 11 consecutive differences (window[i+1] −
  window[i]); absolute level discarded.
- **(b) deltas + scaled anchor:** 12-dim input = 11 deltas plus
  window[-1]/16.0, a fixed constant placing the anchor in the delta range
  (~±25 for 0–400 mg/dL; level resolution ≈ 20 mg/dL per INT8 code — context,
  not precision).

Both train from scratch on the main study's eligible-window weinstock
protocol (same splits, mask, seeds 0–4, training settings). TrendCNN's input
length changes per variant (11 or 12); the label pipeline is untouched.

### Gates, in order

1. **Float gate:** a variant proceeds to INT8 only if its five-seed mean
   held-out directional recall ≥ 0.482 (absolute baseline 0.487 − 0.5 pp).
   Neither passes → D12 reports a negative result in the paper and stops.
   Both pass → deltas-only wins ties (simpler story); otherwise the higher
   one proceeds.
2. **INT8 robustness test** (the actual claim, surviving variant only):
   convert with sequential-200 (d0) and uniform-200 (draws 101–103), 5 seeds.
   Prespecified prediction: uniform−sequential ≈ 0 because 200 early windows
   trivially cover a ±30 range.
3. **Ship gate** (user chose research + ship if it wins): ship only if the
   float gate held AND INT8-under-sequential is within 0.5 pp of its own
   float. Shipping is a separate PR: new bundled models, the delta transform
   in Kotlin (`TrendClassifier`/window path) and Swift, regenerated goldens,
   simulator + on-device parity (iPhone only; Android device retired).

### Honesty constraints

- Weinstock held-out labels were already read by the main study. D12 is
  labeled a post-hoc experiment on previously touched data, same framing as
  the sensitivity tiers. The prespecified predictions above are still frozen
  in a protocol JSON before D12's first held-out metric is computed.
- The delta transform must be byte-reproducible across Python/Kotlin/Swift:
  computed in float32 BEFORE input quantization, with the same subtraction
  order, covered by golden vectors if shipped.

### Deliverables

- `paper/results/calibration-coverage/delta_inputs_r1.json`.
- Paper: one subsection + one table (variants × {float recall, INT8
  sequential, INT8 uniform}); abstract/conclusion touched only if shipped.

## Execution order and testing

A then B; A's two cohorts can run as one plan phase (shared machinery task
first). TDD throughout: the cohort-context generalization, interval scaling,
delta-window construction, and anchor scaling each get RED→GREEN unit tests
on synthetic frames; stages stay resumable by run id; every artifact's
metadata JSON records selection + timings as today. CI never needs
GlucoBench: all new unit tests run on synthetic data only.

## Out of scope

- Any strategy arms beyond sequential/uniform in A or B.
- Device benchmarking of new artifacts (one-phone record exists; D12 ships
  only with parity, not new latency claims).
- arXiv submission mechanics (user's declarations) — happens after this
  phase lands in the paper.
