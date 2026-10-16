# GlucoEdge paper

This directory contains a LaTeX preprint draft describing the implemented
GlucoEdge training, conversion, and Android and iOS deployment pipeline. The September
2026 revision frames the recorded results around calibration-data coverage,
directional recall, and deployment cost. It adds descriptive calculations from
the existing tables; no new training, calibration study, or device benchmark
was run for this revision.

A subsequent prose edit simplifies dense sentences and improves the flow of
the abstract, results, and discussion. It preserves the reported numbers,
equations, tables, citations, limitations, and July 2026 manuscript date.

A provenance-audit revision replaces several "not recorded" caveats with
measured split and calibration facts (a new Results subsection and table). The numbers
come from `python -m experiments.split_audit`, run from the repository root,
whose aggregate output is versioned in `results/split_audit.json`. The audit
reruns only GlucoBench's formatter and the window construction. It reads no
labels, held-out glucose values, or predictions, and trains or evaluates no
model.

A follow-up correction pass dates the manuscript October 2026. It also
describes `rdsvl` as an SME instruction (it had been called SVE) and
updates the rounding text to match the Kotlin client, which now rounds
ties half to even like `numpy.round`.

An October 2026 device revision adds the iOS client: the same artifact
files run through LiteRT's Swift `CompiledModel` API. It reports golden
parity (20/20 vectors per model, INT8 bit-exact) and equal-sample in-app
latency on an iPhone 18 Pro Max (n=100 per model, two rounds), plus a
back-to-back benchmark that separates per-call overhead from the model's
arithmetic. No model was retrained, reconverted, or re-evaluated.

An October 2026 strengthening pass reframes the introduction around the
three questions the recorded evidence answers (conversion parity, the INT8
class-wise trade-off, and the provenance of the input range), with the
calibration-coverage question stated as the motivated follow-up. It adds
two figures built from already-published numbers (per-class recall before
and after INT8; the INT8 input range against development data) and removes
caveats that repeated ones already made in the Limitations section. No
numbers changed.

The October 2026 calibration-coverage revision rewrites the paper around
the executed study (plan
`docs/superpowers/plans/2026-09-19-calibration-coverage-paper-enhancement.md`,
amendments included): corrected split-role/no-gap-fill evaluation, five
training seeds, 100 matrix INT8 artifacts plus a 10-artifact min-max
observer arm, prespecified contrasts with participant-cluster bootstrap
intervals, clip-only/QDQ mechanism controls, and slope/logistic baselines.
Headline: a uniform random 200-window calibration recovers the 13.4-point
directional-recall loss of the toolchain-default sequential selection;
range stratification, budget, and observer changes do nothing. The title
changed to "GlucoEdge: Calibration Coverage and Quantization Trade-offs in
On-Device Glucose Trend Classification". Historical device latency stays
as historical context; the new artifacts are not device-timed yet.
Aggregates live in `results/calibration-coverage/` beside this README.

A sensitivity-tier addendum evaluates the three disjoint window groups the
protocol froze (exact-observed; gap-filled inputs with gap-free endpoints;
gap-filled endpoint) on the same frozen artifacts: both headline findings
reproduce in every tier (new Results subsection and table; the Limitations
bullet now reports this instead of "unevaluated"). Full record in
`results/calibration-coverage/sensitivity_tiers_r1.json`. It also notes the
repository's bundled models were replaced with the study's seed-0
uniform-200 pair.

A device-benchmark addendum executes the plan's prespecified nine-artifact
paired-round protocol on the remaining study phone (iPhone 18 Pro Max,
iOS-only re-scope after the Android device broke): 30 rounds, randomized
order, 6,000 timed calls per artifact, parity + hash verified before
timing. All eight INT8 calibration arms are latency-indistinguishable
(paired differences vs float +0.64 to +0.70 µs), so the calibration fix is
measured, not argued, as deployment-free. New Results table
(tab:devicebench); the device-timing limitation is rewritten accordingly.
Summary JSON: `results/calibration-coverage/device_benchmark_ios_r1.json`.
Pages/figures/tables: 14 pages, 3 figures, 12 tables.

A second-cohort confirmation addendum runs a reduced prespecified matrix
(5 seeds, sequential vs uniform at budget 200, per-cohort protocol frozen
before any evaluation label was read) on two untouched GlucoBench cohorts
with paired predictions: positive on dubosson (wide-range T1D), near zero
on hall (narrow-range). Both held; effect magnitude tracks the
sequential-ceiling deficit (240/401 -> +13.4; 286/400 -> +2.0;
165/303 -> +1.3 primary). New Results subsection + table
(tab:confirmation); Limitations and conclusion updated. Records:
`results/calibration-coverage/confirmation_{dubosson,hall}.json`. Now 13
tables.

A delta-inputs addendum (D12) trains two re-encodings of the same windows
(11 consecutive differences; differences + last value/16) under frozen
gates: both pass the float gate, the anchor variant reaches 0.509 held-out
directional recall (above the absolute baseline 0.487) and its INT8 loses
only ~0.5 pp under the toolchain-default sequential calibration (vs 13.4
for absolute inputs; input grid 6.4x finer). Post-hoc on touched data,
disclosed. New subsection + table (tab:delta); record in
`results/calibration-coverage/delta_inputs_r1.json`. Now 14 tables.

Accuracy sweeps r1-r5 each add a Results subsection and record
(`results/calibration-coverage/accuracy_sweep_r{1..5}.json`). A follow-up
pass brings the front and back matter in line with them: abstract (now
under arXiv's 1,920-character limit, so it can be pasted into the form),
introduction question and contributions, a Limitations bullet on reused
evaluation data, current bundled hashes, the DiaData citation and its
required source-institution disclaimer, and the conclusion.

The panel-review revision (16 October 2026) answers a simulated five-seat
review (`analysis/panel-review-2026-10-16/`: decision letter, roadmap,
reviewer cards, and `response_to_reviewers.md`). The paper now states two
research questions, quantization-calibration coverage and staged model
selection, and the title changed to "GlucoEdge: Quantization-Calibration
Coverage and Staged Model Selection in On-Device Glucose Trend
Classification". The sweeps get their own protocol, results and
discussion sections. Round five is stated at artifact level, and the
second-cohort follow-up is reported as not confirmed (exact sign-flip
tests replace six-participant percentile intervals). The "label-noise
ceiling" is renamed a label-definition stability check. Related work now
covers calibration-data selection and on-device glucose prediction.
Appendices hold the revision history, cohorts, model variants, a held-out
read ledger, and the detail tables moved out of the 18-page body. New
numbers come only from cached predictions and committed records, via
`experiments/panel_revision.py`, whose output is
`results/calibration-coverage/panel_revision_r1.json`. Per-artifact
hashes are in `results/calibration-coverage/artifact_hashes_r1.json`. No
model was retrained or re-evaluated on new data.

## Build

```bash
cd paper
rm -f main.aux main.out main.log main.blg
tectonic --keep-intermediates --keep-logs main.tex
```

Delete the intermediates first: a leftover `main.aux` from an earlier build
can shift page breaks, so only a clean build reproduces the committed PDF.
The command produces `main.pdf` and retains `main.bbl`, which should be included
with the LaTeX sources in an arXiv submission. The `Paper` CI workflow builds
the same way on every change under `paper/` and fails on undefined
references, overfull lines, or non-ASCII source (arXiv compiles with
pdfLaTeX).

## Suggested arXiv metadata

- **Title:** GlucoEdge: Quantization-Calibration Coverage and Staged Model
  Selection in On-Device Glucose Trend Classification
- **Author:** Mohamed Menasy
- **Primary category:** `cs.LG`
- **Possible cross-lists:** `eess.SP`, `q-bio.QM`
- **Comments:** 28 pages (18-page body plus appendices), 3 figures, 23 tables. Source code and model artifacts are
  available at <https://github.com/mohamedmenasy/GlucoEdge>.
- **Abstract** (plain text for the form, 1,915 characters against arXiv's
  1,920 limit; regenerate it if the LaTeX abstract changes):

  ```text
  GlucoEdge, an Android/iOS research prototype, maps 12 five-minute continuous glucose monitor (CGM) readings to one of five classes of the mean rate of change over the next 15 minutes. This paper asks how quantization-calibration coverage affects directional (minority-class) recall and deployment cost, and which model and data changes improve that recall under one-read-per-round staged sweeps. A 2,909-parameter convolutional network trained on Weinstock data was quantized to 8-bit integers (INT8) with four calibration-selection strategies at two budgets. In a prespecified secondary contrast, contiguous first-N selection, spanning two participants and capping the input range at 250.9 mg/dL, cost 13.4 points of held-out directional recall (95% CI 10.3-16.9) against equal-size uniform random selection while accuracy rose. The primary contrast (range stratification) was null: with every randomized selection covering the full input range, stratification, budget and observer choice moved recall by at most about a tenth of a point. A clip-only control attributes the loss to clipping; the fix is free to deploy. Two small cohorts did not confirm the frozen predictions; post hoc, difference-encoded inputs shrank the effect to 0.5 points. Four of five sweep rounds spent one pre-registered read (round four none), three on reused Weinstock test components. A macro-averaged F1 (macro-F1) winner passed every gate but was rejected post hoc as directional recall fell (0.517 to 0.438); training only on DiaData cohorts passed its gates; a seed-4 flatten-head artifact (cosine schedule, validation-chosen epoch) passed all gates on a never-evaluated 2008 trial cohort (+0.20 points [+0.11, +0.29] vs. round three's 1.1-point seed spread) but lowered macro-F1 there (0.410 vs. 0.427), failed two gates on both Weinstock components, and is bundled in the research applications. GlucoEdge is not a medical device.
  ```

Before submission, confirm the author name, email, affiliation, category, and
license in the arXiv form. The funding, competing-interest, and CRediT
declarations follow the Conclusion. The manuscript deliberately labels
the system as a research engineering demonstration and documents the current
evaluation limitations; it does not make a medical-device or treatment claim.
