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

## Build

```bash
cd paper
tectonic --keep-intermediates --keep-logs main.tex
```

The command produces `main.pdf` and retains `main.bbl`, which should be included
with the LaTeX sources in an arXiv submission.

## Suggested arXiv metadata

- **Title:** GlucoEdge: Calibration Coverage and Quantization Trade-offs in
  On-Device Glucose Trend Classification
- **Author:** Mohamed Menasy
- **Primary category:** `cs.LG`
- **Possible cross-lists:** `eess.SP`, `q-bio.QM`
- **Comments:** 13 pages, 3 figures, 10 tables. Source code and model artifacts are
  available at <https://github.com/mohamedmenasy/GlucoEdge>.

Before submission, confirm the author name, email, affiliation, category, and
license in the arXiv form, and complete the funding, competing-interest, and
CRediT declarations flagged in the draft. The manuscript deliberately labels
the system as a research engineering demonstration and documents the current
evaluation limitations; it does not make a medical-device or treatment claim.
