# GlucoEdge paper

This directory contains a LaTeX preprint draft describing the implemented
GlucoEdge training, conversion, and Android deployment pipeline. The September
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

## Build

```bash
cd paper
tectonic --keep-intermediates --keep-logs main.tex
```

The command produces `main.pdf` and retains `main.bbl`, which should be included
with the LaTeX sources in an arXiv submission.

## Suggested arXiv metadata

- **Title:** GlucoEdge: An Engineering Study of On-Device Five-Class Glucose
  Trend Forecasting and INT8 Quantization
- **Author:** Mohamed Menasy
- **Primary category:** `cs.LG`
- **Possible cross-lists:** `eess.SP`, `q-bio.QM`
- **Comments:** 10 pages, 1 figure, 6 tables. Source code and model artifacts are
  available at <https://github.com/mohamedmenasy/GlucoEdge>.

Before submission, confirm the author name, email, affiliation, category, and
license in the arXiv form, and complete the funding, competing-interest, and
CRediT declarations flagged in the draft. The manuscript deliberately labels
the system as a research engineering demonstration and documents the current
evaluation limitations; it does not make a medical-device or treatment claim.
