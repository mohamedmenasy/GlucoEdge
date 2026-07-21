# GlucoEdge paper

This directory contains an arXiv-ready LaTeX manuscript describing the
implemented GlucoEdge training, conversion, and Android deployment pipeline.

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
- **Comments:** 8 pages, 1 figure, 5 tables. Source code and model artifacts are
  available at <https://github.com/mohamedmenasy/GlucoEdge>.

Before submission, confirm the author name, email, affiliation, category, and
license in the arXiv form. The manuscript deliberately labels the system as a
research engineering demonstration and documents the current evaluation
limitations; it does not make a medical-device or treatment claim.
