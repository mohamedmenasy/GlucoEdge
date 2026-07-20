# GlucoEdge Paper Context and Literature Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the GlucoEdge general preprint with a comparative literature review, additional context, verified references, and appropriately bounded claims without changing its experiments or numerical results.

**Architecture:** Keep the existing single-file LaTeX manuscript and BibTeX database. Expand the narrative in place, add one full-width related-work table, then compile and visually inspect the PDF. No new package, script, experiment, or manuscript abstraction is needed.

**Tech Stack:** LaTeX (`article`, `natbib`, `booktabs`), BibTeX, Tectonic, Poppler (`pdfinfo`, `pdftoppm`), ripgrep.

## Global Constraints

- Preserve every reported measurement unless correcting an independently verified transcription error.
- Add context and citations only; do not run or imply new experiments.
- Keep the research-only, non-medical-device, and non-treatment framing explicit.
- Do not imply clinical validity, dosing utility, or prospective/live-sensor deployment.
- Disclose the test-context issue before interpreting aggregate test metrics.
- Treat INT8 input saturation as a plausible contributor, not a proven sole cause.
- Restrict latency conclusions to the single observed device run with unequal sample counts.
- Use verified primary scholarly sources or authoritative official documentation.
- Target approximately eight pages; prefer synthesis over citation padding.

---

### Task 1: Repair and Extend the Bibliography

**Files:**
- Modify: `references.bib:1-141`
- Modify: `main.tex:61,100`

**Interfaces:**
- Consumes: DOI, publisher, proceedings, and official documentation records.
- Produces: stable citation keys used by Tasks 2 and 3.

- [ ] **Step 1: Correct existing records**

Make these exact repairs:

- Rename `wang2023systematic` to `liu2023systematic` and update both uses in
  `main.tex` in the same change so the intermediate build has no undefined
  citation.
- Add `Marling, Colleen` as the fourth author of `mirshekarian2019lstm`.
- Include `(rtCGM)` in the Pettus title.
- Update all Google records to their canonical `developers.google.com` URLs,
  year 2026, and access date 2026-07-20.
- Add `pytorchPT2E2026`, titled `PT2E Quantization`, organization `PyTorch`,
  year 2026, URL
  `https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html`, accessed
  2026-07-20.

- [ ] **Step 2: Add CGM forecasting and evaluation records**

Add these keys and verified identifiers:

| Key | Publication and identifier |
|---|---|
| `reifman2007predictive` | Reifman et al., *Predictive Monitoring for Improved Management of Glucose Levels*, JDST 1(4):478--486, DOI `10.1177/193229680700100405` |
| `facchinetti2011index` | Facchinetti et al., *A New Index to Optimally Design and Compare Continuous Glucose Monitoring Glucose Prediction Algorithms*, DTT 13(2):111--119, DOI `10.1089/dia.2010.0151` |
| `martinsson2020variance` | Martinsson et al., *Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks*, JHIR 4(1):1--18, DOI `10.1007/s41666-019-00059-y` |
| `li2020crnn` | Li et al., *Convolutional Recurrent Neural Networks for Glucose Prediction*, IEEE JBHI 24(2):603--613, DOI `10.1109/JBHI.2019.2908488` |
| `ghimire2024generalize` | Ghimire et al., *Deep Learning for Blood Glucose Level Prediction: How Well Do Models Generalize Across Different Data Sets?*, PLOS ONE 19(9):e0310801, DOI `10.1371/journal.pone.0310801` |
| `bergmeir2018crossvalidation` | Bergmeir et al., *A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction*, CSDA 120:70--83, DOI `10.1016/j.csda.2017.11.003` |
| `kapoor2023leakage` | Kapoor and Narayanan, *Leakage and the Reproducibility Crisis in Machine-Learning-Based Science*, Patterns 4(9):100804, DOI `10.1016/j.patter.2023.100804` |
| `bouthillier2021variance` | Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 3:747--769, official proceedings URL |

- [ ] **Step 3: Add edge inference and quantization records**

Add these keys and verified identifiers:

| Key | Publication and identifier |
|---|---|
| `lane2016deepx` | Lane et al., *DeepX*, IPSN 2016:1--12, DOI `10.1109/IPSN.2016.7460664` |
| `ignatov2019aibenchmark` | Ignatov et al., *AI Benchmark: All About Deep Learning on Smartphones in 2019*, ICCVW 2019:3617--3635, DOI `10.1109/ICCVW.2019.00447` |
| `banbury2021mlperftiny` | Banbury et al., *MLPerf Tiny Benchmark*, NeurIPS Datasets and Benchmarks 1, official proceedings URL |
| `banner2019ptq` | Banner et al., *Post Training 4-Bit Quantization of Convolutional Networks for Rapid-Deployment*, NeurIPS 32, official proceedings URL |
| `nagel2020adaround` | Nagel et al., *Up or Down? Adaptive Rounding for Post-Training Quantization*, ICML, PMLR 119:7197--7206 |
| `hubara2021smallcalibration` | Hubara et al., *Accurate Post Training Quantization With Small Calibration Sets*, ICML, PMLR 139:4466--4475 |

- [ ] **Step 4: Check BibTeX syntax**

Run:

```bash
tectonic --keep-intermediates --keep-logs main.tex
git diff --check -- references.bib
rg -n '^@' references.bib
```

Expected: Tectonic exits 0, the diff check is silent, and the bibliography has
30 entries after adding 14 scholarly sources plus the PT2E documentation entry.

- [ ] **Step 5: Commit**

```bash
git add paper/references.bib paper/main.tex
git commit -m "docs: expand paper bibliography"
```

---

### Task 2: Rewrite the Introduction and Related Work

**Files:**
- Modify: `main.tex:24-124`

**Interfaces:**
- Consumes: citation keys from Task 1.
- Produces: expanded context, literature synthesis, bounded contribution wording, and `tab:relatedwork`.

- [ ] **Step 1: Tighten the abstract and contribution wording**

Replace “commonly studied as a server-side regression problem” with the
narrower observation that published CGM forecasting evaluations largely
emphasize offline predictive accuracy. Describe the result as a case study,
not a universal quantization finding. Preserve every number and the final
research-only warning.

Use “future-derived, five-class 15-minute trend labels” in place of “genuinely
future” and describe the implementation as a reproducible offline replay path.

- [ ] **Step 2: Expand the Introduction**

Add concise paragraphs covering:

1. five-minute CGM cadence and short-horizon computational forecasting;
2. value regression, event prediction, and trend classification under
   heterogeneous horizons, inputs, cohorts, splits, and metrics
   (`liu2023systematic`, `facchinetti2011index`,
   `sergazinov2024glucobench`, `ghimire2024generalize`); and
3. retrospective/current commercial arrows versus this paper's future-derived
   15-minute target (`pettus2017recommendations`, `rodacki2021trend`).

- [ ] **Step 3: Replace Related Work with four subsections**

Use these headings:

```latex
\subsection{Forecasting tasks, horizons, and inputs}
\subsection{Benchmarks and evaluation protocols}
\subsection{Trend representations and class imbalance}
\subsection{Edge inference and quantization}
```

The prose must synthesize, rather than list:

- autoregressive, recurrent/attention, convolutional-recurrent, and
  cross-dataset studies (`reifman2007predictive`, `mirshekarian2019lstm`,
  `martinsson2020variance`, `li2020crnn`, `ghimire2024generalize`);
- subject-dependent, held-out-participant, and cross-dataset evaluation
  (`sergazinov2024glucobench`, `bergmeir2018crossvalidation`,
  `kapoor2023leakage`, `bouthillier2021variance`);
- retrospective arrows versus future labels and why macro recall is needed
  (`pettus2017recommendations`, `rodacki2021trend`,
  `brodersen2010balanced`); and
- mobile inference, runtime/hardware dependence, and PTQ calibration
  (`lane2016deepx`, `li2020crnn`, `ignatov2019aibenchmark`,
  `banbury2021mlperftiny`, `jacob2018quantization`, `banner2019ptq`,
  `nagel2020adaround`, `hubara2021smallcalibration`).

End with GlucoEdge's intersection of public preprocessing, conversion parity,
class-wise evaluation, and one-device replay measurement. Make no “first”
claim and do not compare metric values across incompatible protocols.

- [ ] **Step 4: Add a full-width study comparison table**

Add `table*` with label `tab:relatedwork`, columns `Study`, `Target`, `Horizon`,
`Inputs`, `Evaluation scope`, and `Device evidence`, and these rows:

```latex
Reifman et al. \cite{reifman2007predictive} & Value & 30 min & CGM & 9 participants & None reported \\
Mirshekarian et al. \cite{mirshekarian2019lstm} & Value & 30/60 min & CGM + context & OhioT1DM/synthetic & None reported \\
Martinsson et al. \cite{martinsson2020variance} & Value + variance & 30/60 min & CGM & OhioT1DM & None reported \\
Li et al. \cite{li2020crnn} & Value & 30/60 min & CGM, insulin, meals & 10 real + 10 simulated & Android timing \\
GlucoBench \cite{sergazinov2024glucobench} & Value & Multiple & Dataset-dependent & Five public datasets & None reported \\
Ghimire et al. \cite{ghimire2024generalize} & Value & 30/60 min & CGM & Four datasets & None reported \\
\project (this work) & Trend class & 15 min & CGM & Weinstock aggregate test & Android replay timing \\
```

Caption: “Representative CGM forecasting studies and evaluation scope.
Metrics are not compared across rows because datasets and protocols differ.”

- [ ] **Step 5: Compile and inspect**

Run:

```bash
tectonic --keep-intermediates --keep-logs main.tex
rg -n 'undefined|Citation.*undefined|Reference.*undefined|Overfull' main.log
rg -n 'server-side|state-of-the-art|first to' main.tex
```

Expected: build succeeds; warning scan is silent; no priority or unsupported
server-side claim remains; the table fits the page.

- [ ] **Step 6: Commit**

```bash
git add paper/main.tex
git commit -m "docs: deepen paper context and related work"
```

---

### Task 3: Add Methodological Context and Bound Interpretation

**Files:**
- Modify: `main.tex:126-487`

**Interfaces:**
- Consumes: evaluation, leakage, quantization, and benchmarking sources from Task 1.
- Produces: context-aware Methods, Protocol, Discussion, Limitations, and Conclusion sections.

- [ ] **Step 1: Strengthen formulation and dataset context**

State that the 60-minute history and 15-minute horizon are fixed engineering
choices, not optimized clinical values. Describe the labels as a simplified
future-rate abstraction.

Add that the source cohort consists of older adults with long-standing type 1
diabetes and blinded CGM (`weinstock2016risk`). Explain that interpolation may
place generated values in inputs or targets and that stride-one windows overlap
heavily, so window counts are not independent event counts.

Identify the preliminary five-versus-three test recall as an exploratory,
test-informed development decision rather than unbiased validation evidence.

- [ ] **Step 2: Put the evaluation caveat before Results**

Add `\label{sec:limitations}` to Limitations. At the start of Experimental
Protocol insert:

```latex
The aggregate test frame combines temporal and held-out-participant components.
As detailed in Section~\ref{sec:limitations}, context-prefix rows were not
masked as target starts in this implementation; the resulting metrics document
the shipped artifact but are not a leakage-free estimate of participant-level
generalization.
```

- [ ] **Step 3: Revise causal and deployment claims**

Change “The calibration sample explains part of the predictive degradation”
to “The calibration sample provides one plausible mechanism for part of the
predictive degradation.” Preserve the 240/401 mg/dL ranges and 28% saturation.

In Discussion, say fixed costs “may dominate,” delete “immaterial saving for
most phones,” connect PTQ findings to `banner2019ptq`, `nagel2020adaround`, and
`hubara2021smallcalibration`, and connect device dependence to
`lane2016deepx`, `ignatov2019aibenchmark`, and `banbury2021mlperftiny`.
State that parity excludes divergent client arithmetic but does not isolate
calibration from every other quantization effect.

- [ ] **Step 4: Expand Limitations and tighten Conclusion**

Integrate these points without duplicating existing items:

- dependent overlapping windows (`bergmeir2018crossvalidation`);
- source-cohort limits and absent cross-dataset validation
  (`ghimire2024generalize`);
- possible interpolated inputs/targets;
- omitted meal, insulin, activity, illness, and other covariates;
- single-seed variance (`bouthillier2021variance`);
- absent probability calibration, event-level evaluation, simple predictive
  baselines, and confidence intervals; and
- context-prefix leakage risk (`kapoor2023leakage`).

Keep the conclusion as an engineering case study and retain explicit
prerequisites for scientific or clinical claims.

- [ ] **Step 5: Compile and run guard checks**

Run:

```bash
tectonic --keep-intermediates --keep-logs main.tex
git diff --check -- main.tex
rg -n 'explains part|immaterial saving|genuinely future|complete path' main.tex
rg -n 'not a medical device|must not be used for medical decisions|context-prefix|single seed|overlapping' main.tex
```

Expected: build and diff check succeed; the first scan is empty; the second
finds safety framing and strengthened limitations.

- [ ] **Step 6: Commit**

```bash
git add paper/main.tex
git commit -m "docs: contextualize paper methods and limitations"
```

---

### Task 4: Verify the Artifact and Submission Metadata

**Files:**
- Modify: `README.md:14-26`
- Regenerate: `main.pdf`
- Regenerate: `main.bbl`
- Inspect: `main.log`
- Create temporarily: `tmp/pdfs/glucoedge-revised-*.png`

**Interfaces:**
- Consumes: completed manuscript and bibliography.
- Produces: clean arXiv-ready artifacts and accurate repository metadata.

- [ ] **Step 1: Build twice**

```bash
tectonic --keep-intermediates --keep-logs main.tex
tectonic --keep-intermediates --keep-logs main.tex
```

Expected: both runs exit 0 and regenerate `main.pdf` and `main.bbl`.

- [ ] **Step 2: Audit references and warnings**

```bash
rg -n 'undefined|multiply defined|Overfull|LaTeX Warning' main.log
pdfinfo main.pdf
```

Expected: no undefined or multiply-defined references and no overfull boxes.
Review underfull boxes individually; accept only harmless paragraph spacing.

- [ ] **Step 3: Render and inspect every page**

```bash
mkdir -p tmp/pdfs
pdftoppm -png -r 150 main.pdf tmp/pdfs/glucoedge-revised
```

Check every PNG for clipped/overlapping text, broken tables, unreadable
references, black-square glyphs, poor page breaks, and an illegible
`tab:relatedwork`.

- [ ] **Step 4: Update README**

Use `pdfinfo main.pdf` and the source to replace “6 pages, 1 figure, 4 tables”
with the exact final counts. Preserve all other submission advice.

- [ ] **Step 5: Run final checks**

```bash
tectonic --keep-intermediates --keep-logs main.tex
git diff --check -- main.tex references.bib README.md
rg -n 'undefined|multiply defined|Overfull' main.log
git status --short
```

Expected: build exits 0; checks are silent; status contains only intended paper
changes plus pre-existing user files.

- [ ] **Step 6: Commit**

```bash
git add paper/main.tex paper/references.bib paper/main.bbl paper/main.pdf paper/README.md
git commit -m "docs: publish enhanced GlucoEdge preprint"
```
