# Field Analysis Report (Phase 0)

## Paper Basic Information
- **Title**: GlucoEdge: Calibration Coverage and Quantization Trade-offs in On-Device Glucose Trend Classification
- **Abstract length**: ~330 words
- **Full text length**: ~11,400 words (two-column, incl. tables)
- **Number of references**: ~35 (natbib, unsrtnat)

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | Applied machine-learning systems — post-training quantization (PTQ) and on-device inference |
| Secondary Disciplines | Health informatics / diabetes technology (CGM forecasting); ML evaluation methodology (leakage, pre-registration, clustered resampling); mobile systems engineering |
| Research Paradigm | Quantitative |
| Methodology Type | Controlled computational experiment (factorial calibration matrix: 4 strategies x 2 budgets x 3 draws x 5 seeds) + staged ML model-selection sweeps with pre-registered confirmatory reads |
| Target Journal Tier | `criteria_binding_unavailable` — no author-confirmed review target supplied. Field-general observation only: empirical case-study / negative-result paper of workshop-to-journal maturity; author's stated channel is an arXiv cs.LG preprint. No venue-fit claim is made. |
| Paper Maturity | Pre-submission (complete structure, extensive results, bibliography); but carries accreted revision-history narrative and appended sweep rounds |

## Candidate Venues (for author consideration only — not a venue-fit claim)
1. ML4H / CHIL-type health-ML proceedings — empirical evaluation-protocol and deployment papers
2. Workshop on efficient / on-device ML (e.g. TinyML or MLSys-style workshop) — PTQ calibration case study
3. Health-informatics journal (JBHI / JMIR mHealth class) — if reframed around the CGM application

## Reviewer Configuration Cards

### Reviewer Configuration Card #1
**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Identity Description**: Action editor for an applied health-ML proceedings venue (ML4H/CHIL class) who handles empirical evaluation and deployment papers and enforces reproducibility checklists.
**Review Focus**:
  1. Whether the manuscript states ONE contribution: the calibration-coverage study vs. the five appended accuracy sweeps — title/abstract/discussion fidelity to the actual content.
  2. Originality relative to known PTQ guidance that calibration data must be representative; what is new beyond a well-documented case study.
  3. Organization and readability: revision-history narrative, section ordering, abstract density, figure/table quality (rendered PDF).
**Will particularly care about**: Whether a reader can tell in one pass what was confirmed, what was exploratory, and what the paper's single takeaway is.
**Possible blind spots**: Statistical validity of the bootstrap and multiplicity (R1); CGM-domain correctness (R2/R3).

### Reviewer Configuration Card #2
**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1 (Methodology)
**Identity Description**: Biostatistician specializing in resampling inference for clustered longitudinal sensor data and in pre-registered / confirmatory ML evaluation (leakage, forking paths, selection-biased winners).
**Review Focus**:
  1. Participant-cluster bootstrap design: conditioning on fitted models and calibration sets, few-cluster cohorts (6 clusters), and what the CIs license (artifact vs. recipe/strategy).
  2. Confirmatory discipline across the calibration study and five sweep rounds: primary vs. secondary endpoints, reuse of held-out data, multiplicity, post-hoc overrides, best-seed selection vs. seed spread.
  3. Internal numeric consistency across abstract, text, tables, figures and the cited supplementary JSONs in paper/results/ (incl. whether protocols were committed before reads, via git history).
**Will particularly care about**: Whether each headline number is the estimand the text says it is, with uncertainty that matches the claim.
**Possible blind spots**: Domain plausibility of labels and thresholds; PTQ literature.

### Reviewer Configuration Card #3
**Role**: Peer Reviewer 2
**Display role**: Peer Reviewer 2 (Domain)
**Identity Description**: Researcher in efficient on-device ML for wearable health sensing (post-training quantization, TinyML deployment), who also publishes on CGM glucose-forecasting benchmarks (GlucoBench, OhioT1DM).
**Review Focus**:
  1. Whether PTQ claims (observers, clipping vs. rounding, calibration-set size/selection, "toolchain-default" framing) correctly represent PTQ literature and toolchain behavior (PT2E/LiteRT).
  2. Related-work coverage and characterization in both PTQ calibration-data research and CGM forecasting/on-device glucose prediction; missing key references.
  3. Technical accuracy of quantization arithmetic and runtime details; dataset descriptions (Weinstock, DiaData cohorts, RT-CGM, PEDAP).
**Will particularly care about**: Whether the "sharpest observation" is already known in the PTQ literature and whether prior work is fairly represented.
**Possible blind spots**: Clinical meaning of the target; bootstrap details.

### Reviewer Configuration Card #4
**Role**: Peer Reviewer 3
**Display role**: Peer Reviewer 3 (Cross-disciplinary / Practical)
**Identity Description**: Diabetes-technology clinician-scientist (endocrinology + digital health) with expertise in CGM accuracy standards, sensor noise, and trend-arrow interpretation.
**Review Focus**:
  1. Whether the five-class 15-minute rate-of-change target and "trend arrow" framing are represented accurately for clinical readers; label sensitivity to realistic CGM noise.
  2. Practical meaning of ~0.5 directional recall and of class-specific errors (falling_fast) for people with diabetes; asymmetric error costs; cohort representativeness and sensor-range caps.
  3. Safety and scope language ("not a medical device", "clinical-shape classifier") and potential for misinterpretation by app users or developers.
**Will particularly care about**: Whether clinically loaded wording outruns what was measured.
**Possible blind spots**: PTQ internals; statistical machinery.

## Fixed fifth seat
Devil's Advocate (fixed, not dynamically configured).

## Review Strategy Recommendations
- Paper mixes a prespecified calibration study with five sequential model-selection rounds; reviewers should keep confirmatory and exploratory evidence separate.
- Supplementary result JSONs are cited by filename; cross-checking them is in scope (read-only).
- Expected tension: R2 (novelty vs. PTQ literature) vs. EIC (value as a rigorous case study).
