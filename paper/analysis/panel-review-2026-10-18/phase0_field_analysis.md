# Field Analysis Report

Phase 0, `full` review panel, academic-paper-reviewer v1.11.1 (`field_analyst_agent`). Cold read: only `paper/main.tex` (all 2,706 lines) and the rendered `paper/main.pdf` were consulted. Nothing else under `paper/analysis/` was opened.

## Paper Basic Information
- **Title**: GlucoEdge: Quantization-Calibration Coverage and Staged Model Selection in On-Device Glucose Trend Classification
- **Authorship**: single author (Independent Researcher). Dated October 2026. English.
- **Format**: two-column, 10 pt article, 29 PDF pages. Main text runs pp. 1-19 (12 numbered sections: Introduction through Conclusion, plus Declarations). References take pp. 19-22, and Appendices A-D take pp. 23-29.
- **Abstract length**: about 220 words in one paragraph, with more than a dozen reported quantities (effect sizes, CIs, recalls, macro-F1 values, seed and round identifiers).
- **Full text length**: about 21,900 words in the rendered PDF. That splits into about 15,000 words of main text (including tables and figures), about 1,900 of references and about 4,400 of appendices.
- **Number of references**: 68 bibliography entries (110 `\cite` commands, natbib numeric style, `unsrtnat`).
- **Tables / figures**: 23 tables (9 in the main text, 14 in the appendices) and 3 figures (2 in the main text, 1 in the appendices).

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | **Machine learning for health: deployment-oriented classification of physiological (CGM) time series.** The paper evaluates an on-device classifier of future glucose rate-of-change bands. It has two research questions: (RQ1) how the coverage of the post-training-quantization calibration set affects minority-class recall and deployment cost; (RQ2) which model and data changes improve directional recall under staged, pre-registered selection. |
| Secondary Disciplines | (1) Efficient / on-device ML: static INT8 post-training quantization (PT2E to LiteRT), mobile inference runtimes, latency benchmarking. (2) Diabetes technology / CGM data science: trend-arrow rate bands, sensor LOW/HIGH codes, blinded vs real-time wear, cohort heterogeneity across Weinstock, DiaData, RT-CGM and PEDAP. (3) Statistical methodology of ML evaluation: pre-registration, paired participant-cluster bootstrap, exact sign-flip tests, reuse of held-out data, model-selection gates. |
| Research Paradigm | **Quantitative**: empirical, computational experiments on retrospective public data. Within that, the paper mixes a pre-registered confirmatory design (the RQ1 calibration study; each sweep round's single read) with acknowledged exploratory and post hoc analyses (delta encoding, the second cohorts, the round-one override, descriptive re-reads). |
| Methodology Type | **Statistical Modeling / Machine Learning**, run as **controlled computational experiments**. RQ1 is a factorial calibration study: 4 strategies x 2 nested budgets x 3 draws x 5 training seeds, giving 100 INT8 artifacts plus a 10-artifact observer arm, with clip-only and QDQ mechanism controls. RQ2 is a **sequential (greedy staged) model-selection program** over five rounds with frozen gates. It is also a **comparative study** across cohorts: two GlucoBench second cohorts, plus DiaData cohorts as zero-shot probes. All of it is a secondary analysis of public de-identified data. |
| Target Journal Tier | **`criteria_binding_unavailable`.** No author-confirmed review target was supplied, so no venue, track, article type or venue criteria are bound. I make no venue-fit claim. *Field-general maturity/tier observation only:* the study is methodologically careful and narrowly scoped. Its main positive result (a 13.4-point directional-recall loss under contiguous first-N calibration) quantifies a departure from guidance the paper itself calls established, and it does so in one primary cohort, one architecture and one toolchain configuration. The primary contrast was null and is self-labelled "untestable as posed". RQ2 is a case study of one selection program, not a generalizable method. Rigor and disclosure are well above what applied health-ML preprints typically show, but the contribution is incremental, and the main text (about 19 two-column pages) is long for the claim set. |
| Paper Maturity | **Pre-submission (late-stage, heavily revised).** The structure is complete (IMRaD plus Limitations, Reproducibility/Ethics, Declarations, four appendices). Cross-references and numbering are consistent, and citation formatting is uniform. The language is polished but highly compressed and densely hedged. Appendix A and the read ledger (Table 13) document a July baseline, September and 10 October revisions, and later revision passes through 16 October. The outstanding problems are proportion, focus and density, not completeness. |

## Candidate Venues (for author consideration only)

`criteria_binding_unavailable`: I did not resolve or verify any venue's criteria, and none of the entries below is a venue-fit claim. They are listed only because their scope overlaps the paper's disciplines. The panel should not review against any of them.

1. *A general ML-methodology journal that publishes empirical evaluation and reproducibility studies* (e.g., Transactions on Machine Learning Research). Rationale from the paper's side: its strongest material is evaluation-protocol rigor and its lessons about selection and endpoints.
2. *An applied health-ML proceedings venue or biomedical-informatics journal* (e.g., ML4H / CHIL proceedings, IEEE JBHI). Rationale from the paper's side: CGM classification, held-out-participant and cross-cohort evaluation.
3. *An efficient/on-device ML workshop or symposium* (e.g., tinyML Research Symposium), for a condensed RQ1-only version. Alternatively, a diabetes-technology journal for a reframed, clinically scoped version. Rationale from the paper's side: the PTQ calibration pitfall in time-series deployment.

## Reviewer Configuration Cards

> **Cross-disciplinary note (Edge Case 1).** This paper spans three or more disciplines: health time-series ML, on-device quantization, CGM/diabetes technology, and the statistics of ML evaluation. The coverage strategy is:
> - **R2** takes the core discipline (health time-series ML: endpoint, baselines, external validity, literature).
> - **R3** takes the efficient/on-device secondary discipline from a practical, toolchain and runtime angle.
> - **R1** takes the evaluation-statistics secondary discipline.
> - The **Journal-Fit Reviewer** takes contribution, proportion and disclosure.
>
> The CGM-clinical and regulatory angle has **no dedicated seat** this round. R2 covers its data side (sensor-derived labels, cohort and sensor heterogeneity). The Journal-Fit Reviewer covers its positioning side ("not a medical device" while distributing bundled models). The synthesizer should treat clinical interpretation as a known partial blind spot.
>
> **Panel-diversity rationale.** Earlier simulated rounds used an applied health-ML proceedings action editor, a clustered-resampling / pre-registration biostatistician, a PTQ/TinyML + CGM-forecasting researcher, and a diabetes-technology clinician-scientist. Within the configuration principles, this round deliberately moves each seat to a different lens on the same needs:
> - a generalist ML-methodology/reproducibility editor;
> - a statistician of experimental variance and selective inference rather than cluster bootstrap;
> - an endpoint/metric and external-validation specialist rather than a PTQ/benchmark researcher;
> - a compiler/runtime systems practitioner rather than a clinician.
>
> This choice rests on the identity list in the dispatch only; no earlier review content was read.

### Reviewer Configuration Card #1

**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Identity Description**: Senior handling editor working field-general (no target venue configured). Has long cross-venue experience adjudicating empirical machine-learning methodology papers: reproducibility studies, negative-result and "lessons-learned" reports, and evaluation-protocol papers. Also has cross-listed handling experience with digital-health ML submissions. Reads as a generalist ML-methodology editor deciding whether a careful but narrow empirical study gives a broad ML-evaluation readership something it would change its practice for.
**Review Focus**:
  1. **Contribution and coherence.** Is this one paper or two? RQ1 (calibration coverage) and RQ2 (five staged sweeps) share a model family and a split but not a claim. Is RQ1's headline new knowledge, or a well-instrumented quantification of a pipeline bug against guidance the paper itself calls established (Sections 1, 2.4, 9.1)? It is a prespecified *secondary* contrast added by amendment, while the primary contrast is null and "untestable as posed". Does the take-home rule ("cover the deployment input range; do not gate on accuracy") go beyond existing practice?
  2. **Proportion and framing.** The main text is about 15,000 words over about 19 pages, with 23 tables. The abstract is numerically dense and much of it concerns RQ2's round-by-round history. Does the title ("On-Device", "INT8") match an artifact in which only the final fully connected layer runs in int8 (Section 7.9)? Should the round-by-round sweep narrative, the read ledger and the bundling history move to a supplement or a separate report?
  3. **Disclosure, positioning and reporting completeness.**
     - Appendix D and the read ledger (Table 13, `3755d55`) credit an "external review", versioned at `paper/analysis/accuracy-review-2026-10-13/`, with proposing round five's flatten head and schedule. The manuscript does not say who or what performed that review. The dispatch describes `paper/analysis/` as the home of earlier *simulated* reviews. If that review was AI-generated, calling it "external review" without an AI-use statement is a disclosure issue (the Declarations contain no AI-use statement).
     - The "not a medical device" positioning sits alongside distributed Android/iOS apps with a bundled model and an unevaluated on-device LLM "Explain" caption.
     - Check reporting completeness against health-prediction-model reporting norms (intended use, populations, failure modes; a model card is referenced).
**Will particularly care about**: whether the paper states one clear, transferable finding at the strength the evidence supports, and whether its self-critical hedging, applied throughout, has buried that finding instead of sharpening it.
**Possible blind spots**: will not audit the statistical machinery (bootstrap, sign-flip test, gate logic) or the quantization toolchain internals, and may underweight technical problems in the sweep half if the recommendation is to cut that half anyway.

### Reviewer Configuration Card #2

**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1
**Identity Description**: Statistician in machine-learning experimental methodology, specializing in the design and analysis of computational experiments: crossed random effects for training seeds, data order and sampling draws, and variance decomposition of benchmark results. Also specializes in selective inference for model selection (winner's curse, post-selection intervals) and in adaptive data analysis (holdout reuse, sequential looks, design of non-inferiority decision rules).
**Review Focus**:
  1. **Design of the calibration factorial.** Five seeds are crossed with selections shared across seeds, and nested 200/1,000 budgets are used. Intervals are conditional on the fitted models and the calibration sets, with seed and draw variation "reported separately". Does the inferential target (one fitted artifact vs a recipe) match the wording of the claims? Could the primary contrast's failure to manipulate input range (H1 "untestable as posed") have been foreseen with a manipulation check before freezing? If so, what evidential status is left for a study whose positive headline is a protocol-amendment secondary contrast?
  2. **Adaptivity and selection across the program.**
     - The ledger lists about 15 analyses, roughly 10 of which touch the same Weinstock held-out labels.
     - The delta-encoding variant was chosen on held-out data.
     - The round-one rejection used an override rule written down after the read and later generalized ("the two cases that arose fit one rule").
     - The stable-recall floor is relative to the champion and drifts.
     - Round five's +0.20 [+0.11, +0.29] comes from a best-of-five seed at a validation-chosen epoch, compared with one champion artifact under an interval that omits training-seed variance. Then it is set against a seed spread from a different cohort.
     - Does the "pre-registered/confirmatory" label survive this full analysis path?
  3. **Decision rules and small-sample tests.**
     - Gate 1 is worded "beats the champion" but operates as non-inferiority (lower bound > -0.5 points). The margin has no stated justification.
     - The frozen "near-zero" predictions have no margin and so cannot be falsified.
     - Multiplicity across the contrasts table (11 contrasts x 2 components).
     - Validity of the exact sign-flip test on participant "additive contributions" whose magnitudes scale with each participant's window count. Is the symmetry null appropriate, and what is it testing?
**Will particularly care about**: whether any claim labelled confirmatory is still confirmatory once every look, amendment and post hoc rule in the read ledger is counted, and whether reported effects are distinguishable from seed-level noise at the level of the recipe rather than of a chosen artifact.
**Possible blind spots**: may treat directional recall as a given endpoint without asking whether it is the right one; will not judge whether the quantization behavior is a toolchain artifact or the clinical meaning of the classes.

### Reviewer Configuration Card #3

**Role**: Peer Reviewer 2
**Display role**: Peer Reviewer 2
**Identity Description**: Machine-learning-for-health researcher in physiological time-series classification under severe class imbalance and ordinal targets. Specializes in evaluation-metric and operating-point design (cost-sensitive and ordinal metrics, threshold and logit adjustment, precision-recall trade-offs). Also works on external validation of wearable-sensor models across cohorts with label/prevalence shift and noisy, sensor-derived labels.
**Review Focus**:
  1. **Endpoint validity.** Directional recall is the unweighted mean of four recalls. It ignores precision and ordinal distance, and with inverse-frequency weights it selects an over-calling operating point. The bundled model predicts every movement class 1.7-5.2x as often as it occurs, falling_fast precision is 0.136 on held-out, 63% of displayed movement classes are followed by a stable rate, and the round-five PEDAP gain looks like an operating-point shift. Is the paper's central RQ2 "lesson" (macro-F1 optimized the wrong thing) reversible, i.e. is directional recall the more gameable metric? Would validation-tuned thresholds or logit adjustment, ordinal metrics (class-index MAE, weighted kappa), per-class AUROC/AUPRC or a cost-based frame separate discrimination from operating point?
  2. **Baselines and literature in the primary discipline.**
     - Test the claim that "no published baseline exists".
     - The baselines are weak: recent slope with untuned thresholds; logistic regression with fixed C=1.
     - Obvious comparators are missing: thresholded value forecasts from the GlucoBench model zoo or simple extrapolators (the Huber-regression collapse in round four is not the same test), gradient-boosted trees on the 12 readings, and an estimate of the label-noise ceiling.
     - Coverage of the class-imbalance, ordinal-classification and label-shift literature.
  3. **External validity of cross-cohort evaluation.**
     - Round five's primary cohort, RT-CGM, mixes DexCom SEVEN, MiniMed Paradigm and FreeStyle Navigator sensors that cannot be assigned to subjects. Its excess collinear-triple fraction (0.305 vs <=0.229) suggests interpolation or sample-and-hold, which would distort rate labels. Is it a valid ship-decision cohort?
     - Training (G5/G6, mixed or real-time wear, closed loop in part) differs from evaluation (blinded G4 Weinstock), and stable recall differs widely across cohorts (0.436 vs 0.605).
     - A two-sample 15-minute rate label from noisy readings is measured against +/-1 mg/dL/min thresholds.
     - What do "zero-shot" gains of 0.2 points mean under these shifts?
**Will particularly care about**: whether the metric that drove every ship decision measures a useful classifier or a prevalence-driven operating point, and whether a fair, literature-anchored baseline would change the paper's picture of what the CNN adds.
**Possible blind spots**: will take the quantization pipeline and latency claims at face value; may underweight the statistical-adaptivity issues R1 raises; is not a clinician and so may miss clinical-workflow implications of over-calling.

### Reviewer Configuration Card #4

**Role**: Peer Reviewer 3
**Display role**: Peer Reviewer 3
**Identity Description**: Mobile ML systems engineer-researcher working on the compiler and runtime stack for on-device inference: PyTorch 2 export (PT2E) quantization flows, TFLite/LiteRT conversion and operator lowering, and XNNPACK CPU kernels. Also publishes on on-device benchmarking methodology and cross-platform numerical reproducibility. This is a practical, systems-engineering angle on the deployment half of the paper, distinct from PTQ-algorithm research.
**Review Focus**:
  1. **Toolchain artifact vs INT8 property.** Section 7.9 reports that both convolutions, their bias additions and the pooling run in float32 on dequantized tensors, and only the final FC runs in int8. The paper never explains why: unannotated conv1d, the lowering path or a configuration choice. This bears on every deployment conclusion:
     - "INT8 is slower than float", "no measured speed benefit" and "the fix is free at deployment";
     - whether the stratification and observer nulls would hold in a full-integer graph (NPU/DSP/MCU targets), where internal activation-range coverage matters;
     - whether the RQ1 finding is really about clipping at a single graph-input quantize op.
     Also check: the observer's forced inclusion of zero, which wastes the codes below 39 mg/dL; the delta+anchor encoding sharing one per-tensor input scale across channels of very different magnitude; and the feasibility of the proposed fixed input qparams in LiteRT Torch.
  2. **Benchmark methodology.** A 2,909-parameter model is timed at the microsecond level with back-to-back calls on one iPhone. The timer includes quantize and dequantize, the thread count is unset, and only 9 of 110 artifacts are timed. In-app sessions report about 0.12 ms against about 3 us in the microbenchmark, a roughly 40x gap. Energy and memory are absent. Does any of this support a general latency statement? Are 31-70% savings on files of 12-208 KB practically meaningful? Is a byte-identical size and operator inventory sufficient evidence that the untimed artifacts behave the same way?
  3. **Reproducibility and parity engineering.**
     - Coverage of the 20-window golden parity set, and bit-exact INT8 claims across Arm devices and simulators.
     - LiteRT version skew: 2.1.0 on Android, 2.3.0 on iOS, Python 2.1.5.
     - Non-bitwise retraining across CPU and MPS (batch 64 vs 512), and what that means for reproducing reported numbers from released artifacts, hashes and per-participant counts.
     - The emulator workaround, and the adequacy of the iOS no-network source scan as a privacy guard.
**Will particularly care about**: whether the paper's deployment conclusions describe INT8 quantization of this model or an incompletely quantized graph produced by one converter configuration, and whether the claims are therefore scoped to the toolchain rather than to "on-device INT8".
**Possible blind spots**: will not judge statistical inference, endpoint choice or clinical meaning; may undervalue the paper's evaluation-protocol contributions because they are not systems contributions.

---

**Fixed fifth seat:** the Devil's Advocate is a fixed panel role, not dynamically configured by this analysis. No card is emitted for it.

## Review Strategy Recommendations

- **Unequal evidential grade across the two RQs.** The robust, prespecified-secondary RQ1 result (+13.38 [+10.29, +16.88]; clip-only control reproduces 13.14 of 13.35 points) sits beside a primary contrast that could not test its hypothesis and beside an RQ2 whose ship decisions rest on conditional, single-artifact intervals of +0.20 to +0.77 points. The synthesizer should grade each claim separately rather than let RQ1's strength carry RQ2.
- **Do not read self-disclosure as validation.** The manuscript discloses unusually much: a read ledger, deviations, post hoc labels, the protocol-to-report map, and the round-five protocol file misdescribing the calibration pool. Reviewers should check that each disclosed deviation leaves the claim it touches standing at the strength the abstract and Conclusion state, and should cross-check that headline numbers agree across the abstract, introduction bullets, results tables and Conclusion.
- **Held-out reuse is structural.** The calibration study is a "locked, corrected reanalysis" of previously scored participants, and rounds one to three, the delta-encoding experiment and several descriptive passes reuse the same Weinstock held-out split. Only round five's RT-CGM read is on labels no round had evaluated, and that cohort carries the sensor-provenance doubts raised in Card #3.
- **Toolchain generality is a gating question.** If R3 establishes that the hybrid float/int8 graph comes from configuration, the deployment-cost conclusions and the scope of the stratification and observer nulls narrow considerably. The synthesizer should resolve this before weighing R1's precision-oriented requests.
- **Integrity and disclosure item for the Journal-Fit Reviewer.**
  - The manuscript credits an "external review" (path under `paper/analysis/`, commit `3755d55`) with proposing round five's design, without identifying its nature or independence. It contains no AI-use statement.
  - Reviewers must not open that directory (this round is a cold read). They should ask the author to disclose what the review was and how it was produced.
  - **Manuscript-as-data check:** I found no text in the manuscript aimed at reviewers or the panel (no directives about reviewer identities, leniency, decisions or what to ignore). The revision-history appendix and read ledger are author disclosures and are handled as data.
- **Partial blind spot this round.** No seat is a CGM clinician or a regulatory-science specialist. The synthesizer and the Devil's Advocate should watch for:
  - the clinical meaning of a display that over-calls movement (63% of movement labels followed by a stable rate; near-zero fast-fall recognition below 70 mg/dL);
  - the "not a medical device" statement versus distributed apps with bundled weights and an LLM caption;
  - data-licence and DUA questions around publishing per-participant confusion counts.
- **Expected tensions.**
  - **R2 vs the paper's RQ2 lesson.** R2 may argue that directional recall, not macro-F1, is the metric that is easier to game. If that holds, the sweep program's main methodological lesson is contestable, and the Journal-Fit Reviewer must decide whether RQ2 remains a contribution.
  - **R1 vs R3.** R1 may ask for more seeds, recipe-level comparisons and multiplicity control. R3 may argue that some deployment conclusions are toolchain artifacts that more statistics cannot fix. Present these as ordered rather than additive requests.
  - **Journal-Fit Reviewer vs R1.** If the Journal-Fit Reviewer recommends splitting off or cutting the sweep half, many of R1's adaptivity concerns move with it. The synthesizer should give conditional recommendations (what is required if RQ2 stays, and what if it goes).
  - **R2 and R3 on delta+anchor encoding.** R2 sees an input-representation choice selected on held-out data; R3 sees a per-tensor scale-sharing design. Assign the selection issue to R1 and R2 and the quantization-design issue to R3 to avoid duplicate comments.
- **Register.** The manuscript is at pre-submission maturity, so a standard, evidence-based register fits. No developmental-feedback adjustment is needed.
