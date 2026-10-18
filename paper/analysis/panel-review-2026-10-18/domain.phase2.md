contract_role: domain

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: block
block_class: repairable
trigger: "A generalization, deployment or cross-cohort claim goes beyond the reported evidence"

Basis: W1 and W2. Two RQ2 claims rest on directional recall alone: the sweep program's "main lesson" and the round-five adoption ("passed all gates", with INT8 within 0.25 to 0.5 points of its float). I recomputed metrics from the manuscript's own published per-participant counts. Chance-corrected ordinal agreement and per-class precision reverse both verdicts, and they show that the bundled INT8 artifact departs from its float about as much as the headline calibration failure does. This is repairable from cached results without retraining. RQ1's measured contrasts stand.

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

### Reviewer Identity
Peer Reviewer 2 (domain). I work in machine learning for health on physiological time-series classification under severe class imbalance and ordinal targets. My focus is evaluation-metric and operating-point design (cost-sensitive and ordinal metrics, threshold and logit adjustment, precision-recall trade-offs) and the external validation of wearable-sensor models across cohorts with label or prevalence shift and noisy, sensor-derived labels. I took the quantization pipeline as described and checked only how its outputs are evaluated. I am not a clinician.

### Overall Recommendation
Major Revision

### Confidence Score
4

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D2 (1): cited datasets, benchmarks, licences and prior results described faithfully | Contract D2; Phase 1 scoring plan item (1) | PARTLY_MEETS | Table 11 RT-CGM row; Sections 2.3, 4.1, Appendix B (see W5, S5) | I checked the Weinstock, GlucoBench, Kistkins, JDRF, CITY, DCLP3, REPLACE-BG, SENCE, WISDM, PEDAP and trend-arrow descriptions against primary sources or registries. Almost all are accurate. The RT-CGM row omits that the cohort includes a second randomized trial, misstates the wear mode as unstated, and misses that the 96 unscored subjects are recorded at a 10-minute cadence | I did not read the full text of the NEJM report; the wear mode comes from the trial registry and a companion report | no: Minor, text-level repair |
| D2 (2): imbalance-aware, ordinal, calibration and quantization terms used in their standard sense | Contract D2; Phase 1 item (2) | MEETS | Sections 1, 2.3, 4.3, 6.1 | Balanced accuracy, recall/sensitivity, PPV and the three meanings of "calibration" are all defined correctly, and the limits of per-class recall are stated | none identified | no |
| D2 (3): claims about operating point, selection lessons and deployment fidelity stay within the evidence | Contract D2; Phase 1 item (3) | DOES_NOT_MEET | Sections 9.2, 9.3; Tables 9, 20, 22 (see W1 to W4) | Endpoint-relative verdicts are presented as general lessons. Recomputed QWK and precision reverse the round-one rejection and the round-five adoption. The directional-recall fidelity gate misses a 45-72% rise in falling_fast calls caused by INT8 conversion | QWK and MAE themselves embed costs; without logits I cannot compute threshold-free metrics | yes: drives the D2 block |
| D2 (3b): claims about sensor-derived labels and cross-cohort transfer | Contract D2; Phase 1 item (3) | PARTLY_MEETS | Section 8.4; Limitations 7 and 8; Appendix B (see W6) | Label lag, sensor drift and wear and therapy heterogeneity are disclosed. The label-noise ceiling and differential sensor noise across cohorts are not quantified | depends on sensor error models that the paper did not simulate | no |
| D2 (4): one domain quantity agrees across abstract, text, tables and result files | Contract D2; Phase 1 item (4) and hint 1 | MEETS | Abstract; Sections 1, 7, 8, 12; Tables 6, 9, 20, 22, 23 (see S3) | I cross-checked about 25 headline values against the versioned JSON and count files, recomputing the error profile and parameter counts, and all match | I checked only the values I recomputed | no |
| D2 (5): no wording implies clinical validity | Contract D2; Phase 1 item (4) | MEETS | Sections 9.3, 11.2 (see S7) | The displayed-class semantics, the retrospective-only scope and the exclusion of medical decisions are stated precisely | the unevaluated LLM caption belongs to the Journal-Fit Reviewer's remit | no |

### Summary Assessment
This is an unusually transparent applied health-ML manuscript. It is careful about terminology, cohort provenance and what a trend label is and is not. Section 2.4 correctly places the calibration result (RQ1) as a measurement of a departure from established converter guidance. The clip-only control makes the mechanism credible.

The problem in my remit is the evaluation frame. One endpoint, the unweighted mean recall of the four minority classes, together with inverse-frequency class weights, drives every training, selection and ship decision. This choice fixes an operating point that rewards calling movement, yet the paper never states what an error costs. Using the per-participant confusion counts the author published, I recomputed chance-corrected ordinal agreement (quadratic-weighted kappa, QWK) and per-class precision:
- The macro-F1 winner rejected in round one is clearly better than the champion on both Weinstock components.
- The round-five artifact now bundled is worse than the champion on both measures on every component, the primary RT-CGM endpoint included.
- INT8 conversion of the bundled model raises its falling_fast call rate by 45-72%. On the Weinstock components, the QWK cost of that conversion equals the cost of the paper's own headline calibration failure, yet directional recall moves by 0.5 points or less.

The RQ2 "lesson" and the claim that recall-based gates guard deployment therefore go beyond the evidence. The baselines are compared at mismatched operating points. The RT-CGM cohort description contains verifiable errors. All of this can be fixed from cached outputs. The core of RQ1 survives.

### S1: Trend label carefully separated from device trend arrows
The paper states that its future-rate label is related to commercial trend arrows but not equivalent to them. It anchors the plus or minus 1 and 2 mg/dL/min bands to arrow conventions, says outright that the bands were not tuned on test labels, and explains why the endpoint-based label is not the instantaneous rate. This is the correct and rarely stated distinction.
**Evidence Anchor**: text: Section 1 "a research representation that neither replicates a manufacturer's algorithm nor gives a dosing rule"

### S2: Sensor-code awareness
Treating 39 and 401 mg/dL as likely stored LOW and HIGH codes, and noting that a coded endpoint makes the label only a bound on the rate, is domain-correct for Dexcom G4 receivers with a reporting range of 40-400. The exclusion check (+13.27 points) shows that the headline does not depend on these windows.
**Evidence Anchor**: text: Section 4.1 "The two values are therefore consistent with stored LOW and HIGH codes"

### S3: Error profile reported beyond recall, and numbers that reconcile
Tables 22 and 23 report precision, over-calling, adjacent-error and reversal shares, and full confusion matrices for the bundled model. Few CGM-classification papers report this much. From the counts in `panel_revision_r1.json` I recomputed the per-class recall and precision, the 1.7-5.2x over-call range, the 63% of displayed movement that is followed by a stable rate, and the 92% and 94% precision of a displayed stable class, and all matched.
**Evidence Anchor**: table: Tables 22 and 23 (bundled-model error profile and row-normalized confusion matrices)

### S4: Honest positioning of the calibration result
The paper does not claim as new the rule that calibration data must represent deployment inputs. It frames RQ1 as a measured departure from documented guidance in a setting (stride-one overlapping windows) where that departure is easy to make. That framing is accurate.
**Evidence Anchor**: text: Section 2.4 "is therefore established guidance, not a finding of this paper"

### S5: Cohort descriptions that state their limits on transfer
The primary cohort is characterized correctly against Weinstock et al. (2016): a case-control design enriched for severe hypoglycaemia, with blinded wear. The paper draws the right consequence for prevalence and transfer, and Table 11 marks unverified fields instead of guessing.
**Evidence Anchor**: text: Section 4.1 "The cohort is therefore enriched for hypoglycemia and variability relative to older adults"

### S6: Published per-participant counts that make the evaluation recomputable
Publishing per-participant confusion counts for every cached arm, under the source licences, is exemplary. It is what allowed the independent re-analysis in W1 to W4 without access to the data.
**Evidence Anchor**: dataset: paper/results/calibration-coverage/participant_counts_glucobench_r1.json.gz and participant_counts_diadata_r1.json.gz

### S7: Precise non-clinical semantics of a displayed class
Section 9.3 defines a displayed class as an estimate of the rate band the sensor trace will average over the next 15 minutes, not a measurement of blood glucose. It also refuses to read the error profile as evidence about use. This is exactly the right scope.
**Evidence Anchor**: text: Section 9.3 "they are not evidence about how a person would use its output"

### W1: The RQ2 "lesson" and the round-five adoption are endpoint-relative, and reverse under ordinal agreement and precision
Directional recall is a mean of minority-class recalls, so it rises whenever the model predicts movement more often. Paired with inverse-frequency class weights, argmax behaves approximately like a Bayes rule that prices an error on a true falling_fast window at about 24 times an error on a true stable window (74.04% / 3.12%, Weinstock training prevalences in Table 2; the threshold-cost correspondence of cost-sensitive learning, Elkan 2001). The paper never states or defends this cost structure. The stable-recall floor is relative to each champion and drifted from 0.475 to 0.436, as Section 9.3 notes.

I pooled the published per-participant counts (`participant_counts_*_r1.json.gz`) into confusion matrices and computed quadratic-weighted kappa (QWK; Cohen 1968) and per-class precision. These are descriptive, post hoc recomputations of frozen predictions:
- **Round one.** The rejected macro-F1 winner beats the champion on QWK: held-out 0.439 vs 0.385, temporal 0.438 vs 0.381, five-seed candidate mean 0.437. Its falling_fast precision is 0.256 vs 0.190 (Table 20). It was trained with a class-distance (ordinal) penalty, and it is the better ordinal classifier. Section 9.3 calls this a case of an endpoint that "optimize[d] the wrong thing". It is equally consistent with the opposite reading: directional recall rewarded the champion's over-calling (falling_fast predicted 3.0x as often as it occurs, against 2.0x for the candidate).
- **Round five.** The bundled candidate has lower QWK than the champion on all four components: RT-CGM 0.432 vs 0.441, held-out 0.359 vs 0.381, temporal 0.345 vs 0.370, PEDAP 0.467 vs 0.478. Its falling_fast precision is also lower on all four (RT-CGM 0.214 vs 0.278). The +0.20-point RT-CGM directional gain coincides with lower agreement and precision on that same primary cohort.

QWK is not the "right" metric either. Class-index MAE, for example, favours the constant-stable classifier (held-out 0.355 against the bundled model's 0.786). The point is that the paper's verdicts depend on an unstated cost choice and should not be presented as general methodological evidence.

Minimum remedy, from cached counts with no retraining:
- Restate the lesson as endpoint disagreement under a declared cost structure.
- Report QWK (or an R-EGA-inspired cost matrix in which reversals cost more than adjacent errors) and per-class precision beside every confirmatory directional-recall read.
- Report the result of the stable-logit offset scan that Table 13 lists but the manuscript never gives.

Stronger option: compare candidate and champion at matched operating points via validation-tuned post-hoc logit adjustment (Menon et al. 2021; Saerens et al. 2002), and add per-class one-vs-rest AUROC/AUPRC from cached logits.

Norm grounding: Saito and Rehmsmeier (2015, PLoS ONE) argue that under imbalance, precision-recall evaluation is more informative than recall- or ROC-only summaries. Menon et al. (ICLR 2021) show that post-hoc logit adjustment moves a trained model along its operating curve, so comparing two artifacts at uncontrolled operating points conflates discrimination with threshold. The severity rests mainly on the reversal in the paper's own data, not on the norm alone.
- **Severity**: Major | **Evidence Anchor**: text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | **Confidence**: 4 — recomputed directly from the published counts; ordinal and operating-point metrics are my core area; I had no logits for threshold-free metrics

### W2: INT8 conversion of the bundled model shifts its falling_fast operating point, and the recall-based fidelity gate cannot see it
Section 9.2 offers, as the study's practical reading, that directional or macro recall "exposes" a damaged quantized artifact. Section 9.3 attributes the bundled model's over-calling to "inverse-frequency class weights and a directional-recall endpoint". The published counts for the round-five pair (float vs INT8; `sweep-r5` records) show a third cause: the conversion itself.
- **Falling_fast call rate:** held-out 11.0% to 17.8% of windows (3.21x to 5.19x prevalence), temporal 8.5% to 14.6%, RT-CGM 5.3% to 7.6%.
- **Falling_fast precision:** 0.193 to 0.136, 0.162 to 0.111, and 0.278 to 0.214 on the same three components.
- **QWK:** 0.387 to 0.359 on held-out and 0.377 to 0.345 on temporal. That is the same size as the paper's headline calibration failure measured the same way (float to sequential-200: 0.393 to 0.366 held-out, 0.386 to 0.354 temporal).
- **Directional recall:** moves only 0.525 to 0.520, 0.509 to 0.505 and 0.557 to 0.555 (Table 9), so gate 3 passes.

The float flatten-head model is in fact almost indistinguishable from the champion. On held-out it has QWK 0.387 vs 0.381 and a falling_fast share of 11.0% vs 11.7%. Much of the deployed model's worst over-calling is therefore introduced by conversion and not by training. The effect recurs in the wider models and runs in both directions (the round-three seeds shift little): the round-two winner's falling_fast over-call went from 4.23x to 3.41x, and its seed 2 from 3.29x to 4.27x, each with directional recall unchanged within 0.1 points. By contrast, the 2,909-parameter study model under uniform-200 is faithful (falling_fast 2.70x float vs 2.71x INT8), so the RQ1 nulls are unaffected.

Remedy:
- Gate INT8 fidelity on per-class prediction rates or window-level agreement with the float model, not on directional recall.
- Report float and INT8 profiles side by side in Tables 22 and 23.
- Correct the attribution in Section 9.3.
- Qualify the Section 9.2 recommendation.

The operator-level mechanism (per-tensor quantization of the 1,728-feature flatten input under cross-cohort calibration is one hypothesis) belongs to Peer Reviewer 3.
- **Severity**: Major | **Evidence Anchor**: text: Section 9.2 "Directional (minority-class) recall or five-class macro recall exposes it immediately" | **Confidence**: 4 — computed from published per-participant counts for the float and INT8 arms; window-level agreement and the cause are not visible to me

### W3: "Worse directional classifier" overstates RQ1's degradation; the magnitude is largely an operating-point shift
The direction of the RQ1 finding survives a chance-corrected check. QWK favours uniform-200 over sequential-200 at every seed, by 0.020-0.029 on held-out and 0.023-0.033 on temporal.

The 13.4-point framing is endpoint-specific, however. Sequential-200 predicts falling_fast at 1.09x its prevalence, against 2.71x for uniform-200, and its falling_fast precision is slightly higher (0.220 vs 0.204). Five-seed macro-F1 differs by about 0.3 points (seed 0 favours sequential). Class-index MAE favours sequential at every seed. Clipping thus moves the artifact toward the prevalence-matched operating point and also costs some ordinal discrimination.

The deployment-relevant fact, and the cleaner claim, is loss of fidelity to the validated float model. The 8-point drop in macro recall, the metric that the inverse-frequency-weighted training targets, also supports that claim. Reframe Section 9.2 and the Conclusion's "damaged" and "accuracy illusion" wording around fidelity and the declared cost frame, and report QWK or prediction agreement for the RQ1 arms.
- **Severity**: Minor | **Evidence Anchor**: text: Section 9.2 "The sequential-arm INT8 model is a worse directional classifier with better accuracy" | **Confidence**: 4 — per-seed recomputation from published counts

### W4: Baselines are compared at mismatched operating points, and no forecaster-derived baseline is included
The untuned recent-slope baseline predicts every class at its prevalence (1.00-1.03x), while the float CNN predicts falling_fast 2.70x as often as it occurs. At these mismatched operating points, the CNN's lead of 0.487 vs 0.278 measures mostly operating point. Recomputed from the published counts:
- **Macro-F1:** slope 0.383, CNN five-seed mean 0.377 (seed range 0.345-0.392).
- **QWK:** slope 0.383, CNN 0.393 (seed range 0.383-0.405).
- **Logistic regression:** directional recall 0.461, but QWK 0.338 and macro-F1 0.306, so it ranks below the slope baseline on both.

The claim that "no published baseline exists for this exact task" survived a targeted search, but it understates the obvious comparator. Any 15-minute value forecaster, including the GlucoBench model zoo the paper already uses, induces a class prediction through the paper's own thresholds. The round-four Huber arm is not that test, because it was decoded at fixed bands without operating-point tuning, and Section 8.4 itself attributes its collapse to imbalanced regression.

Remedy:
- Tune the slope and logistic-regression thresholds or logit offsets on validation to the CNN's stable recall, then compare.
- Add a thresholded 15-minute GlucoBench forecaster under the same matching.
- Tune the logistic-regression C.
- **Severity**: Minor | **Evidence Anchor**: text: Section 7.5 "On held-out participants the five-seed float CNN reaches 0.487 directional recall against 0.278 for the recent-slope baseline" | **Confidence**: 4 — recomputed from the published baseline counts; the forecaster baseline is a design recommendation, not a measured result

### W5: The RT-CGM description contains verifiable errors that bear on round five's primary cohort
Three details are inaccurate:
- **Cohort composition.** ClinicalTrials.gov NCT00406133 lists an actual enrollment of 451 and a parallel randomized trial in an HbA1c below 7.0% cohort under an identical protocol. The JDRF CGM Study Group's 2009 Diabetes Care report of that trial randomized 129 participants aged 8-69. So 322 + 129 = 451 explains the discrepancy that the Table 11 caption leaves open, and the cohort includes a better-controlled sub-population that the cited NEJM report does not describe.
- **Wear mode.** The same registry record specifies a blinded first week of CGM for every subject and the download of blinded CGM data in the control group. The control-crossover extension report also starts CGM with a blinded week. In the paper's own taxonomy the wear is therefore "mixed", not "not stated".
- **Cadence.** A review by the DiaData authors (Cinar, van den Boom and Maleshkova, arXiv:2601.11615, dataset table) lists this dataset's sampling as "5 min (355) and 10 min (96)". That matches exactly the 355 scored and 96 unscored subjects in Appendix B. The 96 excluded subjects are therefore very likely 10-minute records that never form runs of 15 five-minute readings. Cadence is thus partly recoverable per subject, and the Appendix B sample-and-hold hypothesis should be re-examined: the subjects most exposed to resampling contribute no windows, which leaves the collinearity excess among 5-minute records unexplained.

Remedy: correct the row and caption, and report whether the round-five contrast holds within the HbA1c 7.0% or higher main cohort.
- **Severity**: Minor | **Evidence Anchor**: table: Table 11, RT-CGM row (n, Age and CGM; wear columns) | **Confidence**: 4 — registry record and DiaData-author review read directly; the mapping of 96 subjects to 10-minute records is inferred from the exact count match

### W6: The label-noise ceiling and differential sensor noise across cohorts are not quantified
The label is a two-sample, 15-minute rate compared against a 1 mg/dL/min boundary. Even under the paper's gentle white noise (sigma of 2 mg/dL), the label agrees with itself on only 0.860 of directional windows. The paper cites the Facchinetti et al. (2014) sensor-error model but does not simulate with it.

This matters for interpretation in two places. Directional recalls near 0.5 cannot be read against an unknown ceiling. The zero-shot comparisons span 2008-era sensors (RT-CGM), G4 (Weinstock) and G6 (PEDAP), whose different error structures change label noise as well as dynamics, so cross-cohort differences in recall mix model transfer with label reliability.

Remedy: a Facchinetti-model simulation of per-class label flip rates at each boundary, plus the per-cohort share of windows whose rate lies within a small margin of a boundary.
- **Severity**: Minor | **Evidence Anchor**: text: Section 8.4 "White noise of this size is also gentler than published CGM error models" | **Confidence**: 3 — the direction of the effect is standard; its size here is unmeasured

### W7: Literature coverage misses operating-point, label-shift and ordinal-evaluation work
The evaluation frame is the paper's central methodological device, yet the bibliography has no entry on:
- post-hoc prior or logit adjustment (Saerens et al. 2002; Menon et al. 2021);
- label shift across cohorts (Lipton et al. 2018);
- ordinal classification metrics (Cohen 1968; Gaudette and Japkowicz 2009);
- precision-recall evaluation under imbalance (Saito and Rehmsmeier 2015);
- cost-sensitive decision thresholds (Elkan 2001).

It also lacks a recent survey of glucose and hypoglycaemia prediction by the DiaData authors (Cinar et al. 2026), which compiles classification studies and the cohort facts used in W5. Adding these would anchor W1 to W4's remedies and the claim that no published baseline exists.
- **Severity**: Minor | **Evidence Anchor**: absence: Section 2.3 and the bibliography — expected literature on post-hoc operating-point adjustment, label shift and ordinal evaluation metrics; checked Sections 2.3, 9.2, 9.3, 10 and all 68 bibliography entries | **Confidence**: 4 — bibliography read in full; the listed works are standard in imbalanced and ordinal evaluation

### Detailed Comments

#### Literature Review
- **Coverage**: CGM forecasting, benchmarks, trend arrows, error grids and PTQ calibration are well covered and accurately represented. I verified the Kistkins et al. (2023) summary against its Table 2 (11 participants, 15-minute data, within-patient splits); it is accurate apart from a minor loose range (see Minor Issues). Coverage of imbalance, operating point, label shift and ordinal evaluation is missing (W7).
- **Integration quality**: Critical synthesis, not enumeration. Table 1's separation of offline evaluation from device evidence is useful.
- **Research gap argument**: Persuasive for RQ1 as a measured departure in overlapping time-series windows. For RQ2 the gap is a case study, and its stated lesson needs reframing (W1).

#### Theoretical Framework
- **Appropriateness**: The implicit framework is balanced-error classification: inverse-frequency weighting with a recall endpoint. It is coherent but never stated as a decision-theoretic choice.
- **Application depth**: The framework is applied consistently but not examined. The paper already cites CG-EGA and its rate-error grid as grading reversals more severely than adjacent errors, but uses that only to motivate per-class recall.
- **Alternative frameworks**: A cost-sensitive frame (an explicit cost matrix, for example derived from R-EGA zones) or an ordinal frame (QWK, cost-weighted error) would make the ship decisions defensible and would have exposed W1 and W2.

#### Academic Argument Quality
- **Factual accuracy**: Cohort, sensor and trend-arrow facts are accurate except for RT-CGM (W5). Numbers reconcile across the abstract, body, tables and result files (S3).
- **Argument logic**: The leaps are from "endpoints disagree" to "macro-F1 optimized the wrong thing" (W1), and from "directional recall within 0.5 points of float" to fidelity (W2).
- **Terminology precision**: Good throughout. "Calibration" is disambiguated, and "balanced accuracy" is correctly equated with five-class macro recall.

#### Contribution to the Field
- **Incremental contribution**: RQ1 shows that, for stride-one CGM windows, contiguous calibration selection becomes a class-specific operating-point shift concentrated in the fast-falling class, and the clip-only control localizes that shift. This is a useful, transferable caution for imbalanced ordinal time-series deployment. RQ2 is a documented case study of staged selection.
- **Positioning**: RQ1 is accurately positioned against converter guidance (S4).
- **Overclaiming**: Mainly in how recall-based metrics are presented as the arbiter of model quality and fidelity (W1 to W3).

#### Missing Key References
- Menon, Jayasumana, Rawat, Jain, Veit and Kumar (2021), "Long-tail learning via logit adjustment", ICLR. Post-hoc operating-point adjustment, needed for matched-operating-point comparisons (W1, W4). Existence checked in this session.
- Saerens, Latinne and Decaestecker (2002), "Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure", Neural Computation. Prior adjustment across cohorts with different class prevalence.
- Lipton, Wang and Smola (2018), "Detecting and correcting for label shift with black box predictors", ICML. Label shift across the zero-shot cohorts.
- Cohen (1968), "Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit", Psychological Bulletin. The ordinal agreement measure used in W1 to W4.
- Gaudette and Japkowicz (2009), "Evaluation methods for ordinal classification", Canadian Conference on Artificial Intelligence (LNCS). Choice of ordinal metrics. Existence checked in this session only through citing papers.
- Saito and Rehmsmeier (2015), "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets", PLoS ONE. Norm for reporting precision under imbalance.
- Elkan (2001), "The foundations of cost-sensitive learning", IJCAI. Correspondence between class weights, thresholds and costs (W1).
- Cinar, van den Boom and Maleshkova (2026), "A review on machine learning approaches for the prediction of glucose levels and hypoglycemia", arXiv:2601.11615. Classification-study survey and JDRF dataset facts (W5, W7). Read in this session.
- Search lead `[UNVERIFIED]`: studies of the accuracy of CGM trend arrows or rate-of-change against reference glucose, as an empirical anchor for the label-noise ceiling (W6).

### Questions for Authors
1. What did the stable-logit offset scan of cached round-two and round-three predictions (Table 13, commit 3755d55) find? At matched stable recall, does any candidate dominate its champion on directional recall?
2. What error costs is the display meant to encode? Is a reversed direction worse than an adjacent band, and is a false movement call on a stable window meant to be about 24 times cheaper than a missed fast fall, as the class weights imply?
3. Are logits cached for the confirmatory reads, so that per-class AUROC/AUPRC and matched-operating-point contrasts can be reported without retraining?
4. For the bundled pair, what is the window-level float and INT8 prediction agreement, and which tensor's quantization drives the rise in falling_fast calls (W2)?
5. Can HbA1c-cohort membership and 5- versus 10-minute cadence be recovered per RT-CGM subject from the Jaeb public dataset, and does the round-five contrast hold within the main HbA1c 7.0% or higher cohort?

### Minor Issues
- Table 22: the time-in-range consensus places 180 mg/dL inside 70-180 and 250 inside 181-250. The paper assigns both to the upper band, which is disclosed. Write "edges adapted from" rather than "edges from".
- Table 22 and Section 9.3: the held-out "below 70 mg/dL" directional recall averages in a falling_fast recall computed from a single window. Report band recall only over classes with adequate support, or pool the classes.
- Report class prevalence per cohort beside precision. Precision depends on prevalence: falling_fast is 2.43% of RT-CGM windows against 3.43% of Weinstock held-out windows (computed from the published counts), and none of the PEDAP or RT-CGM prevalences appear in the manuscript.
- Section 2.3: "best models reach recalls of 0.85-0.98" is slightly loose. In Kistkins et al.'s Table 2, LSTM euglycaemia recall at 15 minutes is 0.78 and logistic-regression hyperglycaemia recall at 1 hour is 0.79.
- Section 3: some arrow taxonomies separate a further band above 3 mg/dL/min. Saying that the five classes match some device taxonomies more closely than others would make "commonly separate" precise.
