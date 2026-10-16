contract_role: perspective
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: warn
trigger: "Localized gaps that leave the central result and its scope correctly understood but make the paper harder to use, or easier to over-read, for one of the two audiences"

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

I read this manuscript as a diabetes-technology clinician-scientist (endocrinology and digital health) who works with CGM accuracy standards, sensor error, and the teaching of trend-arrow use to people with diabetes. I am an outsider to post-training quantization internals and to the bootstrap machinery, which other seats cover. My comments are about what the target, the reported errors and the cohorts mean to clinical readers, to app users, and to developers who may reuse the released models.

The scope and safety language is better than in most ML-for-health manuscripts I review. No health-facing benefit is presented as a finding, and the research-prototype restriction appears in the abstract, Introduction, Related Work, Discussion, Limitations, the data-availability statement and the Conclusion. None of my block conditions fires, and a clinical reader will understand the calibration findings correctly as written. I still score D4 warn, for three gaps a clinical reader would notice. The practical meaning of the shipped model's errors is never translated (W1). The cohorts behind the shipped model and its headline confirmation are never described (W2). And one conclusion about label noise goes further than the single, benign noise model that was tested (W3). Each can be fixed without touching the calibration study.

My Phase 1 checks map onto the findings as follows. Framing of the target: S2 and S8. Cross-field definitions: W4, W5 and W8. Translating metrics: W1 and W9. Claims that cross the field boundary: S5 (on-device privacy, backed by evidence) and W3 (label validity, not backed). Safety scope: S1 and W10. Limitations a clinical reader needs: S6, S7, W2, W6 and W7.

### S1: Scope and safety statements are explicit, repeated and consistent
**Evidence Anchor**: text: §10.2 "Its outputs must not be used for medical decisions." and §1 "they are not intended to guide treatment"

The restriction appears in seven places, including Limitations item 8 and §10.2, and the manuscript says the application shows it in its primary interface. Nothing in the abstract or conclusion contradicts it, which is the failure I see most often in this genre.

### S2: The target is carefully distinguished from commercial trend arrows
**Evidence Anchor**: text: §1 "reproduces a manufacturer's algorithm nor provides a dosing rule" and §3 "simplification based on two samples and does not describe the full intervening"

Clinicians will immediately ask whether the classes are trend arrows. The manuscript answers early and correctly. Arrows describe the recent or current rate over a device-specific look-back. This label is the mean rate over the next 15 minutes, computed from two samples, and the five classes collapse several device taxonomies. The ±1 and ±2 mg/dL/min cut-points are the ones clinicians know from arrow guidance, and the same cut-points appear in the FDA iCGM rate-of-change special controls. That makes the classes recognizable without claiming they are equivalent to arrows.

### S3: The accuracy illusion is shown in a form clinical readers already use
**Evidence Anchor**: text: §7.5 "and zero directional recall, which is why accuracy cannot rank"

Two observations carry the argument: the constant-stable comparator wins on accuracy, and the broken sequential artifact gains accuracy while losing minority recall. Together they are the ML version of a familiar clinical point, that a test's overall agreement depends on prevalence. A clinician needs no ML background to follow it.

### S4: The clipping mechanism is explained in physiological terms
**Evidence Anchor**: text: §8.1 "cannot recognize a fast fall."

Linking the falling_fast collapse to the fact that fast falls start high turns an INT8 range artifact into something an endocrinologist can reason about. Question 5 below suggests one further step.

### S5: On-device and no-network claims are backed by checks, not just asserted
**Evidence Anchor**: text: §5.3 "the merged Android manifest is tested to ensure it does not request"

The paper does not oversell privacy. It describes:
- a manifest test on Android;
- a CI source scan on iOS, honestly described as the weaker guard;
- a synthetic replay trace;
- no CGM data in the repository or the applications.

That is the right standard of evidence for reading on-device inference as a data-protection benefit.

### S6: Label provenance is reported honestly where interpolation could fake a trend
**Evidence Anchor**: text: §7.6 "describe the formatter's grid rather than"

Interpolated grid values form artificially straight segments, and a rate label computed across them is not something the sensor observed. The gap-fill mask, plus the plain statement that labels on gap-filled endpoints describe the grid rather than sensor truth, is exactly the caveat a CGM specialist would ask for.

### S7: Limitations name most omissions a clinician would raise
**Evidence Anchor**: text: §9 "not expert annotations, and the CGM-only model omits meal"

The Limitations state plainly that:
- meal, insulin, activity and illness covariates are missing;
- event-level performance and probability calibration were not evaluated;
- the app only replays data;
- there has been no prospective, usability or clinical validation.

### S8: An arrow-like persistence comparator is included
**Evidence Anchor**: text: §7.5 "recall against 0.279 for the recent-slope baseline and 0.461 for logistic"

The recent-slope baseline classifies the past 15-minute rate with the same thresholds. That is close to what a person does when they read the current arrow as a forecast. Including it gives clinicians a reference point they already think in (see Question 4).

### W1: The practical meaning of the shipped model's errors for people with diabetes is not translated
**Severity**: Major
**Evidence Anchor**: text: §7.12 "and the slow-to-fast over-prediction they cause"
**Confidence**: 4 — CGM trend interpretation and patient education are my core expertise; the arithmetic below uses only values the manuscript reports

The endpoint (mean recall of the four directional classes, gated by a stable-recall floor) is a sound way to expose quantization damage, and the §8.2 "practical reading" is correct for ML pipelines. A clinical reader, though, needs to know what the bundled model would show a person. Three facts that drive harm and trust are neither reported nor interpreted.

1. **False movement labels during stable glucose.** On the held-out component the constant-stable baseline reaches 0.727 accuracy, so about 73% of eligible windows are stable. The round-five model's held-out stable recall is 0.436. That means roughly 41% of all held-out windows would be labelled as moving while the next-15-minute rate stays within ±1 mg/dL/min. Even if every truly directional window were labelled as moving, at least about three in five displayed movement labels would fall in stable periods. Across the two promotions, held-out stable recall went 0.475, 0.459, 0.436 while held-out directional recall went 0.517, 0.523, 0.520. The paper discloses these numbers but never says what they mean: alarm fatigue, unnecessary snacking, or unnecessary corrections.
2. **Errors are not symmetric in cost.** §7.12 shows that the balanced class weights buy directional recall partly by over-calling the fast classes: 47% of falling errors and 46% of rising errors go to the same-direction fast class. The cited arrow guidance (Pettus and Edelman; Rodacki et al.) maps steeper arrows to larger insulin adjustments. A user who reads these class names through that guidance would over-correct on false rising_fast calls, with hypoglycemia downstream, and over-treat on false falling_fast calls. The disclaimers forbid this use, but clinical readers should be told which way the model errs.
3. **Direction reversals and glucose level are not reported.** For the round-two champion, 72% of errors are adjacent-class, so up to 28% are two or more classes away. The share that reverses direction is not given, and the error structure of the bundled round-five model is not reported at all. The clinically weighted question is also where errors happen. A missed falling_fast near 90 mg/dL is not comparable to one at 300 mg/dL, and the directional endpoint ignores glucose level.

**Suggested fix**: The manuscript says every published number regenerates from cached predictions, so these can be computed without retraining. For the bundled model, on the held-out component and on RT-CGM, add:
- (a) the five-class confusion matrix, with per-class recall labelled as sensitivity and per-class precision as positive predictive value;
- (b) the share of true falling_fast windows predicted as rising or rising_fast, and the mirror share for rising_fast. Present this descriptively against 21 CFR 862.1355(b)(1)(v)(J)-(K), which uses the same ±1 and ±2 mg/dL/min cut-points, and state that it is not a regulatory evaluation, since those criteria govern a measuring device rather than a forecaster;
- (c) directional recall stratified by the anchor reading into the time-in-range consensus bands (below 70, 70-180, 181-250, above 250 mg/dL; Battelino et al., Diabetes Care 2019).

Then extend the §8.2 paragraph with two or three sentences on what a displayed class does and does not mean. The minimum acceptable version is the rewritten paragraph plus the confusion matrix.

### W2: The cohorts behind the shipped model and its headline confirmation are not described
**Severity**: Major
**Evidence Anchor**: absence: §7.10 (round two), §7.11 (round three) and §7.13 (round five) — expected a characterization of the five DiaData training cohorts, the RT-CGM confirmatory cohort and the PEDAP probe (source trial, age range, diabetes therapy including automated insulin delivery, CGM model and generation, blinded or real-time wear, collection era, reportable range); checked Abstract, §1, §2.2, §4.1, §7.7, §7.10, §7.11, §7.13, §9, §10.2 and every table caption
**Confidence**: 5 — I verified the absence by reading the whole manuscript, and the authors' own supplementary audit file shows the information exists

The shipped model has never seen a Weinstock training window. It was trained on five DiaData cohorts the manuscript never names, and its promotion was certified on "RT-CGM", which the paper describes only by size and cadence. The authors' supplementary audit (paper/results/diadata_audit_r1.json) identifies them:
- **CITY**: adolescents and young adults, Dexcom G5.
- **DLCP3**: the iDCL Protocol 3 closed-loop trial, Dexcom G6.
- **REPLACE-BG**: adults, Dexcom G4 with 505 software.
- **SENCE**: young children.
- **WISDM**: adults aged 60 and over.
- **RT-CGM**: the JDRF CGM randomized trial (NEJM 2008), with a mix of DexCom SEVEN, Paradigm and FreeStyle Navigator sensors and the device per subject unrecoverable.
- **PEDAP**: the hybrid closed-loop trial in children aged 2 to under 6 (Wadwa et al., NEJM 2023).

For a clinical reader, this changes the interpretation in four ways.
1. **The cohort name reads as a device class.** RT-CGM (or rtCGM, as in the title of the cited Pettus and Edelman paper) is the standard clinical abbreviation for real-time CGM. An abstract that cites a never-read RT-CGM cohort will be read as evidence on current real-time CGM data. In fact the cohort was collected with first-generation sensors for a trial published in 2008.
2. **The future 15-minute trajectory is not the same physiological object across cohorts.**
   - Blinded wear (Weinstock) records unmanaged dynamics.
   - Real-time wear records dynamics after people react to their arrows.
   - Closed-loop arms (DLCP3, PEDAP) record dynamics shaped by a controller that itself acts on predicted glucose.
   A trend model trained across these regimes and deployed to users who will react to its output is a feedback-loop (performative prediction) setting, and that deserves at least one sentence.
3. **Secondary results only become clinically readable with the cohort description.** The 6.4-point drop in falling recall on PEDAP is a drop in preschool children on automated insulin delivery. In this group, recognizing hypoglycemia matters and the children cannot report symptoms themselves.
4. **Some populations are not represented.** There is no type 2 diabetes, pregnancy or inpatient data, although people with type 2 diabetes are a large and growing share of CGM users.

**Suggested fix**:
- Add a cohort table covering every GlucoBench and DiaData cohort used. Give each cohort's role (training, calibration pool, held-out, confirmatory, probe), source trial and citation, n, age range, diabetes type, therapy (MDI, pump, automated insulin delivery), CGM model and generation, blinded or real-time wear, collection years, and reportable range.
- Expand RT-CGM at first use, for example as the JDRF CGM randomized-trial cohort, labelled RT-CGM in DiaData.
- Add a Limitations item on therapy and feedback heterogeneity and on populations not represented.

This is rewriting, not new analysis.

### W3: The label-noise conclusion goes further than the single noise model tested
**Severity**: Major
**Evidence Anchor**: text: §11 "measured label-noise ceiling far above the champion places the remaining"
**Confidence**: 3 — based on my expertise in CGM sensor error; whether realistic noise would close the gap on these data is an empirical question I cannot settle

The round-four measurement is a good idea, and its own numbers already show how fragile a two-point label is. A least-squares slope over the same four horizon readings disagrees with the endpoint label on about 4.6% of directional windows on average. Added Gaussian noise of σ = 2 mg/dL changes about 14%.

But independent noise of 2 mg/dL is a gentle model of CGM error, and the σ = 2 choice carries no citation. Sensor error relative to blood glucose also includes:
- autocorrelated noise;
- slowly varying calibration error and drift;
- a blood-to-interstitial lag of several minutes;
- transient artifacts, such as pressure-induced compression lows and, in fingerstick-calibrated generations, step changes at each calibration.

The endpoint difference of two single readings 15 minutes apart is maximally exposed to independent and transient error, while slow bias largely cancels. So the realistic rate of label changes depends on the noise spectrum, on the sensor generation, and on whether calibration events fall inside windows. The cohorts span first-generation SEVEN/Navigator-era sensors through Dexcom G6.

§11 concludes that the remaining error lies "in the model and data, not the labels". That is stronger than what §7.12 actually measured. For a clinical reader it also reads as a statement about label validity, which the measurement does not address.

**Suggested fix**:
- **Minimum:** limit the §7.12 and §11 sentences to "under σ = 2 mg/dL white noise", and cite a CGM error model to justify the level (for example Facchinetti et al., IEEE Trans Biomed Eng 2014).
- **Stronger:** rerun the frozen validation measurement over a range of σ and with an autoregressive (colored) noise variant, reported by sensor generation. Also state that labels are defined on the sensor trace, which lags blood glucose, so the target is the future CGM-reported rate, not the blood-glucose rate.

### W4: "Calibration" is distinguished from probability calibration but not from CGM sensor calibration
**Severity**: Minor
**Evidence Anchor**: text: §1 "Here, coverage refers to the input values and trajectories"
**Confidence**: 5 — sensor calibration is the first meaning a diabetes-technology reader gives to "calibration" in a CGM title

To diabetes-technology readers, "calibration" in a CGM paper means calibrating the sensor against fingerstick blood glucose, or factory calibration. The title and the first sentence of the abstract use the bare word, and the clarifying sentence in §1 contrasts it only with probability calibration. If the source sensors were fingerstick-calibrated, the data also contain sensor-calibration events (W3), which doubles the ambiguity.

**Suggested fix**: Qualify the term in the title and abstract ("quantization-calibration coverage" or "calibration-set coverage"). Extend the §1 sentence to say it means neither CGM sensor calibration against blood glucose nor calibration of predicted probabilities.

### W5: Clinically loaded or imprecise wording about the target goes beyond the paper's own framing
**Severity**: Minor
**Evidence Anchor**: text: §7.12 "classified with the unchanged clinical thresholds" and §8.2 "a worse clinical-shape classifier" and Abstract "one of five rate-of-change classes 15"
**Confidence**: 4 — based on how clinicians use these terms

§3 carefully frames the boundaries as rate bands borrowed from arrow conventions and as engineering choices. But §7.12 twice calls them clinical thresholds, and §8.2 introduces an undefined "clinical-shape classifier". To a clinician, a clinical threshold means a validated decision cut-point, which is exactly the link to dosing that the paper disclaims. Separately, the abstract's "rate-of-change classes 15 minutes ahead" can be read as the rate at t+15, while the label is the mean rate over the next 15 minutes.

**Suggested fix**: In §7.12, write "the trend-arrow rate bands (±1 and ±2 mg/dL/min)". In §8.2, write "a worse directional classifier". In the abstract, write "the mean rate of change over the next 15 minutes".

### W6: The primary cohort's case-control enrichment and blinded wear are stated without their consequences
**Severity**: Minor
**Evidence Anchor**: text: §2.2 "whose source study collected blinded"
**Confidence**: 4 — the case-control design is stated in the abstract of the cited Weinstock et al. paper

Weinstock et al. (2016) was a case-control study. About half the participants had a severe hypoglycemic event in the prior year, and the cases showed greater glycemic variability. The primary cohort is therefore enriched for hypoglycemia and variability compared with older adults with type 1 diabetes in general. That affects the class prevalences and how far the 13.4-point effect transfers to other populations. Blinded wear also means participants could not react to their readings, so the future trajectories are unmanaged. Real-time users and closed-loop systems change those trajectories (see W2).

**Suggested fix**: Add the case-control design and its enrichment to §4.1 and Limitations item 2, and add one sentence on what blinded wear implies for deployment.

### W7: Readings at the sensor's reportable limits are treated as values, not as censored observations
**Severity**: Minor
**Evidence Anchor**: text: §7.4 "at the 401 sensor cap"
**Confidence**: 4 — based on CGM data-handling practice; I do not know how large the effect on labels is

The calibration pool spans 39-401 mg/dL, one unit outside the 40-400 mg/dL reportable range common to many CGM generations. That suggests LOW and HIGH placeholder values, and §7.4 itself calls 401 the sensor cap. 3.5% of held-out windows contain such a reading. This distorts labels in three ways:
- a rise into the cap is cut off, which biases labels toward slower classes;
- a plateau at HIGH is labelled stable while true glucose may still be climbing;
- if 39 encodes LOW, a stay below 40 mg/dL is also labelled stable, although it is the most dangerous state a CGM can report.

These are exactly the ranges where a clinical reader most wants the label to be right.

**Suggested fix**: State the reportable range and how LOW and HIGH are encoded in §4.1. Report the share of eligible windows whose anchor or endpoint is at a limit. Then exclude or flag those labels in a sensitivity tier, or at least add a Limitations item.

### W8: Several systems terms and acronyms are not defined for clinical readers
**Severity**: Minor
**Evidence Anchor**: text: Abstract "(100 INT8 artifacts)" and §7.12 "A CORN ordinal head reached 0.470."
**Confidence**: 4 — reading as a clinician without ML-systems training

The following are used without expansion at first use: INT8, PTQ, LiteRT, TCN (in the abstract), CORN (also uncited), macro-F1, zero-shot and T1D. Recall is never mapped to the clinical term sensitivity, nor precision to positive predictive value.

**Suggested fix**: Expand each term at first use, cite CORN (Shi, Cao and Raschka, Pattern Analysis and Applications 2023), and add a short glossary or footnote that maps recall to sensitivity.

### W9: The domain's own rate-of-change error frameworks are missing from Related Work
**Severity**: Minor
**Evidence Anchor**: text: §2.3 "Forecasting work evaluates RMSE, where direction surfaces only as a"
**Confidence**: 4 — CG-EGA is standard in CGM accuracy and glucose-prediction evaluation, and I checked that the cited de Bois et al. paper reports clinical acceptability with it

Glucose-prediction work often pairs RMSE with continuous glucose-error grid analysis (CG-EGA; Kovatchev et al., Diabetes Care 2004). Its rate component scores rate-of-change errors by clinical consequence, and the cited de Bois et al. paper itself reports clinical acceptability with CG-EGA. The FDA iCGM special controls also limit opposite-direction rate errors, using the same cut-points this paper uses. Placing the five-class target against these frameworks would show clinical readers that the domain already grades the paper's question (the direction and size of the coming rate). It would also make the "no published baseline" statement more convincing. Whether the current sentence misrepresents prior work is for the domain seat to judge; my point is the missed connection.

**Suggested fix**: Add one paragraph to §2.3 citing Kovatchev et al. 2004 and 21 CFR 862.1355. Explain why per-class recall was chosen instead and what it does not capture. Optionally, report a rate-error-grid-style summary from the cached predictions (see W1).

### W10: Developers who reuse the released model are a stakeholder the paper does not address
**Severity**: Minor
**Evidence Anchor**: text: Abstract "and now ships in both applications"
**Confidence**: 3 — reasoning about stakeholders and misuse; I did not inspect the repository's model documentation (the manuscript mentions MODELS.md only for hashes and provenance)

The application safeguards are good: replay only, no network access, and the restriction shown in the primary interface. But the paper releases models publicly, their class names mirror the arrows clinicians use for dosing, and the text repeatedly says the model "ships". A developer could connect the released artifact to a live CGM feed with little effort. The manuscript gives that reader no intended-use statement, no out-of-scope uses, no input requirements and no known failure modes.

**Suggested fix**: Replace "ships" with "is bundled in the research applications" in the abstract and conclusion. Add a short model card (Mitchell et al., FAT* 2019) to the repository and summarize it in §10.2, covering:
- intended use: offline research replay;
- out-of-scope uses: live decision support, alerts, dosing;
- input requirements: five-minute cadence, mg/dL, gap resets;
- known failure modes: frequent movement labels during stable glucose, censoring at the sensor limits, over-calling the fast classes;
- populations not represented.

### Cross-disciplinary reading recommendations
- **Trend-arrow use in practice:** Aleppo et al., J Endocr Soc 2017 (Dexcom, adults); Laffel et al., J Endocr Soc 2017 (children and adolescents); Kudva et al., J Endocr Soc 2018 (FreeStyle Libre). These describe how people are taught to act on arrows, which is the misuse pathway behind W1 and W10.
- **Evaluating rate accuracy:** Kovatchev et al., Diabetes Care 2004 (CG-EGA with its rate error grid); 21 CFR 862.1355(b)(1)(v)(J)-(K) for the iCGM rate-of-change criteria.
- **Sensor error and physiology:** Facchinetti et al., IEEE Trans Biomed Eng 2014 (CGM error model); Basu et al., Diabetes 2013 (blood-to-interstitial lag); Mensh et al., J Diabetes Sci Technol 2013 (compression artifacts during sleep).
- **Clinical glucose bands and CGM reporting:** Battelino et al., Diabetes Care 2019 (time-in-range consensus); Danne et al., Diabetes Care 2017 (international consensus on CGM use).
- **Reporting standards and feedback loops:** Mitchell et al., FAT* 2019 (model cards); Collins et al., BMJ 2024 (TRIPOD+AI, useful as a checklist even for a non-clinical prototype); Perdomo et al., ICML 2020 (performative prediction).
- **Primary sources for the cohorts already listed in the authors' audit file:** JDRF CGM Study Group, NEJM 2008; Wadwa et al., NEJM 2023; Brown et al., NEJM 2019; Aleppo et al., Diabetes Care 2017; Laffel et al., JAMA 2020; Pratley et al., JAMA 2020.

### Questions for the authors
1. Which blinded sensor did the Weinstock study use, and did it need fingerstick calibration? If it did, can calibration-related jumps be identified in the data?
2. How are LOW and HIGH readings encoded in each cohort, and how many scored labels touch a reportable limit?
3. For the bundled model, what are the confusion matrix, the per-class positive predictive values and the direction-reversal rates, on the held-out component and on RT-CGM?
4. Would you present the recent-slope baseline as the persistence assumption clinicians make when they read the current arrow as a forecast? The CNN's gain over it (0.487 versus 0.279 directional recall, at lower accuracy) is the most clinically interpretable comparison in the paper and deserves that framing.
5. The sequential artifact's ceiling (239.9-250.9 mg/dL) sits at the boundary of the time-in-range consensus band above 250 mg/dL. Would you say explicitly that the toolchain default made the INT8 model blind across that band? It would make the central finding immediately meaningful to clinicians.
6. Does the application show the predicted class next to the current glucose value, as commercial displays do? A rate label without the level invites misreading.

### Notes for other seats
- **For the methodology seat:** each round's stable-recall floor is set relative to the current champion, so held-out stable recall drifted from 0.475 to 0.459 to 0.436 across the two promotions. Separately, the RT-CGM archive appears in the authors' r1 audit with an undefined sampling interval and a higher collinear fraction than any cohort admitted in r1. In r2, all 15.28M of its intervals measure exactly 300 s. For a cohort with mixed first-generation sensors, that regularity suggests resampling, and its effect on rate labels should be stated.
- **For the domain seat:** whether the §2.3 description of how forecasting work is evaluated is accurate (see W9).
