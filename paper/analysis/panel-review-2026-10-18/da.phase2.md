contract_role: da

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: block
trigger: "At least one central claim is not supported by the evidence as presented"
block_class: repairable

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

### Calibration Status

`NOT_CALIBRATED`

This is a seat report. The final panel topology is not known until every seat has finished, so the seat report stays uncalibrated.

### Criterion-Bound Judgements

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D3 argumentative_coherence | contract D3 and the DA Phase 1 scoring plan | PARTLY_MEETS | tab:sweep (table) | RQ1 chain holds; RQ2 lesson and ship logic do not (M1 to M3) | depends on treating RQ2 as central | yes; drives the block |
| Core thesis challenge | DA challenge 1 | PARTLY_MEETS | tab:protocolmap (table) | RQ1 primary untestable as posed; positive result is a descriptive secondary | RQ1 effect is large and mechanistically localized | yes |
| Cherry-picking | DA challenge 2 | PARTLY_MEETS | tab:readledger, row 3755d55 (table) | one held-out read (offset scan) has no reported outcome | outcome unknown to this seat | yes, via M2 |
| Confirmation bias | DA challenge 3 | DOES_NOT_MEET | tab:sweeprecall (table) | operating-point shift penalized in round one, accepted in round five | none identified | yes, via M2 |
| Logic chain | DA challenge 4 | PARTLY_MEETS | tab:mechanism (table) | RQ1 mechanism chain valid; RQ2 lesson needs an unstated premise | none identified | yes, via M1 |
| Overgeneralization | DA challenge 5 | PARTLY_MEETS | tab:baselines (table) | RQ1 nulls well scoped; accuracy-illusion takeaway is not | m3 is minor | no |
| Alternative paths | DA challenge 6 | PARTLY_MEETS | tab:devicebench (table) | ship float and operating-point adjustment not weighed | none identified | no |
| Stakeholder blind spots | DA challenge 7 | PARTLY_MEETS | tab:shippederrors, anchor-band rows (table) | clinical and regulatory voices absent, as Phase 0 flagged | identification only | no |
| So what | DA challenge 8 | PARTLY_MEETS | tab:calibmain (table) | RQ1 quantifies a known error; RQ2 gives no robust affirmative answer | contribution is the EIC's call | no |
| Field-norm severity calibration | DA challenge 9 | MEETS | git commit 3755d55 message (dataset) | only M4 depends on a norm, and it is grounded in COPE | none identified | no |

### Score Rationale (argumentative coherence)

The RQ1 thesis survives the stress test. Every number I cross-checked agrees across the abstract, Introduction, results tables, Conclusion and the versioned result files:
- the uniform-versus-sequential contrast (+13.38 [+10.29, +16.88]);
- the mechanism decomposition (13.12 + 0.23 = 13.35);
- the shipped model's recall, precision, over-call ratio (5.2 for falling_fast) and 63% movement-followed-by-stable figure, all recomputed from the confusion counts in `panel_revision_r1.json`;
- the Dubosson sign-flip p (0.09375, recomputed from the six published contributions);
- the claim that round-five macro-F1 is lower on all three paired splits (RT-CGM 0.410/0.427, held-out 0.342/0.364, temporal 0.335/0.356 in `accuracy_sweep_r5.json`).

No headline number disagrees with its source. The paper also scopes its RQ1 nulls correctly to the saturated regime.

The block comes from the second research question, which takes half the title, half the abstract and the second paragraph of the Conclusion. That half has three problems:
- Its stated "main lesson" depends on an unargued endpoint premise, and the paper's own error profile reverses it (M1).
- Its two decisive post-read judgements apply one diagnosis, an operating-point shift, in opposite directions, and the rule said to reconcile them was written after both decisions (M2).
- The one gated piece of evidence behind the bundled artifact comes from a cohort whose label validity the paper itself doubts (M3).

No single finding is singleton-critical: RQ1 stands, and each RQ2 defect can be repaired by rewording plus at most one analysis on cached predictions. Together, though, they mean the RQ2 thesis cannot stand as written without substantive rewording or new analysis. That makes the block repairable, not fatal. The fatal trigger is not met because the central thesis is not contradicted by the paper's own results, and nothing in RQ1 rests on a structural fallacy.

Scope limit: this grading treats RQ2 as a central claim, as the title and abstract present it. If the synthesizer treats RQ2 as secondary, the same findings describe overreach rather than an unsupported central claim.

### Strengths Acknowledged

- The clip-only control turns an association (sequential calibration, collapsed falling_fast recall) into a localized mechanism: 13.14 of 13.35 points.
- The read ledger, protocol-to-report map and deviation notes are far more complete than is usual for applied health-ML work.
- Numeric bookkeeping is exact in every check I ran.

As the field analysis warns, none of this disclosure repairs the RQ2 logic gap. The paper discloses most of the facts behind M1 to M3, but its summary claims do not take them on board.

### Strongest Counter-Argument

A scholar holding the opposite view would argue that neither half delivers what the title promises.

RQ1's positive result is a careful measurement of a known pipeline error. Two hundred contiguous stride-one windows from two participants cap the graph input at 250.9 mg/dL, and clipping high readings removes what the classifier needs. The pre-registered primary hypothesis was untestable as posed, and every positive contrast is descriptive under the study's own plan. The held-out participants had already shown the pattern under the earlier protocol, and neither frozen transfer prediction was confirmed. The stakes are thin in this deployment as well:
- only the final layer computes in int8;
- INT8 is slower than float on every measured call;
- the applications run float by default;
- the study model saves 5,376 bytes.

The practical rule reduces to "do not clip the input", which shipping float satisfies.

RQ2 is weaker. Its general lesson, that a pre-registered endpoint can pass every gate and still optimize the wrong thing, holds only if directional recall is the right target, and the paper never argues that it is. The round-one candidate it rejected had higher precision in all four directional classes and the highest held-out macro-F1 in the sweep table. The directional-recall-selected bundled model labels 56% of truly stable held-out windows as movement, and its falling_fast precision is 0.136. The program then shipped a round-five artifact whose case rests on:
- one gated gain of +0.20 points, a fifth of the seed spread the paper cites;
- a cohort that the project's own audit had excluded and that the paper suspects of label distortion;
- a loss of 1.7 to 2.2 macro-F1 points on every paired split;
- two failed gates on both Weinstock components.

On this reading, the sweeps show an endpoint choosing an operating point, not a better model.

### Ignored Alternative Explanations and Paths

1. **Round five's RT-CGM gain is an operating-point shift.** The paper makes exactly this diagnosis for PEDAP (fast-class recall up, slow-class recall down, falling_fast precision down) but never checks it on the gating cohort. The macro-F1 drop on RT-CGM fits this reading. The cached RT-CGM confusion counts are enough to settle it.
2. **The accuracy illusion comes from inverse-frequency weighting, not from quantization alone.** The float model's accuracy (0.507) is far below the constant-stable baseline (0.727). For such a model, any perturbation that pushes predictions toward stable raises accuracy. A model trained without class weights could lose accuracy under the same clipping. The illusion is real here, but its generality is conditional on the operating point.
3. **Operating-point adjustment instead of retraining.** The round-one rejection, the round-three gate-2 failure and the round-five gain all turn on where the argmax lands. A validation-fitted logit offset might reproduce these changes without any new model. The stable-logit offset scan listed in the read ledger is this analysis, and the manuscript does not report its outcome.
4. **Ship float.** Given INT8's measured profile (no speed benefit, mostly-float graph, float default in the apps), the simplest alternative to the calibration fix is to not quantize. The paper's case for INT8 rests on size alone, and it never says why that saving matters for a research replay app.

### Missing Stakeholder Perspectives

- **People who use CGM, and the clinicians who advise them.** The error modes they would weigh most are frequent movement labels during stable glucose, and fast falls below 70 mg/dL that the model almost never recognizes (0 of 1 held-out, 2 of 96 RT-CGM). This round has no clinical seat.
- **Regulatory science.** It is an open question how publicly distributed apps that display a trend class sit within software-as-a-medical-device frameworks, whatever the disclaimer says.
- **Data providers and curators.** They have a stake in the publication of per-participant confusion counts under the source licences.

### Unexamined Premise

The paper treats the argmax operating point of an inverse-frequency-weighted model as a property of the model. As a result, every gate, rejection and ship decision compares operating points rather than discrimination. No threshold-free or matched-operating-point comparison appears anywhere in RQ2. If the operating point is really a deployment choice, then three of RQ2's key results may only say where the argmax lands, not which model is better:
- round one's "wrong thing";
- round three's stable-recall failure;
- round five's +0.20 gain.

### Minor Issues

- **m1.** The abstract calls the primary contrast null, while the protocol-to-report map classifies H1 as untestable as posed. A null reads as tested with no effect, and the abstract should use the body's classification. (anchor: text: Abstract "range stratification, the primary contrast, was null"; confidence 4, wording compared directly)
- **m2.** The abstract describes the study model as a network in 8-bit integers. Section 7.9 shows that only the final fully connected layer executes in int8, so the abstract overstates how integer the deployed graph is. (anchor: text: Section 7.9 "The artifact is also less integer than its name"; confidence 4, wording compared directly)
- **m3.** The accuracy-illusion takeaway holds for pipelines whose float operating point under-predicts the majority class (see alternative 2), not for any pipeline. (anchor: text: Section 9.2 "any pipeline that gates a quantized model on raw accuracy would have promoted the broken artifact"; confidence 3, inference from the baseline table)
- **m4.** The 13.4-point magnitude comes from one deterministic ordering. Contiguous selection has no draw-to-draw variance by construction, so the interval cannot reflect which participant block happens to come first. A random-offset contiguous-block analysis on the cached checkpoints would turn this instance into an estimate for the strategy as a whole. (anchor: table: tab:contrasts, row uniform random vs. sequential, 200; confidence 3, design reading)
- **m5.** The noise reference for round five's +0.20 cannot be compared with it. The reference is a float seed spread of a different architecture (the TCN) on a different cohort (Weinstock), while the gain is a paired INT8 difference on RT-CGM. (anchor: text: Introduction "a fifth of round three's 1.1-point float seed spread on Weinstock held-out"; confidence 4, quantities compared directly)
- **m6.** The bundled artifact breaks RQ1's own rule on its gating cohort: it was calibrated on the Weinstock range, and 2.8% of RT-CGM windows exceed its input range. The paper discloses this and gives a sensitivity check, but never notes the tension with its "cover the deployment input range" rule. (anchor: text: Section 8.5 "That gap includes any cost of input clipping"; confidence 4, read directly)
- **m7.** The delta-anchor champion of rounds one and two was chosen on the same held-out data that their confirmatory contrasts reuse. The variants differ by about 1.2 points, so the selection inflation is small, but it tilts those contrasts toward the champion. (anchor: text: Section 7.8 "so the anchor variant was chosen on held-out data"; confidence 3, magnitude bounded by the variant gap)

### Observations (Non-Defects)

- **The delta experiment partly separates the two accounts of class asymmetry, although Section 9.1 says they are not separated.** Under contiguous selection, the delta model's level channel saturates at 320 mg/dL while its difference channels keep their slope information, and it loses only about 0.5 points. That pattern favors the slope-erasure account over the lost-level-cue account. The evidence is partial, because the ceilings and models differ.
- **There is unreported seed-level validation evidence for the flatten head.** In `protocol_sweep_r5.json`, every flatten-head seed beats every pooled-head seed on validation (lowest flatten 0.5370 against highest pooled 0.5363). This is selection-split evidence and not confirmatory, but it is the one seed-consistent signal for the design, and the paper does not mention it.
- **The paper pre-empts many attacks by conceding them.** The concessions sit in the body and appendices, while the abstract and Conclusion keep the stronger framing, which is where M1 to M3 bite.

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|

#### MAJOR

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|
| M1 | D3 | The sweeps' central lesson assumes directional recall is the right target. The paper asserts this (it is the quantity the bundled models had been selected on) but never argues it, and concedes that pre-registration does not settle whether it was the right endpoint. On the paper's own numbers, the rejected round-one candidate had higher precision in all four directional classes, higher F1 in every class, and the highest held-out macro-F1 in the sweep table (0.425 against 0.342 to 0.368 for every bundled pair). The directional-recall-selected bundled model labels 56% of truly stable held-out windows as movement, over-calls every movement class 1.7 to 5.2 times, and has falling_fast precision of 0.136. Under a criterion that counts precision or error costs, the lesson inverts: the program discarded the better-balanced operating point. Repair: restate the lesson as two endpoints that disagree, settled by a stated preference for recall over precision with its cost shown, or add a threshold-free or matched-operating-point comparison of the round-one candidate and its champion. | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | 4 (internal logic checked against tab:sweeprecall, tab:shippederrors and cached confusion counts) | not norm-dependent | not norm-dependent |
| M2 | D3 | The override rule was written after both decisions it is said to fit, and those decisions apply one diagnosis in opposite directions. Round one was rejected after passing every gate because its gain was an operating-point shift toward stable. Round five was shipped although the paper's own PEDAP analysis calls its change consistent with an operating-point shift rather than better discrimination, and although it lost 1.7 to 2.2 macro-F1 points on every paired split in return for a +0.20 directional gain. Round one was overridden on a metric that was not its endpoint, measured on its primary component. A symmetric rule (override when the round's secondary metric falls on the primary component) would also have overridden round five on RT-CGM macro-F1. The single rule therefore reduces to letting directional recall decide, which makes the pre-registered gates decorative whenever they disagree with it. The read ledger lists a stable-logit offset scan of held-out predictions, the one analysis able to separate operating point from discrimination, but its outcome is not reported. Repair: state the override as a post hoc preference that ranks directional recall first; report the offset scan's outcome; give round five's per-class and precision comparison on RT-CGM, as was done for PEDAP. | text: Section 6.2 "No protocol stated when a gate-passing candidate may be overridden. The two cases that arose fit one rule" and Section 8.5 "consistent with an operating-point shift from the slow to the fast classes rather than better discrimination" | 4 (manuscript statements, plus macro-F1 values verified in accuracy_sweep_r5.json) | not norm-dependent | not norm-dependent |
| M3 | D3 | The ship decision for the bundled artifact rests on a single gated cohort whose label validity the paper itself doubts, and the summaries leave that doubt out. Six facts stand against RT-CGM: (1) the first DiaData audit excluded it for an unrecorded interval; (2) a re-audit readmitted all 451 subjects by a rule the paper concedes could not fail; (3) its three sensor types cannot be assigned to subjects; (4) its collinear-triple excess (0.305 against at most 0.229) fits interpolation or sample-and-hold, which the paper says would distort its rate labels; (5) 348 of its 355 scored subjects contribute windows with readings outside 40 to 400 mg/dL (panel_revision_r1.json); (6) every cohort with cleaner provenance points the other way (Weinstock held-out -0.32 with gates 1 and 2 failing, temporal with gates 1 and 2 failing, PEDAP 0.539 against 0.541). The gated gain itself is +0.20 points on an interval that carries no training-seed variance. Yet the abstract and Conclusion describe RT-CGM only as never-evaluated. Repair: state the label-validity doubt wherever the round-five result is summarized, and either run the five-seed RT-CGM comparison or present the bundling as a choice that is not backed by evidence of improvement. | text: Appendix B "but that rule could not fail" and "which would distort the rate labels of round five's primary cohort" | 4 (appendix statements, plus audit verdicts checked in diadata_audit_r1.json and diadata_audit_r2.json) | not norm-dependent | not norm-dependent |
| M4 | integrity and disclosure (EIC remit, D5 and D6; not scored by DA) | The manuscript credits an external review with an exploratory read of held-out labels (read ledger, commit 3755d55) and with proposing round five's flatten head, schedule and epoch selection, which together make up the design of the artifact now bundled. The commit that published that review records it as produced by a separate agent (academic-research-suite), with an AI model as co-author, and recent revision commits of the manuscript also carry AI co-author trailers. The manuscript does not identify the reviewer as an AI system run by the author and contains no AI-use statement. The word external implies an independent party and lends the round-five hypotheses an independence they do not have. Repair: describe the review's nature and provenance, add an AI-use statement that covers both analysis and writing, and drop the word external. | text: Appendix D "An external review published between rounds four and five" | 4 (commit message read directly; manuscript searched for any AI-use statement) | COPE position statement Authorship and AI tools (13 February 2023, publicationethics.org): authors who use AI tools in writing a manuscript or in collecting and analysing data must disclose, in the Methods or a similar section, how the tool was used and which tool it was. | An AI agent analysed held-out predictions and proposed the design that was later shipped. Under COPE's wording that is AI use in data analysis, yet the Declarations and Methods disclose nothing, and the label external misdescribes where the analysis came from. |

