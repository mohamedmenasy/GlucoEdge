contract_role: da
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: block
trigger: "The central thesis is salvageable but is stated more strongly or broadly than its evidence allows, and repair needs re-analysis or substantive rescoping rather than rewording"
block_class: repairable

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

Calibration status: `NOT_CALIBRATED` (seat-level report; the panel topology is not final).

### Genuine strengths

The calibration study is unusually well instrumented. Protocols are committed with hashes. A provenance audit found and removed context-target overlap (13.6% of test windows) and duplicated validation windows (35.1%). The samplers are label-free. The clip-only and QDQ controls separate range truncation from grid resolution, and deployment cost is accounted for at the byte and device level. I re-derived the core numbers from `aggregate_r1.json` and they hold. The sequential-200 penalty appears at every seed (held-out directional recall 0.337–0.376 against float 0.472–0.509). The temporal component mirrors it (+12.43). All three sensitivity tiers keep it (+10.5 to +13.2). The mechanism arithmetic closes (13.12 + 0.23 = 13.35 points). Rejecting a gate-passing round-one candidate once the authors saw what it optimized, and publishing that, is a mark of integrity. The Limitations section concedes most of the data reuse a skeptic would raise.

### Core thesis as reconstructed

The paper makes five claims.

1. In a compact INT8 CGM trend classifier, calibration-window coverage of the population's input range decides whether quantization damages minority-class recall. The toolchain-default sequential selection costs 13.4 points while accuracy rises.
2. Any randomized selection restores float-level recall, the other knobs add nothing, and the mechanism is input clipping.
3. The fix is free at deployment.
4. The effect reproduces on two further cohorts.
5. Staged sweeps under a one-read discipline then promoted DiaData training and a flatten head.

Claims 1 and 3, and the clipping mechanism in claim 2, survive my attack. Claim 4 and the interpretive half of claim 2 ("the other knobs add nothing") do not survive as stated (M1, M2). Claim 5 holds only for one artifact (M3).

### Strongest Counter-Argument

The opposing case is that the paper measures, with unusual care, a failure the field already knows to avoid. Static post-training quantization is documented to need a representative calibration set. The first 200 windows of a pool sorted by participant and time are not representative, and the 13.4-point loss shows how bad one unrepresentative set can be in one cohort. The factorial looks rich, but the paper's own controls single out one variable, the input quantizer's upper edge, and on it the 200-window cells take two values: about 251 mg/dL for sequential and the 401 mg/dL sensor cap for every randomized draw. The nulls for stratification, budget and observer therefore compare arms the mechanism says must match, and the clip-only control reproduces the loss without a quantized model. On this view the practical content is one check: compare the INT8 input range with the deployment population's tail, or set it from the sensor's reporting range. The 110 artifacts are not needed for that. The external support is thin. In two small cohorts the effect shrinks to 2.0 and 1.3 points, both pre-specified intervals include zero, and the narrow-range cohort's better-powered component shows the effect it was predicted not to show. The sweep half of the Abstract adds gains of 0.2 to 0.6 points, inside the paper's own 1.1-point seed spread, and the shipped round-five model would fail the paper's gates on both Weinstock components. INT8 is never faster than float on mean latency in any reported measurement and saves about 5 KB for the study model, so the quantization trade-off has no measured benefit side. What remains is a transparent engineering case study and a clean demonstration of the accuracy illusion. It is not a general result about calibration coverage.

### D3 judgment

My score is block (repairable). The fatal trigger does not fire, because nothing in the published record contradicts claims 1 and 3 or the clipping mechanism. But three claims are stated more strongly or broadly than the evidence allows: claim 4, the interpretive half of claim 2, and claim 5. Repair needs substantive rescoping of the Abstract, Limitations items 2 and 3, and the Conclusion. It also needs two modest analyses: per-cohort clip shares for M1, and either seed-aware evidence or a dropped attribution for M3. That is more than the localized rewording my warn trigger covers.

### Challenge-level judgments (not totaled or mapped to severity)

- Core thesis: partly meets. The Weinstock core holds; its generality and the meaning of its nulls are overstated (M1, M2).
- Cherry-picking: meets. Costs are often reported more fully than gains. The omissions I found (the budget-by-strategy interaction, and PEDAP's round-five fast-class gains) do not favor the thesis.
- Confirmation bias: partly meets. Predictions are declared to have held on unstated criteria (M1). Seed noise is invoked against the secondaries but not against the primary (M3).
- Logic chain: partly meets. A contrast that is degenerate for the identified mechanism becomes a design rule (M2).
- Overgeneralization: partly meets. Examples: "reproduces on two further cohorts", "nearly calibration-proof", "reduces to one rule".
- Alternative paths: partly meets. Delta encoding was tested; a data-free input range and shipping float were never discussed.
- Stakeholders: I name the absent voices below and do not elaborate.
- So what: partly meets. The accuracy-illusion lesson is useful, but the benefit side of quantization is never measured (see Unexamined Premise).
- Field-norm self-check: M3's severity rests on a norm grounded in a reference the manuscript itself cites. M1 and M2 rest only on the manuscript's own numbers.

### Ignored Alternative Explanations/Paths

1. A data-free input range. The pool spans exactly 39–401 mg/dL (Table 5 caption), and §7.4 names "the 401 sensor cap". Fixing the input tensor's quantization range from the sensor's reporting limits would remove the clipping failure for any calibration selection, and the paper's controls show clipping is essentially the whole loss. Re-encoding (delta inputs) was tested; this simpler option is never discussed.
2. Shipping float. INT8 is never faster than float on mean latency in any reported measurement: 2.78 μs against 3.42–3.47 μs (Table 19), 0.334 against 0.484 ms on Android (Table 20), and 0.124 against 0.126 ms in-app for the round-five pair. INT8 saves 5,376 bytes for the 2,909-parameter model, and both applications already default to float (§5.3).
3. Round five's gain as an operating-point shift. Macro-F1 is lower on the primary cohort (0.410 vs. 0.427). On PEDAP the candidate gains on the fast classes (falling_fast 0.705 vs. 0.638, rising_fast 0.633 vs. 0.611 in accuracy_sweep_r5.json) and loses on the slow ones (falling −6.4, rising −3.1). That pattern looks like probability mass moving among directional classes, not better discrimination. The paper reports only the losses.
4. Slope erasure as the mechanism. Clipping maps every reading above 250.9 mg/dL to one code, which flattens the clipped part of a high fast fall into a plateau. That explains both the falling_fast collapse and the inflated stable predictions without appeal to "how high the trajectory started". It also fits §8.1's own remark that the label depends on differences between readings.
5. Hall's temporal effect follows from the mechanism. Hall's sequential ceiling was 165.0 mg/dL, not the roughly 240 the frozen prediction assumed, so the mechanism predicts a non-zero effect. The "thin tail" account is post hoc and untested, because clip shares are not reported.

### Missing Stakeholder Perspectives

- People who wear CGMs
- Clinicians and diabetes educators who teach trend-arrow use
- App and embedded developers who would adopt the calibration rule
- CGM manufacturers whose arrow conventions anchor the thresholds
- Regulators who separate research software from medical-device software

### Unexamined Premise

The title and research question assume that quantization involves a real trade-off for this model class. The paper's own measurements show no benefit side: INT8's mean latency never beats float's on either phone, and INT8 saves about 5 KB for the study model. No constrained target where INT8 might pay off is measured, the sizes of the shipped round-five pair are not reported, and both applications default to float. The calibration failure is real, but its practical stakes for this application are not shown.

### Minor issues (prose only; no band table)

1. Read discipline misstated. The Abstract's "each judged by one pre-registered read" and the Conclusion's "Each round spent one pre-registered confirmatory read" conflict with round four, where "no confirmatory read was performed". The Abstract also omits that the Weinstock held-out labels were read in rounds one to three (Limitations item 3). Anchor: text: Conclusion, "Each round spent one pre-registered confirmatory read". Confidence 5 (text check).
2. Observer bound too tight. The Conclusion's "by at most eight hundredths of a point" fails for the observer arm on range-coverage-200, draw 101. The five-seed held-out mean is 0.48653 with the histogram observer and 0.48771 with min–max, a 0.118-point difference (aggregate_r1.json; §7.3 prints 0.487 vs. 0.488). Anchor: text: Conclusion, "moved it by at most eight hundredths of a point". Confidence 5 (recomputed).
3. A measurable "nothing". The Introduction's "add nothing measurable" conflicts with Table 7's balanced 1,000 vs. 200 contrast, +0.08 [+0.02, +0.15]. The interval excludes zero, so the effect is tiny but measurable. Anchor: table: Table 7, Balanced 1,000 vs. 200 row. Confidence 5.
4. Participant range misstated. §7.2 says every randomized 200-window set covers 96–127 participants, but selection_coverage records 92 participants for uniform-200 draw 102 and 90 for draw 103. Anchor: dataset: aggregate_r1.json selection_coverage, uniform_random_200_d102 and uniform_random_200_d103. Confidence 5.
5. "Only" overstates. The Abstract's "gained only on the majority class" conflicts with Table 13, where the winner's rising recall is 0.365 against 0.334 (temporal 0.409 vs. 0.399). The body's "almost entirely" is accurate. Anchor: table: Table 13, rising row. Confidence 5.
6. Undefined draws not disclosed. Dubosson's contrast had 1.2% undefined bootstrap draws (undefined_fraction 0.012). Table 11 does not disclose this, while Table 7's caption reports the equivalent for the main study. Anchor: dataset: confirmation_dubosson.json, contrast_uniform200_vs_sequential200 temporal entry. Confidence 5.
7. Pre-registered secondaries unreported. protocol.json lists a budget-by-strategy interaction, and its "1000 vs 200 within each strategy" contrast includes sequential. Neither appears with an interval in aggregate_r1.json or Table 7. Both would favor the thesis, so this is a completeness issue, not selective reporting. Anchor: absence: Table 7 and aggregate_r1.json contrasts — expected budget-by-strategy interaction and sequential 1,000 vs. 200 intervals; checked protocol.json, aggregate_r1.json, §7.3. Confidence 5.
8. Frozen "hypotheses" not on record. §1 and §6 say the frozen protocol includes hypotheses, but protocol.json has no hypothesis or prediction field, unlike the confirmation and delta protocols. The direction expected for the null primary contrast is therefore not on the published record. Anchor: absence: protocol.json — expected a hypotheses or prediction field; checked protocol.json keys, §6, §10.1. Confidence 4 (the plan file it references is outside the supplement).
9. Delta-encoding claim overreaches. The Abstract's "nearly calibration-proof" rests on what the run's own record calls a "post-hoc experiment on previously touched held-out data", tested against one adversarial selection. The anchor variant advanced over the plain variant on held-out recall (0.5088 vs. 0.4969). The ship gate passed by 0.000045, with a float-minus-sequential interval of +0.33 to +0.67 points straddling the 0.5-point threshold (the margin is disclosed). Anchor: dataset: delta_inputs_r1.json, protocol disclosure and ship_gate. Confidence 5.
10. "Headroom" overstated. Round four's binding measurement compares two labels computed from the same future readings (g[t..t+3]). That bounds label-definition noise, not recall attainable from the 12 inputs, so an abort threshold of 2 points could hardly fire (observed headroom 42.1 points in exp0_sweep_r4.json). "places the remaining error in the model and data, not the labels" holds only if "data" covers the missing covariates conceded in Limitations item 6. Anchor: dataset: exp0_sweep_r4.json, estimator and headroom fields. Confidence 4 (logic).
11. "Toolchain-default" unsupported. The conversion toolchain does not choose calibration windows. The first-N selection was the project's own script, and nothing cited shows it to be a default (Abstract, §5.2, §8.1). Anchor: text: §5.2, "the toolchain-default pattern that produced the shipped artifact". Confidence 4.
12. Clinical drift in wording. "the unchanged clinical thresholds" (§7.12) and "a worse clinical-shape classifier" (§8.2) cut against §2.3's framing of the bands as a research target with "no basis for treatment decisions". Anchor: text: §8.2, "a worse clinical-shape classifier". Confidence 5.
13. "Never-read" is inaccurate. A rejected round-three arm trained on RT-CGM labels, as §7.13 discloses, so the Abstract's "a never-read cohort" should say the cohort was never evaluated. Anchor: text: Abstract, "zero-shot on a never-read cohort". Confidence 5.
14. Mechanism statements in tension. §8.1 calls the missing grid cost "consistent with a label that depends on the difference between readings rather than their absolute code". It then concludes "A model that cannot see how high the trajectory started cannot recognize a fast fall" (see alternative 4). Anchor: text: §8.1, "cannot see how high the trajectory started". Confidence 4.

### Observations (Non-Defects)

- The sweep diary answers a different question from the one §1 poses, and it fills about half the Abstract. This is a structural point for the editor-in-chief seat, not scored here.
- Round one shows the gates can be overridden by evidence outside them. Round five was not overridden, although its pre-declared secondaries would fail gates 1 and 2 on both Weinstock components. Both decisions are defensible, but a stated override rule would make them auditable.
- Re-derived and consistent: Tables 6–8 and 10, the round-five gate values, the split-audit percentages (24.4% and 26.3% of windows, 18.5% and 20.5% of readings above 239.9 mg/dL) and the device-benchmark means.
- Process disclosure, not a manuscript finding. This seat's Phase 1 call ran with repository context injected by the harness: the project's CLAUDE.md and a memory index holding one-line audit facts. I opened neither the manuscript nor the repository in Phase 1, and the Phase 1 plan is written from the contract alone. Even so, the synthesizer should treat Phase 1 blindness as imperfect for any seat run in this workspace.

#### CRITICAL
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|---|

#### MAJOR
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|---|
| M1 | D3 | Cross-cohort confirmation is overstated and internally inconsistent. The Abstract says "The effect reproduces on two further cohorts", yet §7.7 pre-registered a near-zero difference for hall and then reports "Both predictions held". Hall cannot both reproduce the effect and confirm its absence. Dubosson's pre-specified temporal contrast is +2.0 [−0.0, +3.1]; its lower bound is −0.0034 points in confirmation_dubosson.json, short of the fully positive interval the paper treats as resolution in §7.11 and §7.13. Hall's primary rests on 6 held-out clusters, while its 50-participant temporal component gives +4.3 [+1.1, +7.3]. The frozen premise of the hall prediction, that sequential selection cannot meaningfully under-cover the population range, failed: the sequential ceiling was 165.0 mg/dL against a 303 maximum. The dose-response sentence quotes Weinstock's ceiling as 239.9, which is the historical artifact's ceiling (Table 4); the 13.4-point contrast used 250.9 (Table 5). With hall's primary (+1.3), the listed gaps (about 150, 114 and 138 mg/dL) do not order the effects (13.4, 2.0, 1.3). With hall's temporal (+4.3) they do, but then the near-zero prediction fails. The paper cannot keep both "tracks the gap" and "near zero as predicted". The quantity the mechanism implies, the share of windows above each ceiling, is not reported for either cohort. Repair: rescope the Abstract, Limitations item 2 and the Conclusion; state what counted as a prediction holding; report per-cohort clip shares before claiming the effect tracks the deficit. | table: Table 11, second-cohort confirmation, dubosson temporal and hall rows | 5 (direct check of confirmation_dubosson.json and confirmation_hall.json against the text) | Not norm-dependent | Severity rests on the manuscript's own statements, tables and result files |
| M2 | D3 | The null primary contrast cannot carry the take-home it is given. All 18 randomized selections span 39–401 mg/dL in aggregate_r1.json selection_coverage and receive the same INT8 input range, 0–400.8 (Table 5). The pre-registered primary (range coverage vs. balanced random at 200) and the randomized budget contrasts therefore compare arms with identical input quantization. The paper's own controls put essentially the whole loss on input clipping (Table 8: float−clip 13.14 of 13.35 points; the INT8 interior is about 0.2). The primary therefore had no input-range difference to detect, and its +0.06 [−0.02, +0.13] says nothing about whether range-aware selection helps. The Conclusion still turns it into a rule ("Coverage means covering participants; the other knobs were the wrong ones"), as does §8.1 ("the refinements evaluated here are exhausted"; "calibration design reduces to one rule"). The sequential-1,000 cell contradicts the participant reading: 6 of 127 participants that reach 401 mg/dL recover fully (uniform−sequential at 1,000 is +0.05 [−0.05, +0.15], Table 7), so the budget did matter for sequential, through range. The operative variable is the selection's maximum relative to the population tail; participant spread is only how random draws reach it. Where random draws fell short of the tail, the range-coverage arm was not run: the uniform ceilings were 352.7 against a 400 maximum in dubosson and 233.9 against 303 in hall. Repair: report the primary as uninformative about input range in this pool; restate the take-home as input-range coverage; and either test range-aware selection where random draws under-cover or scope the nulls to the saturated regime. | table: Table 5, 200-window calibration-set coverage, randomized rows vs. Sequential row | 5 (verified against aggregate_r1.json selection_coverage and contrasts) | Not norm-dependent | Severity rests on the manuscript's own coverage, contrast and mechanism tables and its Conclusion text |
| M3 | D3 | The round-five win sits inside seed noise that the paper invokes against its secondaries but not against its primary, and it is credited to one of three simultaneous changes. The certification compares one candidate seed (4) with one champion seed (round-two seed 1). Its participant-cluster interval (+0.11 to +0.29) is conditional on those two fitted models (§6; Limitations item 4). §7.12 records a 1.1-point five-seed held-out spread that "rivals every effect size the sweeps have chased", 5.5 times the certified +0.20. In accuracy_sweep_r3.json, seed choice alone moves the held-out contrast from +0.05 (the TCN's five-seed INT8 mean, 0.5239 vs. 0.5234) to +0.77 (selected seed 2). §7.13 dismisses the held-out −0.3 and temporal +0.3 as "both inside seed noise" but certifies a smaller +0.2. The candidate also changes the head, the cosine schedule with its kept-epoch rule, and the seed at once, so the Abstract's credit to "a position-preserving flatten head" is confounded. The record shows costs that never reach the Abstract. On both Weinstock components the candidate fails gates 1 and 2 (gate1_beats_champion and gate2_stable_floor are false in accuracy_sweep_r5.json). PEDAP directional recall is 0.539 vs. 0.541. Macro-F1 is lower on the primary cohort itself (0.410 vs. 0.427). Repair: present round five as an artifact-level ship decision; report all five flatten-head seeds against the champion on RT-CGM, or give a seed-aware interval, before crediting the head or the schedule; and carry the costs into the Abstract. | table: Table 18, round-five confirmatory read, RT-CGM row | 4 (checked against accuracy_sweep_r3.json, accuracy_sweep_r5.json and protocol_sweep_r5.json; round-five per-seed RT-CGM results are unpublished, so the size of the seed noise is taken from rounds three and four) | Bouthillier et al. 2021, MLSys 3:747–769, cited in §2.2: model comparisons on a fixed split must account for variance from training seeds and implementation choices | The certified +0.20 is about one fifth of the paper's own 1.1-point seed spread. It comes from one seed per arm with a model-conditional interval, and the paper credits it to an architectural change |
