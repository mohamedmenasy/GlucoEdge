# Editorial Decision Package

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from a candidate or prose-named profile. `PROFILE_MEASURED` remains unavailable until a closed profile artifact and replay validator bind the exact target fields to the completed panel's `execution_topology_sha256` (`c259b3f71e07a5edf63a2840650cfafb0d9380f68f5533b18fdd78f82f6e2603`). Individual seats also remain `NOT_CALIBRATED` at emission; the DA card records this explicitly.

## Sprint-Contract Mechanical Synthesis (v3.6.2)

Contract `reviewer/reviewer_full/v2` (`contract_sha256` e9712090…ef7af). Dispatch roles in order: eic, methodology, domain, perspective, da. All five cards passed the structural conformance gate and the panel layer-1 check (orchestrator).

**Binding precondition (#684).** This is an explicitly unbound run. Each seat's Phase 1 output carries the `criteria_binding_unavailable` disclosure, and no Phase 2 card makes a venue-alignment claim (the EIC card states that no venue criteria are bound). The synthesis therefore makes no venue-alignment claim. Binding conformance is not a score or a failure condition.

**Criterion judgements** (`judgement_scale: sprint_contract`; copied verbatim from the cards; every `not_assessed` here is structural, from an ineligible seat; there are no eligible abstentions):

| Seat (contract_role) | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| EIC (eic) | not_assessed | not_assessed | not_assessed | not_assessed | block | block (repairable) |
| R1 (methodology) | block (repairable) | not_assessed | block (repairable) | not_assessed | not_assessed | not_assessed |
| R2 (domain) | not_assessed | block (repairable) | not_assessed | not_assessed | not_assessed | not_assessed |
| R3 (perspective) | not_assessed | not_assessed | not_assessed | warn | not_assessed | not_assessed |
| DA (da) | not_assessed | not_assessed | block (repairable) | not_assessed | not_assessed | not_assessed |

**Step 1: role-scoped scoring matrix.** Only assessed scores from eligible roles count. The verdict is the worst assessed eligible score.

| Dim | Priority | Eligible roles | Assessed eligible scores (trigger as written by the seat) | Verdict |
|---|---|---|---|---|
| D1 methodology_rigor | mandatory | methodology | R1 block, repairable: "headline effects resting on a single training seed or a single resampling draw with no seed or draw variability shown" | block |
| D2 domain_accuracy | mandatory | domain | R2 block, repairable: "A domain error or omission that changes what a reported result, comparison or contribution claim means but that revision can correct without invalidating the core result" | block |
| D3 argumentative_coherence | mandatory | da, methodology | R1 block, repairable: "non-significance or overlapping intervals asserted as equivalence". DA block, repairable: "The central thesis is salvageable but is stated more strongly or broadly than its evidence allows, and repair needs re-analysis or substantive rescoping rather than rewording" | block |
| D4 cross_disciplinary_relevance | high | perspective | R3 warn: "Localized gaps that leave the central result and its scope correctly understood but make the paper harder to use, or easier to over-read, for one of the two audiences" | warn |
| D5 writing_and_structure | normal | eic | EIC block (no block_class, because D5 is not mandatory): "A headline number in the abstract, conclusions or contribution list contradicts, or cannot be traced to, the table, figure or result file it summarizes beyond rounding at the stated precision" | block |
| D6 venue_fit_and_contribution | mandatory | eic | EIC block, repairable: "the closest prior on-device, quantized or health time-series work is missing so novelty cannot be checked" | block |

**Step 2: failure conditions.** Two-stage evaluation: the cross-reviewer quantifier is applied per dimension over its assessed eligible seats, then the expression's dimension quantifier is applied.

| Condition | Severity | Cross-reviewer quantifier | Expression | Evaluation | Fired |
|---|---|---|---|---|---|
| F1 | 95 | any | any mandatory dimension has a fatal block | D1, D2, D3, D6: no assessed seat declared `fatal` | false |
| F2 | 90 | any | any mandatory dimension scores 'block' | D1 (R1), D2 (R2), D3 (R1, DA), D6 (EIC) each have ≥1 block | true |
| F3 | 70 | majority | two or more mandatory dimensions score 'warn' or worse | D1 n=1, owner seat block; D2 n=1, block; D3 n=2, both seats block; D6 n=1, block. 4 dimensions ≥ 2 | true |
| F4 | 60 | any | any high-priority dimension scores 'block' | D4: R3 warn, not block | false |
| F5 | 40 | any | any dimension scores 'warn' or worse | D1–D6 each have ≥1 seat at warn or worse | true |
| F0 | 10 | all | every dimension scores 'pass' | No dimension has all seats at pass | false |

**Step 3: precedence.** The fired conditions are F2 (90), F3 (70) and F5 (40). F2 has the highest severity, so its action applies. The DA CRITICAL table is empty, so the adjudication list is empty and no consistency marker applies.

dimension_verdicts: [D1=block, D2=block, D3=block, D4=warn, D5=block, D6=block]

fired_conditions: [F2, F3, F5]

da_critical_adjudications: []

editorial_decision=major_revision

The cross-model blind decision check (Step 4b) was not run because `ARS_CROSS_MODEL` is unset, so there is no checkpoint line. Under the sprint-contract boundary, nothing below re-runs or extends this arithmetic.

---

## Part 1: Editorial Decision Letter

### Manuscript Information
- **Title**: GlucoEdge: Calibration Coverage and Quantization Trade-offs in On-Device Glucose Trend Classification
- **Manuscript ID**: none assigned (panel `glucoedge-main-tex-round-1`)
- **Submission Date**: not provided
- **Decision Date**: 2026-10-16
- **Review Round**: 1

Dear Author(s),

Thank you for submitting your manuscript, "GlucoEdge: Calibration Coverage and Quantization Trade-offs in On-Device Glucose Trend Classification", for simulated peer review. No target venue was bound for this review, so this letter makes no claim about fit to any named venue. Five role-separated review seats assessed the manuscript, one of them in the Journal-Fit Reviewer role. Their execution provenance is reported below, axis by axis, and is not reduced to a binary independence claim.

### Review Panel Provenance (#540/#740)

- **Typed artifact**: `review/provenance.json` (scratchpad)
- **Artifact SHA-256**: `3b81956db52a6460189a667d07d026d3b216700cb122d57363e5c6360bfbba4d` (raw bytes)
- **Panel ID**: `glucoedge-main-tex-round-1`
- **Normalized manifest SHA-256**: `5c7a0156a30aa671010d7e92fa8648b6f6c1c4d88f16bade7afc9fd82d0f4917`
- **Execution topology SHA-256**: `c259b3f71e07a5edf63a2840650cfafb0d9380f68f5533b18fdd78f82f6e2603`
- **Fresh-context scope**: `within_panel_attempt_only`. This does not compare retries or prior rounds.
- **Replay**: validated by the dispatching layer, and re-validated by the synthesizer with `review_panel_provenance.py validate` (PASS). This letter renders the artifact's values without inference. The prose here is not the Schema 6 machine carrier.

| Seat | Role ID | Actor type | Context ID | Peer outputs visible | Model family | Provider | Human reviewer ID |
|---|---|---|---|---|---|---|---|
| EIC | eic | model | subagent-seat-eic | false | claude-opus-5-5 | anthropic | null |
| R1 | methodology | model | subagent-seat-r1 | false | claude-opus-5-5 | anthropic | null |
| R2 | domain | model | subagent-seat-r2 | false | claude-opus-5-5 | anthropic | null |
| R3 | perspective | model | subagent-seat-r3 | false | claude-opus-5-5 | anthropic | null |
| DA | da | model | subagent-seat-da | false | claude-opus-5-5 | anthropic | null |

| Provenance axis | Status |
|---|---|
| Role-separated | true |
| Within-panel invocation-context separation | true |
| Blind to peer outputs | true |
| Model-family distinct | false |
| Provider distinct | false |
| Human-reviewer distinct | false |

- **Binary independence claim**: not computed (`independence_claim: not_computed_from_personas`). Persona or role diversity establishes `role_separated` only. The panel is not described as independent.
- **Correlated-error disclosure** (required; reason `same_model_family`): All model-executed review seats used one model family; role separation does not remove correlated-error risk.

### Process Disclosures (orchestrator observations, not findings)

1. **No review-target binding (#684).** Every seat disclosed `criteria_binding_unavailable`, and the panel makes no venue-alignment claim. The Phase 0 candidate venues are for the author's consideration only.
2. **No cross-model track.** `ARS_CROSS_MODEL` is unset. All five seats ran on one model family and one provider (see the provenance above).
3. **Partial Phase 1 paper-blindness.** The harness auto-injected the project's CLAUDE.md and the user's memory index into every seat's context. The orchestrator reports that R1 and the DA disclosed this; the DA's disclosure is also in its card (Observations). A transcript audit found that no seat accessed the manuscript or repository during Phase 1 and no seat opened another seat's card, and all Phase 1 outputs passed the 12-word manuscript-shingle check.
4. **Conformance.** Each seat ran the structural conformance linter on its own draft as a final preflight, and changed only formatting. The orchestrator then re-ran the gate on every final card (all PASS) and ran the panel layer-1 check (PASS).
5. **Sources.** Seats were permitted to consult the versioned supplementary results, git history, and (R2) web sources.
6. **Synthesizer's read-only checks of disputed points.** The synthesizer read the frozen `prediction` fields and contrasts in `confirmation_dubosson.json` and `confirmation_hall.json`, `artifacts_summary.json` and `device_benchmark_ios_r1.json` (timed-artifact count), the `tab:contrasts` source rows against `aggregate_r1.json` contrasts and `selection_coverage`, the `exp0_sweep_r4.json` estimator, the round-five gate flags in `accuracy_sweep_r5.json`, and the results directory listing with `git show --stat c8f3de6`. It also read the abstract, contribution list and conclusion in `paper/main.tex`. Nothing was modified.
7. **Instruction-data boundary.** The EIC and R2 found no reviewer- or editor-directed text in the manuscript or supplementary files, and none appeared in the passages the synthesizer consulted.

### Decision: Major Revision

### Consensus Analysis

Counts run over the four non-DA seats (EIC, R1, R2, R3). Silence (`not-mentioned`) counts as neither agreement nor dissent. DA positions are reported alongside but never counted. Sub-claim (SC) IDs are defined in the Part 2 inventory.

#### Points of Agreement (Consensus)

- **[CONSENSUS-4]**: None. No sub-claim was raised by all four non-DA seats.
- **[CONSENSUS-3]**: **SC-23.** The trend-arrow rate bands are called "the unchanged clinical thresholds" (§7.12) and the sequential artifact "a worse clinical-shape classifier" (§8.2). Both phrases contradict the paper's own framing in §3 and its research-only scope. EIC-W15, R2-W9 and R3-W5 agree, R1 is silent, and DA-m12 concurs. This becomes R13.

**Corroborated findings (2/4, no conflict) and the headline clusters, with corroboration counts** (each cluster was checked against the cards):

| Cluster (SC) | Non-DA seats (of 4) | DA | Seats (of 5) | Disposition | Roadmap |
|---|---|---|---|---|---|
| Novelty not positioned against the PTQ calibration-data literature (SC-2) | EIC-W2, R2-W2 (2) | strongest counter-argument | 3 | Corroborated | R2 |
| "Toolchain-default" framing unsupported (SC-5) | EIC-W4, R2-W1 (2) | m11 | 3 | Corroborated | R5 |
| Round-five read framed as recipe- or design-level, inside seed noise (SC-4) | EIC-W3, R1-W3 (2) | M3 | 3 | Corroborated | R4 |
| Round-five attribution to the head alone (SC-30) | R1-W4 (1) | M3 | 2 | Single (+DA) | R16 |
| Second-cohort "confirmation" (SC-6) | EIC-W5, R1-W1 (2) | M1 | 3 | **SPLIT**, see D2 | R6 |
| Label-noise "ceiling" as a definition (SC-39) | R1-W10, R2-W3 (2) | m10 | 3 | Corroborated | R18 |
| Label-noise conclusion, "error in model and data, not labels" (SC-40) | R1-W10, R2-W3, R3-W3 (3) | m10 | 4 | **SPLIT**, see D4 | R19 |
| "One read per round" contradicted by round four (SC-14) | EIC-W8, R1-W6 (2) | m1 | 3 | Corroborated | R9 |
| 239.9 vs 250.9 mg/dL ceiling in §7.7 (SC-17) | EIC-W10, R1-W8 (2) | M1 | 3 | Corroborated | R11 |
| "At most eight hundredths" bound vs observer 0.12 points (SC-15) | EIC-W9, R1-W8 (2) | m2 | 3 | Corroborated | R10 |
| Clinical-threshold wording (SC-23) | EIC-W15, R2-W9, R3-W5 (3) | m12 | 4 | CONSENSUS-3 | R13 |
| "Gained only on the majority class" (SC-13) | EIC-W7, R1-W9 (2) | m5 | 3 | Corroborated | R8 |
| Delta encoding unlabelled as post-hoc and held-out-selected (SC-12) | EIC-W7, R1-W6 (2) | m9 | 3 | Corroborated | S5 |
| Headline is a prespecified secondary while the primary was null (SC-11) | EIC-W7, R1-W5 (2) | — | 2 | Corroborated | S4 |
| Budget-by-strategy interaction unreported (SC-25) | EIC-W17, R1-W5 (2) | m7 | 3 | Corroborated | S9 |
| Shipped model's INT8 calibration pool unstated (SC-9) | EIC-W6, R1-W11 (2) | — | 2 | Corroborated | S2 |
| Stale device-timing sentence and artifact count (SC-19) | EIC-W11, R1-W7 (2) | — | 2 | Corroborated | S7 |
| Forecasting evaluation described as RMSE-only (SC-42) | R2-W4, R3-W9 (2) | — | 2 | Corroborated | S18 |
| 39/401 mg/dL codes treated as measurements (SC-44) | R2-W6, R3-W7 (2) | — | 2 | Corroborated | S20 |
| RT-CGM cadence audit cannot fail (SC-46) | R2-W8, R3 note N1 (2) | — | 2 | Corroborated | S22 |
| Sweep methods uncited, CORN included (SC-47) | R2-W11, R3-W8 (2) | — | 2 | Corroborated | S23 |

**Agreed strengths** (reported for the rationale; not part of the weakness count):
- The calibration core re-derives from the versioned records (EIC-S5, R1-S7, R2-S1 and S3, and the DA's re-derivations).
- The mechanism controls isolate input clipping (EIC-S2, R1-S3, R2-S3, R3-S4; the DA agrees).
- Analysis status is labelled throughout, and a decision that ran against the author's interest is disclosed (EIC-S1 and S7, R1-S4, R3-S6; the DA agrees).
- The device benchmark is a sound paired protocol (EIC-S4, R1-S6, R2-S4).
- The provenance-masked, participant-disjoint protocol is a benchmark-level contribution (EIC-S8, R1-S1, R2-S5).
- The accuracy illusion is the most transferable lesson (EIC-S3, R3-S3).
- Scope and safety language is consistent (EIC-S9, R3-S1).

#### Points of Disagreement

**D1: Severity of the missing on-device health / on-device glucose prior work (SC-3)**, a SPLIT.
- **EIC-W2 (major, confidence 4)**: no quantized on-device health or wearable time-series classifier is cited. This is the EIC's own D6 trigger.
- **R2-W10 (minor, confidence 4)**: Table 1's only device-evidence row is Li et al. 2020. Zhu et al. (IEEE IoT J 2022) and the OhioT1DM citation are missing, but "no comparison depends on it".
- **Type**: severity disagreement arising from different criteria (D6 positioning vs D2 accuracy).
- **Editor's Resolution**: required (R3), with both severities transported unchanged.
- **Resolution Rationale**: The omission is the stated basis of the D6 block in the owner seat's card, and R2 supplies a concrete, verified missing work. R2's minor rating concerns whether any result changes meaning, and that does not conflict with the omission's bearing on novelty checking. There is no factual conflict.

**D2: Second-cohort confirmation (SC-6)**, a SPLIT on severity and on remedy direction.
- **EIC-W5 (minor, confidence 4)**: "reproduces" overstates the result. EIC's remedy keeps the sentence "the pre-registered predictions held on their designated components" and adds hall's temporal +4.3 points.
- **R1-W1 (major, confidence 5)**, joined by **DA-M1 (major band, confidence 5)**: neither designated interval excludes zero, dubosson's frozen prediction has no decision rule, hall's "near zero" has no margin, and hall's frozen premise failed (165 mg/dL sequential ceiling). On this reading the claim cannot be that the predictions held. R1 asks for each outcome to be stated separately, with the interpretation labelled exploratory.
- **Type**: severity disagreement plus a direction disagreement on the remedy.
- **Editor's Resolution**: R1's remedy is adopted (R6, required). EIC's added hall-temporal sentence is kept. Transported severities: EIC minor, R1 major.
- **Resolution Rationale**: Evidence first. The frozen records read "…is POSITIVE…" (`confirmation_dubosson.json`) and "…is NEAR ZERO…" (`confirmation_hall.json`) and specify no decision rule or margin. The designated intervals have lower bounds of −0.0000338 (dubosson temporal) and −0.0014 (hall held-out). Whether a prediction "held" therefore cannot be judged from the pre-registration, and EIC's proposed sentence would keep that ambiguity. Expertise first. Inferential status is a methodology question in R1's remit, and the claim sits in the abstract, Limitations item 2 and the conclusion. R2-W2's in-passing listing of "the cohort replication" as new is not a position on this sub-claim; the novelty statement (R2) follows this arbitration.

**D3: Severity of the undescribed cohorts behind the shipped model (SC-8)**, a SPLIT.
- **R3-W2 (major, confidence 5)**: the five DiaData training cohorts and the RT-CGM confirmatory cohort are never characterized, although `diadata_audit_r1.json` holds the information.
- **R2-W8 (minor, confidence 4)**: RT-CGM's devices and era are undescribed, but the paired contrast stays internally valid.
- **EIC-W6 (minor, confidence 4)**: the training data of the shipped model can only be inferred from the sweep sections.
- **Type**: severity disagreement (perspective difference).
- **Editor's Resolution**: required (R7), with severities transported unchanged.
- **Resolution Rationale**: Three of the four non-DA seats independently find the cohort behind an abstract-level claim ("a never-read cohort (RT-CGM, …)") undescribed. R2's minor rating addresses internal validity, which R3 does not dispute. R3's major rating addresses whether the abstract claim can be interpreted. Both hold. The fix is rewriting, not new analysis (R3).

**D4: Label-noise conclusion (SC-40)**, a SPLIT on severity and remedy scope.
- **R1-W10 (minor, confidence 3) and R2-W3 (minor, confidence 5)**: both labels are computed from the same future readings, so the measurement checks label-definition stability and bounds no achievable recall. R2 adds that the perturbation omits lag, drift and autocorrelated error.
- **R3-W3 (major, confidence 3)**: the conclusion overreaches the single σ white-noise model. R3's minimum remedy qualifies the sentence to "under σ = 2 mg/dL white noise"; its stronger remedy reruns the measurement with colored noise by sensor generation.
- **Type**: severity disagreement plus partial remedy incompatibility. R3's minimum remedy alone would keep the "ceiling" framing that R1 and R2 reject.
- **Editor's Resolution**: required (R19). The remedy rests on R1 and R2's definitional reading (R18 is also required), and R3's noise-model qualifier applies to any label-robustness sentence that remains. The colored-noise rerun is optional. Transported severities: minor, minor, major.
- **Resolution Rationale**: `exp0_sweep_r4.json` defines both labels from g[t..t+3], which supports the definitional reading. The claim appears in the contribution list ("a measured label-noise ceiling") and the conclusion, and three non-DA seats plus the DA (m10) agree that it overreaches.

**D5: Mechanism narrative (SC-62)**, a disagreement between a DA finding and another seat, outside the four-seat count.
- **R3-S4** credits §8.1's account that fast falls start high.
- **DA-m14 (minor band, confidence 4) and DA-A4** find that account in tension with §8.1's own difference-based remark and propose slope erasure: clipping flattens the high part of a fast fall.
- **Type**: existence disagreement over the explanatory narrative. Both sides accept clipping as the mechanism (Table 8).
- **Editor's Resolution**: **unresolved dissent**. The panel does not decide between the two accounts. The author must reconcile the two §8.1 statements and either test the competing account or present both (S36, should_fix).
- **Resolution Rationale**: Neither the cards nor the record discriminate between the accounts without new analysis.

**D6: "Add nothing measurable" (SC-63)**, a disagreement between a DA finding and another seat.
- **DA-m3 (minor band, confidence 5)**: the balanced 1,000 vs 200 contrast is +0.08 [+0.02, +0.15] points, so the effect is tiny but measurable.
- **R1's statistical-reporting judgement**: "The tight main-study intervals do justify 'nothing measurable'".
- **Type**: existence disagreement.
- **Editor's Resolution**: both points stand. Replace the phrase with a quantitative bound consistent with R10 (S37, should_fix).
- **Resolution Rationale**: In `aggregate_r1.json`, `participant_balanced_random_1000_vs_200` has ci95 [0.00016, 0.00150], which excludes zero, so the literal wording fails. R1's point that the effect is practically negligible also holds.

**D7: What the null contrasts show (SC-67)**, a disagreement between a DA finding and another seat.
- **EIC's significance paragraph** lists "the nulls for stratification, budget and observer in this model class" among the genuinely new results.
- **DA-M2 (major band, confidence 5)**: every randomized 200-window selection receives the same input range, so the pre-registered primary had no input-range difference to detect. R1-W4 corroborates the input-range reading through the sequential-1,000 row.
- **Type**: existence and perspective disagreement.
- **Editor's Resolution**: DA-M2 is upheld (R20, required as a D3 block basis). The novelty statement (R2) presents these nulls as scoped to the saturated regime.
- **Resolution Rationale**: In `aggregate_r1.json` `selection_coverage`, every randomized 200-window selection spans 39–401 mg/dL, while sequential spans 39–251 mg/dL. Table 7's sequential-1,000 row shows recovery through range.

**Factual discrepancies resolved on the record** (these are not disagreements over existence or severity):
- **Timed-artifact count (EIC-W11 vs R1-W7).** `artifacts_summary.json` lists 110 INT8 and 5 float artifacts, and `device_benchmark_ios_r1.json` times 9 artifacts (1 float and 8 INT8). So 8 of the 110 INT8 artifacts were timed and 102 were not. R1's count matches the record; EIC's "110 minus 9" does not. Both seats' requirement is unchanged (S7).
- **Round-four record (EIC-W18).** `accuracy_sweep_r4.json` is versioned in `paper/results/calibration-coverage/`. What is absent there is a frozen round-four protocol file. R1-S2 dates the round-four abort rule to commit c8f3de6 (a design spec), committed before the result. EIC-W18's request narrows accordingly (S10).

#### Devil's Advocate Findings

- **DA-CRITICAL**: none. The CRITICAL table is empty, so there are no adjudications and no rejection rationales.
- **DA-M1** (second-cohort confirmation) is corroborated by R1-W1 and EIC-W5/W10. It drives R6, R11 and R12.
- **DA-M2** (null primary) is corroborated in part by R1-W4 (R17). The remainder is DA-only and is upheld under D7 (R20).
- **DA-M3** (round five) is corroborated by EIC-W3 and R1-W3/W4 (R4, R16). The cost sub-claim is DA-only (R21).
- **DA prose items** (minor issues m1–m14, alternatives A1–A5, unexamined premise, observations) are mapped to roadmap items in Part 2. None is dismissed.

### Decision Rationale

Under contract `reviewer/reviewer_full/v2` the decision is mechanical, and it is not softened here. Four mandatory dimensions are scored block by their eligible seats: D1 by R1, D2 by R2, D3 by R1 and the DA, and D6 by the EIC. F2 therefore fires (severity 90, Major Revision). F3 and F5 also fire, at lower severity. F1 does not fire, because every mandatory block is declared repairable and no seat declared a fatal block.

The narrative record supports this outcome on the merits. No seat disputes the calibration core:
- R1 re-derived the 13.4-point contrast and found per-seed effects of 13.05–14.14 points.
- The DA re-derived the per-seed penalty, the three sensitivity tiers and the mechanism arithmetic (13.12 + 0.23 = 13.35).
- R2 found the quantizer arithmetic exact.
- EIC, R1 and R2 each matched headline numbers to the versioned records.

What fails is the framing built around that core:
- the novelty and "toolchain-default" positioning (EIC, R2, DA);
- the second-cohort confirmation (EIC, R1, DA);
- the round-five recipe-level claim (EIC, R1, DA);
- the label-noise "ceiling" (R1, R2, R3, DA);
- several contradictions in the abstract and conclusion.

Minor Revision is unavailable under the contract once a mandatory dimension blocks. It would also misdescribe the work. The Required items include restructuring the contribution, re-analysing the six-cluster intervals and the clip shares, and either five-seed round-five evidence or restating round five at artifact level. That work warrants re-review.

Reject is not supported. Every block is repairable, every repair uses existing data, and the EIC judges that the calibration study, the provenance-masked protocol and the open dual-platform artifacts would survive any reasonable revision. The arbitrations (D1–D7) set obligation classes and remedies only; they do not change the mechanical decision.

### Blocking Issues (0–3, immutable source order)

These are the three sub-claims cited as a block basis by the most seat–dimension pairs: R4 by R1 (D1 basis, D3 support), the DA (D3) and the EIC (D6 support); R6 by R1 (D3), the DA (D3) and the EIC (D6 support); R2 by the EIC (D6, primary basis) and R2 (D2). R5 ("toolchain-default"; R2 D2 basis, EIC D6 support) ties R2 on that count. It is left out because R2 is the EIC's named primary D6 basis. All 21 Required items remain must_fix. This digest is not a work order.

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| R2 | Novelty is not positioned against the PTQ calibration-data literature (primary D6 basis; D2 co-basis) | EIC, R2 (+DA counter-argument) | absence: §2 Related Work and §8.1 — expected engagement with studies of PTQ calibration-data selection and with quantized on-device health or wearable time-series classifiers; checked §2.1 to §2.4, §8.1, and all 36 entries of the rendered reference list | REV-02 |
| R4 | The round-five read is framed as recipe-level confirmation, although it comes from a single-seed, two-artifact contrast smaller than the seed spread | EIC, R1, DA | text: §1 contribution list "a position-preserving flatten head confirmed zero-shot on a never-read cohort" versus §7.12 "the five-seed held-out spread (1.1 points) rivals every effect size the sweeps have chased" | REV-04 |
| R6 | The second-cohort "confirmation" is stated beyond what the frozen predictions and intervals allow | EIC, R1, DA | text: Abstract "The effect reproduces on two further cohorts"; §Second-cohort confirmation "Both predictions held, and the magnitudes line up with the mechanism" | REV-06 |

We encourage you to consider the reviewers' comments carefully and to submit a substantially revised manuscript. The revised manuscript will undergo another round of review.

**Journal-supplied deadline**: NOT PROVIDED. No deadline, duration or work estimate is inferred.

---

## Part 2: Revision Roadmap

**Conventions.**
- **Order.** Items follow immutable primary-source order: seat order EIC, R1, R2, R3, DA, then finding ordinal, sub-claim ordinal, then channel. Severity, obligation and cost never enter the order. `R<n>` and `S<n>` are transport references, not ranks.
- **Source positions.** Findings are `<seat>-W<n>[.<sub>]` (channel `finding`). R3's questions use channel `question` and its notes for other seats use channel `editorial`. DA positions follow the card's order of appearance: strongest counter-argument (CA), alternatives A1–A5, stakeholders, unexamined premise (UP), minor issues m1–m14, observations O1–O4 (channel `editorial`), then MAJOR M1–M3.
- **Obligation rule.** An item is `must_fix` if any of these holds: (a) the sub-claim is part of a finding that the scoring seat names as the basis or support of a block score; (b) it reached CONSENSUS-3/4; (c) it is an arbitrated SPLIT whose problem is upheld; (d) at least two non-DA seats find, without conflict, that a statement in the abstract, contribution list or conclusion misstates the paper's own tables or versioned records. Otherwise anchored findings with a concrete fix are `should_fix`. Optional extensions, questions and non-defect observations are `consider`.
- **Transported metadata.** Severity and confidence are transported from each source and never re-derived. Untagged sources are marked `[SEVERITY-SOURCE: letter-fallback]` or `[CONFIDENCE-SOURCE: none stated]`. The Evidence Anchor column shows the primary source's typed anchor, unless a corroborating source's anchor is more specific; the source is then named in brackets.
- **Machine carrier.** The closed `revision-roadmap/1.0` core cannot be emitted from this run. It needs `base_draft_sha256`, `block_manifest_sha256` and exact `proposed_targets` block IDs from a bound block manifest, and none is bound. The rows below carry every reviewer-owned field the core requires. Cost-scope locators name sections until academic-paper revision mode anchorizes the draft.
- **consensus_level mapping.** CONSENSUS-3 maps to CONSENSUS-3 and SPLIT to SPLIT. Corroborated (2/4), single-reviewer and DA-only items map to SINGLE-VERIFIER, with the other seats carried in `corroborating_sources`; the closed enum has no 2/4 value. Question and editorial items carry `source_kind` and no severity, anchor or confidence.
- **Not emitted.** No author triage, work order or claim-strength authorization is emitted.

**Obligation counts**: must_fix 21, should_fix 37, consider 2 (60 items covering 68 sub-claims).

### Sub-claim inventory (Step 1b, condensed)

Cell notation is `finding · severity · confidence`. "—" means not mentioned (silence). "(disputed)" marks the conflicting position in a SPLIT. DA cells are reported but not counted.

| SC | Sub-claim | EIC | R1 | R2 | R3 | DA | Disposition |
|---|---|---|---|---|---|---|---|
| SC-1 | Two studies in one manuscript; no single takeaway | W1·maj·5 | — | — | — | O1 (non-defect) | Single 1/4 |
| SC-2 | Novelty not positioned against the PTQ calibration-data literature | W2.1·maj·4 | — | W2·maj·4 | — | CA | Corroborated 2/4 |
| SC-3 | Closest quantized on-device health and on-device glucose work missing | W2.2·maj·4 | — | W10·min·4 (disputed: severity) | — | — | SPLIT, D1 |
| SC-4 | Round-five read framed as recipe/design-level; below the seed spread | W3·maj·4 | W3·maj·5 | — | — | M3.1·maj·4 | Corroborated 2/4 |
| SC-5 | "Toolchain-default" label unsupported and contrary to cited guidance | W4·maj·3 | — | W1·maj·5 | — (Q5 echoes the phrase; not a position) | m11·min·4 | Corroborated 2/4 |
| SC-6 | "Reproduces" / "Both predictions held" without a decision rule or margin; hall premise contradicted | W5·min·4 (disputed: severity, remedy) | W1.1–5·maj·5 | — (W2 lists "the cohort replication"; not a position) | — | M1.1·maj·5 | SPLIT, D2 |
| SC-7 | Methods, Figure 1 and Table 1 describe a non-shipping model; input encoding unstated | W6.1, W6.3·min·4 | — | — | — | — | Single 1/4 |
| SC-8 | Training cohorts and RT-CGM cohort of the shipped model uncharacterized | W6.2·min·4 | — | W8.1·min·4 | W2.1–2·maj·5 (disputed: severity) | — | SPLIT, D3 |
| SC-9 | Shipped model's INT8 calibration pool unstated | W6.4·min·4 | W11.3·min·4 | — | — | — | Corroborated 2/4 |
| SC-10 | Abstract density and length | W7.1·min·4 | — | — | — | — | Single 1/4 |
| SC-11 | 13.4-point headline is a prespecified secondary; null primary not stated | W7.2·min·4 | W5.4·min·5 | — | — | — | Corroborated 2/4 |
| SC-12 | Delta encoding is post-hoc and its variant held-out-selected; "nearly calibration-proof" | W7.3·min·4 | W6.1·min·4 | — | — | m9·min·5 | Corroborated 2/4 |
| SC-13 | "Gained only on the majority class" contradicted by Table 13 | W7.4·min·4 | W9.1·min·4 | — | — | m5·min·5 | Corroborated 2/4 |
| SC-14 | "Each round one read" contradicted by round four | W8·min·5 | W6.3·min·4 | — | — | m1.1·min·5 | Corroborated 2/4 |
| SC-15 | "At most eight hundredths" contradicted by observer 0.12 points | W9.1·min·5 | W8.2·min·5 | — | — | m2·min·5 | Corroborated 2/4 |
| SC-16 | "The same 200 windows" wording | W9.2·min·5 | — | — | — | — | Single 1/4 |
| SC-17 | §7.7 pairs the historical 239.9 ceiling with the corrected effect | W10.1·min·5 | W8.1·min·5 | — | — | M1.2·maj·5 | Corroborated 2/4 |
| SC-18 | mg/dL gaps do not order cohorts; per-cohort clipped-window share unreported | W10.2·min·5 | — | — | — | M1.3·maj·5; A5 | Single 1/4 (+DA) |
| SC-19 | Stale device-timing sentence and artifact count | W11·min·5 | W7.1–2·min·5 | — | — | — | Corroborated 2/4 |
| SC-20 | Revision history narrated in the main text | W12·min·4 | — | — | — | — | Single 1/4 |
| SC-21 | Tables below reading size; sweep tables unmerged | W13·min·4 | — | — | — | — | Single 1/4 |
| SC-22 | Figure 3 legend drawn over the bars | W14·min·4 | — | — | — | — | Single 1/4 |
| SC-23 | "Clinical thresholds" / "clinical-shape classifier" wording | W15·min·4 | — | W9·min·5 | W5.1·min·4 | m12·min·5 | CONSENSUS-3 (R1 silent) |
| SC-24 | Sweep reference conventions; "Macro" column label | W16·min·4 | — | — | — | — | Single 1/4 |
| SC-25 | Frozen budget-by-strategy interaction unreported | W17·min·4 | W5.2·min·5 | — | — | m7.1·min·5 | Corroborated 2/4 |
| SC-26 | recency_collision.json and round-four frozen design not versioned | W18·min·4 | — (S2 dates the r4 abort rule to c8f3de6) | — | — | — | Single 1/4 (narrowed) |
| SC-27 | Licence of distributed weights; ethics statement form | W19·min·3 | — | — | — | — | Single 1/4 |
| SC-28 | Undefined bootstrap draws and per-class support undisclosed | — | W1.6·maj·5 | — | — | m6·min·5 | Single 1/4 (+DA) |
| SC-29 | Percentile participant bootstrap at six clusters | — | W2·maj·4 | — | — | — | Single 1/4 |
| SC-30 | Round-five credit to one of several co-varying factors | — | W4.1·min·4 | — | — | M3.2·maj·4 | Single 1/4 (+DA) |
| SC-31 | "Coverage means covering participants" vs the input-range mechanism | — | W4.2·min·4 | — | — | M2.1·maj·5 | Single 1/4 (+DA) |
| SC-32 | Seed and draw variation unreported despite the frozen plan | — | W5.1·min·5 | — | — | — | Single 1/4 |
| SC-33 | Sequential 1,000 vs 200 contrast without an interval | — | W5.3·min·5 | — | — | m7.2·min·5 | Single 1/4 (+DA) |
| SC-34 | Frozen hypotheses H1–H4 not stated | — | W5.5·min·5 | — | — | m8·min·4 | Single 1/4 (+DA) |
| SC-35 | Held-out reuse under-counted; no read ledger; round one decided post hoc | — | W6.2·min·4 | — | — | m1.2·min·5 | Single 1/4 (+DA) |
| SC-36 | Operator inventory and size published only for the 110 study artifacts | — | W7.3·min·5 | — | — | — | Single 1/4 |
| SC-37 | Spliced sensitivity-tier interval | — | W8.3·min·5 | — | — | — | Single 1/4 |
| SC-38 | Macro-F1 gain decomposed with recall | — | W9.2·min·4 | — | — | — | Single 1/4 |
| SC-39 | "Ceiling" and "headroom" misdescribe label-definition agreement | — | W10.1·min·3 | W3.1·min·5 | — | m10.1·min·4 | Corroborated 2/4 |
| SC-40 | "Remaining error in model and data, not labels" beyond the tested noise model | — | W10.2·min·3 | W3.2·min·5 | W3·maj·3 (disputed: severity, remedy scope) | m10.2·min·4 | SPLIT, D4 |
| SC-41 | Recomputation depends on local-only artifacts; sweep batch size unstated | — | W11.1–2·min·4 | — | — | — | Single 1/4 |
| SC-42 | Forecasting evaluation described as RMSE-only | — | — | W4·min·4 | W9·min·4 | — | Corroborated 2/4 |
| SC-43 | Kistkins et al. results misquoted | — | — | W5·min·5 | — | — | Single 1/4 |
| SC-44 | 39/401 mg/dL codes treated as measurements, not censoring | — | — | W6·min·4 | W7·min·4 (+Q2) | — | Corroborated 2/4 |
| SC-45 | What calibration changes inside the INT8 artifact misstated | — | — | W7·min·4 | — | — | Single 1/4 |
| SC-46 | RT-CGM cadence audit cannot fail; 96 subjects without windows; collinear fraction | — | — | W8.2·min·4 | N1.2 (note; untagged) | — | Corroborated 2/4 |
| SC-47 | Sweep methods uncited, including CORN | — | — | W11.1·min·5 | W8.2·min·4 | — | Corroborated 2/4 |
| SC-48 | Soup collapse and regression mean-seeking are known results | — | — | W11.2·min·5 | — | — | Single 1/4 |
| SC-49 | Fixed input range from the sensor's reportable range not considered | — | — | W12·min·4 | — | A1 | Single 1/4 (+DA) |
| SC-50 | Absent resolution penalty is predicted by 8-bit error analysis | — | — | W13·min·4 | — | — | Single 1/4 |
| SC-51 | "6.4× finer" omits the delta model's level channel | — | — | W14·min·5 | — | — | Single 1/4 |
| SC-52 | Practical meaning of the shipped model's errors untranslated | — | — | — | W1·maj·4; N1.1 | stakeholders | Single 1/4 (+DA) |
| SC-53 | Questions Q1, Q4, Q5, Q6 | — | — | — | Q1, Q4, Q5, Q6 | — | Question channel |
| SC-54 | Wear and therapy regimes, RT-CGM naming, PEDAP population, unrepresented groups | — | — | — | W2.3–6·maj·5 | — | Single 1/4 |
| SC-55 | "Calibration" not distinguished from CGM sensor calibration | — | — | — | W4·min·5 | — | Single 1/4 |
| SC-56 | "Rate-of-change classes 15 minutes ahead" ambiguity | — | — | — | W5.2·min·4 | — | Single 1/4 |
| SC-57 | Weinstock case-control enrichment and blinded wear | — | — | — | W6·min·4 | — | Single 1/4 |
| SC-58 | Systems terms undefined; recall not mapped to sensitivity | — | — | — | W8.1·min·4 | — | Single 1/4 |
| SC-59 | Developers as stakeholders; model card; "ships" wording | — | — | — | W10·min·3 | stakeholders | Single 1/4 (+DA) |
| SC-60 | No measured benefit side to INT8; shipping float not discussed | — | — | — (S4 reports INT8 slower; not a position) | — | A2; UP | DA only |
| SC-61 | Round-five gain as an operating-point shift; PEDAP fast-class gains unreported | — | — | — | — | A3 | DA only |
| SC-62 | §8.1 mechanism statements in tension; slope erasure | — | — | — | — (S4 credits the start-high account) | A4; m14·min·4 | DA only; dissent D5 |
| SC-63 | "Add nothing measurable" vs a budget interval excluding zero | — | — (reporting judgement contrary) | — | — | m3·min·5 | DA only; dissent D6 |
| SC-64 | Randomized participant range misstated (96–127 vs 92 and 90) | — | — | — | — | m4·min·5 | DA only |
| SC-65 | "Never-read" cohort inaccurate | — | — | — | — | m13·min·5 | DA only |
| SC-66 | No stated gate-override rule | — | — | — | — | O2 (non-defect) | DA only |
| SC-67 | Null primary uninformative about input range; nulls unscoped | — (novelty list counts the nulls as new) | — | — | — | M2.2·maj·5 | DA only; dissent D7 |
| SC-68 | Round-five costs not carried into the abstract | — | — | — | — | M3.3·maj·4 | DA only |

### Required Revisions (Must Fix)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| R1 | REV-01: state one contribution, or two explicit research questions with matching structure | SC-1 | major | text: §1 "With calibration settled, the work then asks which model and data changes improve directional recall" | 5 (editorial structure judgment) | EIC-W1; DA-O1 | must_fix | section: Title, Abstract, §1, §7.9–§7.13, §8, §11 | interpretive_ambiguity_remains, manuscript: single takeaway |
| R2 | REV-02: position the novelty against the PTQ calibration-data-selection literature | SC-2 | EIC major; R2 major | absence: §2 Related Work and §8.1 — expected engagement with studies of PTQ calibration-data selection and with quantized on-device health or wearable time-series classifiers; checked §2.1 to §2.4, §8.1, and all 36 entries of the rendered reference list | EIC 4; R2 4 | EIC-W2; R2-W2; DA-CA | must_fix | section: §2.4, §8.1, §1 novelty statement | claim_scope_unsupported, claim: §1/§8.1 novelty statement |
| R3 | REV-03: cite the closest quantized on-device health and on-device glucose work | SC-3 | EIC major / R2 minor (SPLIT, D1) | table: tab:relatedwork (Table 1, Device evidence column) [R2-W10] | EIC 4; R2 4 | EIC-W2; R2-W10 | must_fix | section: §2, Table 1 | evidence_gap_remains, table: Table 1 |
| R4 | REV-04: present round five as an artifact-level ship decision, or report recipe-level multi-seed evidence | SC-4 | EIC major; R1 major; DA major | text: §1 contribution list "a position-preserving flatten head confirmed zero-shot on a never-read cohort" versus §7.12 "the five-seed held-out spread (1.1 points) rivals every effect size the sweeps have chased" | EIC 4; R1 5; DA 4 | EIC-W3; R1-W3; DA-M3 | must_fix | re_analysis: five-seed RT-CGM evaluation from cached checkpoints, or sentence: Abstract, §1, §11 | claim_scope_unsupported, claim: round-five design-level claim |
| R5 | REV-05: replace the "toolchain-default" label and quote the cited guidance accurately | SC-5 | EIC major; R2 major; DA minor | text: §5.2 "the toolchain-default pattern that produced the shipped artifact" | EIC 3; R2 5; DA 4 | EIC-W4; R2-W1; DA-m11 | must_fix | sentence: Abstract, §1, §5.2, §7.8, §8.1, §11; quotation in §2.4 | claim_scope_unsupported, claim: toolchain-default premise |
| R6 | REV-06: report each frozen second-cohort prediction's outcome; remove "reproduces" and "held" | SC-6 | EIC minor / R1 major / DA major (SPLIT, D2) | text: Abstract "The effect reproduces on two further cohorts"; §Second-cohort confirmation "Both predictions held, and the magnitudes line up with the mechanism" [R1-W1] | EIC 4; R1 5; DA 5 | EIC-W5; R1-W1; DA-M1 | must_fix | section: §7.7, Table 11, Abstract, Limitations item 2, §11 | claim_scope_unsupported, claim: two-cohort confirmation |
| R7 | REV-08: characterize the training and RT-CGM cohorts behind the shipped model | SC-8 | EIC minor / R2 minor / R3 major (SPLIT, D3) | absence: §7.10 (round two), §7.11 (round three) and §7.13 (round five) — expected a characterization of the five DiaData training cohorts, the RT-CGM confirmatory cohort and the PEDAP probe; checked Abstract, §1, §2.2, §4.1, §7.7, §7.10, §7.11, §7.13, §9, §10.2 and every table caption [R3-W2] | EIC 4; R2 4; R3 5 | EIC-W6; R2-W8; R3-W2 | must_fix | section: new cohort table; §7.11; §10.2 | reader_traceability_reduced, section: §7.10–§7.13 |
| R8 | REV-13: correct "gained only on the majority class" | SC-13 | EIC minor; R1 minor; DA minor | text: Abstract "gained only on the majority class and was rejected" [R1-W9] | EIC 4; R1 4; DA 5 | EIC-W7; R1-W9; DA-m5 | must_fix | sentence: Abstract; §1 contribution list | claim_scope_unsupported, claim: abstract round-one sentence |
| R9 | REV-14: correct the one-read-per-round statement | SC-14 | EIC minor; R1 minor; DA minor | text: §11 Conclusion "Each round spent one pre-registered confirmatory read" | EIC 5; R1 4; DA 5 | EIC-W8; R1-W6; DA-m1 | must_fix | sentence: Abstract; §1; §11 | claim_scope_unsupported, claim: read-discipline statement |
| R10 | REV-15: correct the "eight hundredths" bound | SC-15 | EIC minor; R1 minor; DA minor | text: §11 Conclusion "stratification, budget, and observer choice moved it by at most eight hundredths of a point" | EIC 5; R1 5; DA 5 | EIC-W9; R1-W8; DA-m2 | must_fix | sentence: §11 | claim_scope_unsupported, claim: §11 bound |
| R11 | REV-17: use the corrected-protocol ceiling (250.9 mg/dL) in §7.7 | SC-17 | EIC minor; R1 minor; DA major | text: §7.7 "Weinstock's ceiling of 239.9 against a 401 maximum cost 13.4 points" | EIC 5; R1 5; DA 5 | EIC-W10; R1-W8; DA-M1 | must_fix | sentence: §7.7 | claim_scope_unsupported, section: §7.7 |
| R12 | REV-18: report per-cohort clipped-window shares and rebase the dose-response sentence on them | SC-18 | EIC minor; DA major | table: Table 11, second-cohort confirmation, dubosson temporal and hall rows [DA-M1] | EIC 5; DA 5 | EIC-W10; DA-A5; DA-M1 | must_fix | re_analysis: clipped-window share per cohort and component from cached windows; §7.7 | evidence_gap_remains, section: §7.7 dose-response argument |
| R13 | REV-21: neutral rate-band wording | SC-23 | EIC minor; R2 minor; R3 minor; DA minor | text: §8.2 "a worse clinical-shape classifier with better accuracy" | EIC 4; R2 5; R3 4; DA 5 | EIC-W15; R2-W9; R3-W5; DA-m12 | must_fix | sentence: §7.12, §8.2 | claim_scope_unsupported, claim: research-only scope wording |
| R14 | REV-25: disclose undefined-draw fractions and per-class support in Table 11 | SC-28 | R1 major; DA minor | dataset: confirmation_dubosson.json, contrast_uniform200_vs_sequential200 temporal entry [DA-m6] | R1 5; DA 5 | R1-W1; DA-m6 | must_fix | sentence: Table 11 (values exist in the versioned records) | reporting_requirement_unmet, table: Table 11 |
| R15 | REV-26: small-cluster-valid inference, or no intervals, for six-cluster cohorts | SC-29 | major | text: §Second-cohort confirmation "its frozen protocol assigns the contrast to the temporal component (six clusters)" | 4 (undercoverage well established; not re-simulated) | R1-W2 | must_fix | re_analysis: dubosson temporal and hall held-out intervals | evidence_gap_remains, table: Table 11 intervals |
| R16 | REV-27: attribute round five to the bundled candidate | SC-30 | R1 minor; DA major | text: Abstract "a position-preserving flatten head passed all gates zero-shot on a never-read cohort" | R1 4; DA 4 | R1-W4; DA-M3 | must_fix | sentence: Abstract; §7.13; §11 | claim_scope_unsupported, claim: round-five attribution |
| R17 | REV-28: state the take-home as input-range coverage | SC-31 | R1 minor; DA major | text: Conclusion "Coverage means covering participants; the other knobs were the wrong ones." | R1 4; DA 5 | R1-W4; DA-M2 | must_fix | sentence: §8.1; §11 | claim_scope_unsupported, claim: §11 take-home |
| R18 | REV-34: rename the round-four quantity; drop "ceiling" and "headroom" | SC-39 | R1 minor; R2 minor; DA minor | text: §Accuracy sweep round four "the task is model- and data-limited, not label-noise-limited" | R1 3; R2 5; DA 4 | R1-W10; R2-W3; DA-m10 | must_fix | sentence: §1 contribution list; §7.12; §11 | claim_scope_unsupported, claim: "measured label-noise ceiling" |
| R19 | REV-35: limit the label-noise conclusion to the tested noise model | SC-40 | R1 minor / R2 minor / R3 major (SPLIT, D4) | text: §11 "measured label-noise ceiling far above the champion places the remaining" [R3-W3] | R1 3; R2 5; R3 3 | R1-W10; R2-W3; R3-W3; DA-m10 | must_fix | sentence: §7.12; §11 (optional re_analysis: colored-noise rerun) | claim_scope_unsupported, claim: §11 label-noise conclusion |
| R20 | REV-59: scope the null contrasts to the saturated input-range regime | SC-67 | major (DA) | table: Table 5, 200-window calibration-set coverage, randomized rows vs. Sequential row | 5 (verified against aggregate_r1.json selection_coverage and contrasts) | DA-M2 | must_fix | section: §7.3, §8.1, §11 (or re_analysis: range-aware selection where random draws under-cover) | claim_scope_unsupported, claim: §8.1 design rule |
| R21 | REV-60: report round-five costs wherever the result is summarized | SC-68 | major (DA) | table: Table 18, round-five confirmatory read, RT-CGM row | 4 (checked against accuracy_sweep_r3.json, accuracy_sweep_r5.json and protocol_sweep_r5.json) | DA-M3 | must_fix | sentence: Abstract; §11 | reporting_requirement_unmet, claim: abstract round-five sentence |

### Required Item Details

> **Ordinal contract (#576/#670):** `R<n>` follows immutable roadmap source order filtered to `obligation_class == must_fix`; author-selected display order and triage never enter the derivation. Blocks are exactly `R1..R21`.

**R1: One contribution, or two stated research questions**
- **Problem**: The title, research question, Methods and Discussion are written around the calibration-coverage study. The five accuracy sweeps carry their own endpoints, gates and data, and their methods appear only inside Results (§7.9–§7.13). The sweeps also close the abstract. As a result, no single takeaway can be recovered in one pass.
- **Source**: EIC-W1 (major, 5). DA Observation 1 adds that the sweep diary answers a different question from §1 and fills about half the abstract.
- **Requirement**: Either make the calibration-coverage study the paper, or state two research questions up front. In the first case, move the sweeps to an appendix or a separate report, keep the round-one gate lesson as one Discussion paragraph, and move client-engineering detail to an appendix. In the second case, give the sweeps their own Methods and Discussion subsections and retitle.
- **Acceptance criteria**: Title, abstract, contribution list and conclusion name the same contribution or contributions, and every reported study has a Methods description outside Results.

**R2: Position the novelty against the PTQ calibration-data literature**
- **Problem**: §2.4 and §8.1 position the "sharpest observation" against vendor guidance and three PTQ method papers. They do not engage work in which calibration-data choice is itself the studied lever. A reader therefore cannot separate what is new from what that literature anticipates.
- **Source**: EIC-W2 (major, 4); R2-W2 (major, 4); the DA's strongest counter-argument ("measures, with unusual care, a failure the field already knows to avoid").
- **Requirement**: Add a calibration-data-selection paragraph to §2.4. It should cite the works named by EIC-W2 and R2-W2: Williams and Aletras 2024; Yu, Yang and Shi 2021; Cai et al. 2020; Krishnamoorthi 2018; Nagel et al. 2021; Wu et al. 2020. Restate the §1 and §8.1 novelty around the overlapping-window concentration mechanism, class-asymmetric clipping, the accuracy illusion, and their participant-clustered measurement with a mechanism control. Present the result as confirmation and measurement of a known lever. Word the nulls as R20 requires and the cohort evidence as R6 requires.
- **Acceptance criteria**: §2.4 characterizes the named calibration-data literature, and the §1 and §8.1 novelty statement names the time-series-specific mechanism instead of "nominal sample count said nothing".

**R3: Cite the closest on-device and health time-series prior work**
- **Problem**: No quantized on-device health or wearable time-series classifier is cited. Table 1 lists Li et al. (2020) as the only device-evidence row, although Zhu et al. ran a recurrent glucose predictor on a low-power SoC. Two rows give OhioT1DM as their evaluation scope without citing it.
- **Source**: EIC-W2 (major, 4); R2-W10 (minor, 4). A SPLIT, arbitrated under D1.
- **Requirement**: Add a related-work paragraph on quantized on-device health and wearable time-series classifiers. Add Zhu et al. (IEEE Internet of Things Journal 2022) as a device-evidence row in Table 1, and cite Marling and Bunescu (2020) for OhioT1DM.
- **Acceptance criteria**: Table 1 and §2 include on-device glucose-prediction prior work with device evidence and the OhioT1DM citation.

**R4: Present round five as an artifact-level ship decision, or supply recipe-level evidence**
- **Problem**: The round-five read compares one seed-4 candidate artifact with one champion artifact, and its interval is conditional on those two fitted models. The +0.20-point gain is about a fifth of the paper's own 1.1-point five-seed spread. Even so, the contribution list ("confirmed"), the abstract and the conclusion ("promoted") state design-level conclusions. R1 shows from `accuracy_sweep_r3.json` that a chosen-seed effect of +0.77 points shrank to +0.05 at the five-seed mean.
- **Source**: EIC-W3 (major, 4); R1-W3 (major, 5); DA-M3 (major band, 4).
- **Requirement**: Choose one of two routes. (a) Evaluate all five seeds of the candidate and champion recipes on RT-CGM as a descriptive analysis (the confirmatory read is already spent), with seed-and-participant uncertainty from a two-stage bootstrap or a mixed model. (b) Restate the contribution list, abstract and conclusion at artifact level ("the seed-4 flatten-head artifact passed the pre-registered gates") and place the +0.20-point gain beside the 1.1-point seed spread.
- **Acceptance criteria**: No sentence credits the flatten head or the cosine schedule with an improvement unless a multi-seed RT-CGM comparison is reported, and every citation of the +0.20-point gain states the seed spread.

**R5: Replace the "toolchain-default" label**
- **Problem**: Six passages call contiguous, dataset-order first-N selection the toolchain default. Neither toolchain chooses calibration data. The cited RepresentativeDataset page asks for a subset "randomly chosen, in no particular order", and §5.1 describes the selection as this pipeline's own choice.
- **Source**: EIC-W4 (major, 3); R2-W1 (major, 5); DA-m11 (minor band, 4). R3-Q5 uses the paper's phrase in a question; that is not a position on SC-5, and the answer should use the renamed arm.
- **Requirement**: Rename the arm throughout (for example, "contiguous first-N, dataset-order selection"). Quote the cited RepresentativeDataset and LiteRT guidance accurately in §2.4 and §8.1, and state that the sequential arm contravenes it. If prevalence is argued, cite the toolchain, version and example code that feed calibration data in stored order.
- **Acceptance criteria**: No passage calls the sequential arm a toolchain default without a cited toolchain source, and §8.1 quotes the random, unordered selection instruction.

**R6: Report the second-cohort predictions as the frozen records allow**
- **Problem**: Four places claim a confirmation: the abstract ("reproduces on two further cohorts"), §7.7 ("Both predictions held"; "near zero as predicted"), Limitations item 2, and the conclusion. The record does not support that:
  - The frozen dubosson prediction ("POSITIVE") states no decision rule, and the hall prediction ("NEAR ZERO") states no margin.
  - Both designated intervals include zero.
  - The hall premise (a ceiling near 240 mg/dL) was contradicted by a realized sequential ceiling of 165 mg/dL.
  - Hall's 50-participant temporal component shows +4.3 [+1.1, +7.3] points.
- **Source**: EIC-W5 (minor, 4); R1-W1 (major, 5); DA-M1 (major band, 5). A SPLIT, arbitrated toward R1 under D2.
- **Requirement**:
  - State each frozen prediction's outcome separately. For dubosson: a positive point estimate whose interval includes zero. For hall: the designated held-out contrast is inconclusive, the temporal component is positive at +4.3 points, and the frozen premise was contradicted by the 165 mg/dL sequential ceiling.
  - Remove "reproduces", "Both predictions held" and "near zero as predicted" from the abstract, §7.7, Limitations item 2 and the conclusion.
  - Label the interpretation exploratory.
- **Acceptance criteria**: No passage states that the predictions held or that the effect reproduced, and §7.7 reports each designated contrast with its interval and the outcome of the hall premise.

**R7: Characterize the cohorts behind the shipped model**
- **Problem**: The shipped model was trained on five DiaData cohorts that the manuscript never names, and was certified on "RT-CGM", which is described only by size and cadence. The information is already in `diadata_audit_r1.json`.
- **Source**: EIC-W6 (minor, 4); R2-W8 (minor, 4); R3-W2 (major, 5). A SPLIT, arbitrated under D3.
- **Requirement**: Add a cohort table for every GlucoBench and DiaData cohort used. Give each cohort's role, source trial and citation, n, age range, diabetes type, therapy, CGM model and generation, blinded or real-time wear, collection years and reportable range, and mark the device as unrecoverable where it is. Describe RT-CGM's devices and era in §7.11 and §10.2.
- **Acceptance criteria**: Every cohort named in the abstract, §7.10–§7.13 or Table 1 appears in the cohort table with its role and sensor generation.

**R8: Correct "gained only on the majority class"**
- **Problem**: Table 13 shows that rising recall also improved (0.365 versus 0.334). The body's "almost entirely" is accurate; the abstract's "only" and the contribution list's "majority-class only" are not.
- **Source**: EIC-W7 (minor, 4); R1-W9 (minor, 4); DA-m5 (minor band, 5).
- **Requirement**: Replace the "only" wording in the abstract and the contribution list with wording consistent with Table 13, and ground the no-ship decision in the directional-recall drop (0.517 to 0.438).
- **Acceptance criteria**: The abstract and contribution list agree with Table 13's per-class values.

**R9: Correct the read-discipline statement**
- **Problem**: The abstract ("each judged by one pre-registered read") and the conclusion ("Each round spent one pre-registered confirmatory read") contradict §7.12 and the Table 17 caption: round four performed no confirmatory read.
- **Source**: EIC-W8 (minor, 5); R1-W6 (minor, 4); DA-m1 (minor band, 5).
- **Requirement**: In the abstract, the contribution list and the conclusion, state that four rounds spent one confirmatory read each and that round four ended on an identity outcome without a read.
- **Acceptance criteria**: The count of confirmatory reads in the abstract, contribution list and conclusion matches §7.12 and the Table 17 caption.

**R10: Correct the bound on the null knobs**
- **Problem**: The conclusion's "at most eight hundredths of a point" fails for the observer arm. For range coverage 200, draw 101, the observer moves recall from 0.48653 to 0.48771, a change of 0.12 points (`aggregate_r1.json`).
- **Source**: EIC-W9 (minor, 5); R1-W8 (minor, 5); DA-m2 (minor band, 5).
- **Requirement**: Restate the bound as about a tenth of a point, or report the observer comparison separately, consistent with the quantitative wording adopted under S37.
- **Acceptance criteria**: Each quantitative bound in the abstract, contribution list and conclusion is at least the largest corresponding contrast in `aggregate_r1.json`.

**R11: Use the corrected-protocol ceiling in §7.7**
- **Problem**: §7.7 pairs the historical artifact's 239.9 mg/dL ceiling (Table 4) with the corrected-protocol 13.4-point effect. The sequential-200 ceiling for that effect is 250.9 mg/dL (Table 5).
- **Source**: EIC-W10 (minor, 5); R1-W8 (minor, 5); DA-M1 (major band, 5).
- **Requirement**: Use the corrected sequential-200 ceiling (250.9 mg/dL, maximum 251; Table 5) in the §7.7 mechanism argument, and keep 239.9 for the historical artifact only.
- **Acceptance criteria**: §7.7 quotes the Table 5 ceiling for the 13.4-point contrast.

**R12: Rebase the dose-response argument on clipped-window shares**
- **Problem**: The mg/dL gaps (about 150, 114 and 138) do not order the effects (13.4, 2.0 and 1.3) when hall's primary contrast is used. They do order them with hall's temporal +4.3, but that contradicts the near-zero prediction. The quantity the mechanism implies, the share of windows above each cohort's sequential ceiling, is not reported.
- **Source**: EIC-W10 (minor, 5); DA-M1 (major band, 5); DA-A5.
- **Requirement**: Report the share of windows above each cohort's sequential-200 ceiling, per component (Weinstock, dubosson, hall), and base any dose-response statement on those shares rather than on mg/dL gaps.
- **Acceptance criteria**: §7.7 or Table 11 reports the clipped-window shares, and the dose-response sentence cites them.

**R13: Keep the rate-band wording neutral**
- **Problem**: §7.12 calls the bands "the unchanged clinical thresholds" and decodes "at the clinical thresholds", and §8.2 introduces "clinical-shape classifier". Yet §3 presents the same bands as trend-arrow conventions, and the paper disclaims clinical use.
- **Source**: EIC-W15 (minor, 4); R2-W9 (minor, 5); R3-W5 (minor, 4); DA-m12 (minor band, 5). CONSENSUS-3, with R1 silent.
- **Requirement**: Use "trend-arrow rate bands (±1 and ±2 mg/dL/min)" in §7.12, and "directional classifier" or "directional-recall profile" in §8.2.
- **Acceptance criteria**: No occurrence of "clinical threshold" or "clinical-shape" remains in the manuscript.

**R14: Disclose small-cohort fragility in Table 11**
- **Problem**: 1.2% of the dubosson bootstrap draws had an undefined endpoint and were dropped before the percentile step. Table 7's caption reports the equivalent figure for the main study. Directional support is also small: 28 falling_fast windows in dubosson temporal, and 41 falling_fast and 50 rising_fast windows in hall held-out.
- **Source**: R1-W1 (major, 5); DA-m6 (minor band, 5).
- **Requirement**: Add per-class support and the undefined-draw fraction for each designated contrast to Table 11, and state how undefined draws are handled.
- **Acceptance criteria**: Table 11 reports per-class support and undefined-draw fractions (for example, 0.012 for the dubosson temporal contrast).

**R15: Small-cluster-valid inference for six-cluster cohorts**
- **Problem**: The percentile pairs-cluster bootstrap is applied unchanged at six clusters (dubosson temporal; hall held-out). Six clusters admit only 462 distinct resamples, and percentile intervals at that cluster count are expected to be too narrow (Cameron, Gelbach and Miller 2008).
- **Source**: R1-W2 (major, 4).
- **Requirement**: For cohorts with fewer than about 15 participants, report the per-participant paired differences. Then either use a small-cluster-valid procedure (a wild cluster bootstrap-t with Rademacher or Webb weights, or exact sign-flip randomization over participants) or present those cohorts without intervals.
- **Acceptance criteria**: Every interval for a cohort with fewer than about 15 participants names a small-cluster-valid procedure, or no interval is shown and per-participant differences are reported.

**R16: Attribute round five to the bundled candidate**
- **Problem**: The candidate differs from the champion in head, learning-rate schedule, validation-chosen kept epoch and seed. The abstract credits the head alone, and the conclusion names head and schedule.
- **Source**: R1-W4 (minor, 4); DA-M3 (major band, 4).
- **Requirement**: Attribute the round-five result to the bundled candidate (head, cosine schedule, validation-chosen epoch and seed), and state that the factors are not separately identified at the confirmatory read.
- **Acceptance criteria**: The abstract, §7.13 and the conclusion name the bundle, not a single factor.

**R17: State the take-home as input-range coverage**
- **Problem**: "Coverage means covering participants" credits participant spread. However, sequential-1,000 covers only 6 participants yet recovers fully (+0.05 [−0.05, +0.15]), and the clip-only control reproduces the loss. The operative variable is the selection's maximum relative to the population tail.
- **Source**: R1-W4 (minor, 4); DA-M2 (major band, 5).
- **Requirement**: Rephrase the conclusion's take-home and the §8.1 rule as covering the deployment input range (randomized selection achieves this in this pool), consistent with the sequential-1,000 row of Table 7 and the clip-only control.
- **Acceptance criteria**: The conclusion's take-home names input-range coverage.

**R18: Rename the round-four quantity**
- **Problem**: `exp0_sweep_r4.json` compares two labels computed from the same future readings, g[t..t+3]. Their agreement of 0.954 measures label-definition stability, not the recall attainable from 12 past inputs. So the "ceiling" and the 42-point "headroom" bound nothing a past-only model can reach.
- **Source**: R1-W10 (minor, 3); R2-W3 (minor, 5); DA-m10 (minor band, 4).
- **Requirement**: Call the round-four measurement a label-definition stability check. Remove "ceiling" and "headroom" from the contribution list, §7.12 and the conclusion, and state that the recall attainable from CGM-only inputs is unknown.
- **Acceptance criteria**: No occurrence of "label-noise ceiling" or "headroom" remains, and §7.12 defines the quantity as agreement between two labels computed from the same horizon readings.

**R19: Limit the label-noise conclusion**
- **Problem**: §7.12 ("not label-noise-limited") and the conclusion ("places the remaining error in the model and data, not the labels") go beyond what was measured. The measurement used independent noise of σ = 1 and 2 mg/dL only, which omits blood-to-interstitial lag, calibration drift, autocorrelated noise and transient artifacts.
- **Source**: R1-W10 (minor, 3); R2-W3 (minor, 5); R3-W3 (major, 3); DA-m10. A SPLIT, arbitrated under D4.
- **Requirement**: Remove or rescope the claim that the remaining error lies in the model and data rather than the labels. Any label-robustness sentence that remains must name the noise model (independent σ = 1 and 2 mg/dL) and cite a CGM sensor-error model (Facchinetti et al. 2014). §7.12 should also state that labels are defined on the sensor trace. A colored-noise rerun by sensor generation is optional.
- **Acceptance criteria**: §7.12 and the conclusion claim nothing about where the remaining error lies beyond the tested noise model.

**R20: Scope the null contrasts to the saturated regime**
- **Problem**: Every randomized 200-window selection spans 39–401 mg/dL and receives the same INT8 input range (0–400.8; Table 5). The pre-registered primary contrast (+0.06 [−0.02, +0.13]) and the randomized budget contrasts therefore compared arms with identical input quantization. Even so, §8.1 and the conclusion turn these nulls into a design rule. Range-aware selection was never run on the cohorts where random draws under-covered the range (dubosson's uniform ceiling 352.7 against a maximum of 400; hall's 233.9 against 303).
- **Source**: DA-M2 (major band, 5). Tension with the EIC's novelty list, arbitrated under D7.
- **Requirement**: Report the primary contrast as uninformative about input-range-aware selection in this pool, where all randomized 200-window selections reach the same input range. Then either scope the stratification and budget nulls to that saturated regime, or test range-aware selection on a cohort where randomized draws under-cover the range (dubosson, hall).
- **Acceptance criteria**: §7.3, §8.1 and the conclusion state the shared-input-range condition wherever the nulls are generalized, or report the additional test.

**R21: Carry round-five costs wherever the result is summarized**
- **Problem**: The abstract reports only the round-five gain. The record also shows costs:
  - On both Weinstock components the candidate fails gates 1 and 2 (`gate1_beats_champion` and `gate2_stable_floor` are false in `accuracy_sweep_r5.json`).
  - PEDAP directional recall is 0.539 versus 0.541.
  - Macro-F1 on the primary cohort is lower (0.410 versus 0.427).
- **Source**: DA-M3 (major band, 4).
- **Requirement**: Wherever the round-five result is summarized, report beside the +0.20-point gain the Weinstock gate outcomes, the primary-cohort macro-F1 change and the PEDAP directional recall.
- **Acceptance criteria**: Each summary of the round-five result states its Weinstock gate outcomes and the primary-cohort macro-F1 change.

### Suggested Revisions (Should Fix / Consider)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| S1 | REV-07: update Figure 1, §4.3 and Table 1, and add a model-variants table (architecture, parameters, encoding, loss, training data, calibration pool, hash) | SC-7 | minor | figure: Figure 1, end-to-end workflow (rendered page 5) | 4 | EIC-W6 | should_fix | section: §4.3, Figure 1, Table 1 | reader_traceability_reduced, figure: Figure 1 |
| S2 | REV-09: state the INT8 calibration pool of the shipped model | SC-9 | EIC minor; R1 minor | text: §Reproducibility "so a fresh run may produce different numbers" [R1-W11] | EIC 4; R1 4 | EIC-W6; R1-W11 | should_fix | sentence: sweep methods, §7.13 | method_reproducibility_unresolved, section: sweep methods |
| S3 | REV-10: shorten the abstract to about 200 words | SC-10 | minor | text: Abstract "difference-encoded inputs make quantization nearly calibration-proof" | 4 | EIC-W7 | should_fix | section: Abstract | editorial_conformance_unmet, section: Abstract |
| S4 | REV-11: label the 13.4-point headline as a prespecified secondary, and state the null primary | SC-11 | EIC minor; R1 minor | text: §Experimental Protocol "Secondary contrasts are range coverage versus uniform random, 1,000 versus 200 within each strategy, and uniform random versus sequential at each budget." [R1-W5] | EIC 4; R1 5 | EIC-W7; R1-W5 | should_fix | sentence: Abstract; §1 | reporting_requirement_unmet, claim: abstract headline contrast |
| S5 | REV-12: label delta encoding as post-hoc and the anchor variant as held-out-selected; temper "nearly calibration-proof" | SC-12 | EIC minor; R1 minor; DA minor | dataset: delta_inputs_r1.json, protocol disclosure and ship_gate [DA-m9] | EIC 4; R1 4; DA 5 | EIC-W7; R1-W6; DA-m9 | should_fix | sentence: Abstract; §7.8 | reporting_requirement_unmet, claim: abstract delta-encoding sentence |
| S6 | REV-16: write "of 200 windows", not "the same 200 windows" | SC-16 | minor | text: §11 Conclusion "stratification, budget, and observer choice moved it by at most eight hundredths of a point" | 5 | EIC-W9 | should_fix | sentence: §11 | interpretive_ambiguity_remains, claim: §11 randomized-selection sentence |
| S7 | REV-19: update the §6 device-timing sentence and the Limitations item 7 count (per the record, 8 of 110 INT8 artifacts timed and 102 untimed) | SC-19 | EIC minor; R1 minor | text: §6 "none of the 110 new artifacts has been device-timed" | EIC 5; R1 5 | EIC-W11; R1-W7 | should_fix | sentence: §6; Limitations item 7 | reader_traceability_reduced, section: §6 |
| S8 | REV-20: presentation fixes. Move revision history to a provenance appendix; restore type sizes and merge the sweep confirmatory tables; move the Figure 3 legend off the bars; name each reference model with its precision and split; label the column "Macro recall" | SC-20, SC-21, SC-22, SC-24 | minor ×4 | text: §10.1 "The September revision reused the July results" (W12); table: Table 7 (W13); figure: Figure 3 (W14); table: Table 17 caption (W16) | 4 ×4 | EIC-W12, W13, W14, W16 | should_fix | section: §10.1, §7.14, Table 20; Tables 1, 6, 7, 10–19; Figure 3 | editorial_conformance_unmet, manuscript: typesetting and structure |
| S9 | REV-22: report the frozen budget-by-strategy interaction, or state why it was dropped | SC-25 | EIC minor; R1 minor; DA minor | absence: §6 secondary-contrast list and Table 7 — expected the budget-by-strategy interaction contrast that protocol.json lists among the frozen secondary contrasts; checked §6, Table 7, and the contrasts block of aggregate_r1.json | EIC 4; R1 5; DA 5 | EIC-W17; R1-W5; DA-m7 | should_fix | re_analysis: interaction interval from aggregate predictions; §6 | reporting_requirement_unmet, table: Table 7 |
| S10 | REV-23: version or cite recency_collision.json and the round-four frozen design (spec at commit c8f3de6) | SC-26 | minor | absence: paper/results/ and paper/results/calibration-coverage/ — expected recency_collision.json, the diagnostic cited in §7.13 as the round-five motivation; checked both directory listings and the §10.1 inventory of versioned records | 4 | EIC-W18 (narrowed on the record) | should_fix | other (surface_id: supplementary_record): paper/results/calibration-coverage/; §10.1 | method_reproducibility_unresolved, dataset: paper/results/calibration-coverage/ |
| S11 | REV-24: state the licence of the distributed weights and add a conventional ethics statement | SC-27 | minor | text: §10.2 "We do not claim an institutional ethics determination for this work." | 3 | EIC-W19 | should_fix | sentence: §10.2 | reporting_requirement_unmet, section: §10.2 |
| S12 | REV-29: add a protocol-to-report table covering H1–H4, the primary and every secondary with their status; report seed and draw variation; give an interval for sequential 1,000 vs 200 | SC-32, SC-33, SC-34 | R1 minor; DA minor | text: §Experimental Protocol "Secondary contrasts are range coverage versus uniform random, 1,000 versus 200 within each strategy, and uniform random versus sequential at each budget." | R1 5; DA 5 and 4 | R1-W5; DA-m7; DA-m8 | should_fix | section: §6 plus a protocol-to-report table | reporting_requirement_unmet, section: §6 |
| S13 | REV-30: add a held-out read ledger; disclose in the abstract that rounds 1–3 reused one split and that round one was decided post hoc | SC-35 | R1 minor; DA minor | text: Abstract "Five staged accuracy sweeps followed, each judged by one pre-registered read."; Limitations "Weinstock held-out and temporal labels were read in rounds one to three" | R1 4; DA 5 | R1-W6; DA-m1 | should_fix | section: read-ledger table; Abstract | reader_traceability_reduced, section: sweep sections |
| S14 | REV-31: publish the operator inventory and size for the confirmation and delta artifacts | SC-36 | minor | text: Limitations "The remaining 101 matrix, confirmation-cohort, and delta-encoded artifacts were not device-timed" | 5 | R1-W7 | should_fix | other (surface_id: supplementary_record): confirmation and delta records | method_reproducibility_unresolved, dataset: confirmation_*.json, delta_inputs_r1.json |
| S15 | REV-32: quote the widest single-tier interval, not a spliced one | SC-37 | minor | text: Conclusion "moved it by at most eight hundredths of a point" [R1-W8 parent anchor] | 5 | R1-W8 | should_fix | sentence: §7.6 sensitivity-tier sentence | claim_scope_unsupported, section: §7.6 |
| S16 | REV-33: report per-class F1 and precision for the round-one comparison | SC-38 | minor | text: §Accuracy sweep "decomposes the macro-F1 gain by class" | 4 | R1-W9 | should_fix | re_analysis: per-class F1 and precision from the round-one predictions | interpretive_ambiguity_remains, table: tab:sweeprecall |
| S17 | REV-36: recomputability. Publish per-artifact hashes, per-participant confusion counts where source terms allow, and the sweep batch size (512 vs the stated 64) | SC-41 | minor | text: §Reproducibility "so a fresh run may produce different numbers" | 4 | R1-W11 | should_fix | other (surface_id: supplementary_record): per-artifact hashes; §10.1 | method_reproducibility_unresolved, dataset: paper/results/calibration-coverage/ |
| S18 | REV-37: acknowledge rate-error-grid and time-gain evaluation in §2.3 | SC-42 | R2 minor; R3 minor | text: §2.3, "Forecasting work evaluates RMSE, where direction surfaces only as a safety penalty on incoherent predicted variation" | R2 4; R3 4 | R2-W4; R3-W9 | should_fix | section: §2.3 | claim_scope_unsupported, section: §2.3 |
| S19 | REV-38: correct the Kistkins et al. quotation | SC-43 | minor | text: §2.3, "recalls of 0.85--0.96 at 15-minute and 1-hour" | 5 | R2-W5 | should_fix | sentence: §2.3 | claim_scope_unsupported, section: §2.3 |
| S20 | REV-39: treat 39 and 401 mg/dL as censoring codes; report shares by class and add a sensitivity tier | SC-44 | R2 minor; R3 minor | text: §7.4, "at the 401 sensor cap" | R2 4; R3 4 | R2-W6; R3-W7; R3-Q2 | should_fix | re_analysis: censored-window shares by class and a sensitivity tier; §4.1 | evidence_gap_remains, section: §4.1 |
| S21 | REV-40: state what calibration changes inside the INT8 artifact; publish the per-op dtype map; state the CPU backend and thread count | SC-45 | minor | text: §7.15, "changes nothing the device sees except the two quantization parameters" | 4 | R2-W7 | should_fix | sentence: §7.15, plus the supplement | interpretive_ambiguity_remains, section: §7.15 |
| S22 | REV-41: RT-CGM audit. Use a cadence rule that can fail; explain the 96 subjects without windows and the collinear fraction | SC-46 | R2 minor; R3 note [SEVERITY-SOURCE: letter-fallback] minor | text: §7.11 (round three), "A cadence audit admitted all 451 RT-CGM subjects under" | R2 4; R3 [CONFIDENCE-SOURCE: none stated] | R2-W8; R3-N1 | should_fix | section: §7.11 | evidence_gap_remains, section: §7.11 |
| S23 | REV-42: cite the sweep methods (focal loss, TCN, cosine schedule, CORN, model soups, ordinal smoothing, distance penalty) | SC-47 | R2 minor; R3 minor | text: §7.12, "A CORN ordinal head reached 0.470" | R2 5; R3 4 | R2-W11; R3-W8 | should_fix | sentence: §7.9–§7.12 | reader_traceability_reduced, section: §7.9–§7.12 |
| S24 | REV-43: frame soup collapse and regression mean-seeking as known results | SC-48 | minor | text: §7.12, "A CORN ordinal head reached 0.470" | 5 | R2-W11 | should_fix | sentence: §7.12 | interpretive_ambiguity_remains, section: §7.12 |
| S25 | REV-44: discuss a fixed input range taken from the sensor's reportable range (the extra arm is optional) | SC-49 | R2 minor; DA-A1 [SEVERITY-SOURCE: letter-fallback] minor | text: §8.1, "calibration design reduces to one rule" | R2 4; DA [CONFIDENCE-SOURCE: none stated] | R2-W12; DA-A1 | should_fix | section: §8.1 (optional re_analysis: fixed-input-range arm) | interpretive_ambiguity_remains, section: §8.1 |
| S26 | REV-45: present the null resolution cost as the result 8-bit error analysis predicts | SC-50 | minor | text: §8.1, "The feared resolution trade-off" | 4 | R2-W13 | should_fix | sentence: §8.1 | interpretive_ambiguity_remains, section: §8.1 |
| S27 | REV-46: report both effective resolutions of the delta encoding | SC-51 | minor | text: §7.8, "model's INT8 input grid is $6.4\times$ finer (scale 0.247 versus 1.572" | 5 | R2-W14 | should_fix | sentence: §7.8 | claim_scope_unsupported, section: §7.8 |
| S28 | REV-47: translate the shipped model's error profile. Give a confusion matrix with sensitivity and PPV, direction-reversal shares and level-stratified recall, and extend the §8.2 paragraph. The minimum is the paragraph plus the confusion matrix | SC-52 | major | text: §7.12 "and the slow-to-fast over-prediction they cause" | 4 | R3-W1; R3-N1; DA stakeholders | should_fix | re_analysis: from cached predictions; §8.2 | interpretive_ambiguity_remains, section: §8.2 |
| S29 | REV-48: answer R3's questions Q1, Q4, Q5 and Q6 (Q2 is under S20 and Q3 under S28; answer Q5 using the arm renamed under R5) | SC-53 | — (question channel) | — | — | R3-Q1, Q4, Q5, Q6 | consider | other (surface_id: response_letter): answers | interpretive_ambiguity_remains, manuscript: response letter |
| S30 | REV-49: add a Limitations item on wear and therapy regimes and the feedback loop, the PEDAP population and unrepresented populations; expand RT-CGM at first use | SC-54 | major | absence: §7.10 (round two), §7.11 (round three) and §7.13 (round five) — expected a characterization of the five DiaData training cohorts, the RT-CGM confirmatory cohort and the PEDAP probe; checked Abstract, §1, §2.2, §4.1, §7.7, §7.10, §7.11, §7.13, §9, §10.2 and every table caption | 5 | R3-W2 | should_fix | section: Limitations; §7.13 | claim_scope_unsupported, section: Limitations |
| S31 | REV-50: terminology. Qualify "calibration"; write "the mean rate over the next 15 minutes"; expand acronyms at first use; map recall to sensitivity | SC-55, SC-56, SC-58 | minor ×3 | text: §1 "Here, coverage refers to the input values and trajectories"; Abstract "one of five rate-of-change classes 15"; Abstract "(100 INT8 artifacts)" | 5; 4; 4 | R3-W4; R3-W5; R3-W8 | should_fix | sentence: Title, Abstract, §1, first uses | interpretive_ambiguity_remains, manuscript: terminology |
| S32 | REV-51: state the Weinstock case-control enrichment and what blinded wear implies | SC-57 | minor | text: §2.2 "whose source study collected blinded" | 4 | R3-W6 | should_fix | sentence: §4.1; Limitations item 2 | claim_scope_unsupported, section: §4.1 |
| S33 | REV-52: add an intended-use statement and a model card; write "bundled in the research applications" | SC-59 | minor | text: Abstract "and now ships in both applications" | 3 | R3-W10; DA stakeholders | should_fix | section: §10.2; repository model card | interpretive_ambiguity_remains, section: §10.2 |
| S34 | REV-53: state that no INT8 latency or size benefit was measured for this model class, and scope "trade-offs" accordingly | SC-60 | [SEVERITY-SOURCE: letter-fallback] minor | table: Table 19; table: Table 20; text: §5.3 (locators cited in DA-A2 prose; no typed anchor in the card) | [CONFIDENCE-SOURCE: none stated] | DA-A2; DA-UP | should_fix | sentence: Title; §8; Limitations | claim_scope_unsupported, claim: title "Quantization Trade-offs" |
| S35 | REV-54: report PEDAP per-class changes (gains and losses) and discuss the operating-point-shift reading | SC-61 | [SEVERITY-SOURCE: letter-fallback] minor | dataset: accuracy_sweep_r5.json PEDAP per-class values (cited in DA-A3 prose; no typed anchor in the card) | [CONFIDENCE-SOURCE: none stated] | DA-A3 | should_fix | sentence: §7.13 | reporting_requirement_unmet, section: §7.13 |
| S36 | REV-55: reconcile the §8.1 mechanism statements (the start-high account vs slope erasure; dissent left unresolved) | SC-62 | minor | text: §8.1, "cannot see how high the trajectory started" | 4 | DA-A4; DA-m14 (dissent with R3-S4, D5) | should_fix | sentence: §8.1 | interpretive_ambiguity_remains, section: §8.1 |
| S37 | REV-56: replace "add nothing measurable" with a quantitative bound | SC-63 | minor | table: Table 7, Balanced 1,000 vs. 200 row | 5 | DA-m3 (dissent with R1's reporting judgement, D6) | should_fix | sentence: §1 contribution list; §7.3 | claim_scope_unsupported, claim: §1 contribution "add nothing measurable" |
| S38 | REV-57: correct the randomized participant range (90–127) and change "never-read" to "never evaluated" | SC-64, SC-65 | minor; minor | dataset: aggregate_r1.json selection_coverage, uniform_random_200_d102 and uniform_random_200_d103; text: Abstract, "zero-shot on a never-read cohort" | 5; 5 | DA-m4; DA-m13 | should_fix | sentence: §7.2; Abstract; §1 | claim_scope_unsupported, section: §7.2 |
| S39 | REV-58: state a gate-override rule for the sweeps | SC-66 | — (editorial channel; non-defect) | — | — | DA-O2 | consider | sentence: sweep protocol description | reader_traceability_reduced, section: sweep protocols |

### Source-Traceability Checklist

> This list keeps immutable source order and does not suggest a work order. Author triage is collected later as an explicit, complete sidecar.

- [ ] R1 (REV-01): obligation `must_fix`. Single contribution, or two stated research questions with matching structure.
- [ ] R2 (REV-02): obligation `must_fix`. Calibration-data-selection literature in §2.4; novelty restated in §1 and §8.1.
- [ ] R3 (REV-03): obligation `must_fix`. On-device health and glucose prior work; Zhu et al. 2022 row; OhioT1DM citation.
- [ ] R4 (REV-04): obligation `must_fix`. Five-seed RT-CGM evidence, or artifact-level restatement of round five.
- [ ] R5 (REV-05): obligation `must_fix`. Rename the "toolchain-default" arm; quote the cited guidance.
- [ ] R6 (REV-06): obligation `must_fix`. Second-cohort outcomes stated per prediction; "reproduces" and "held" removed.
- [ ] S1 (REV-07): obligation `should_fix`. Figure 1, §4.3 and Table 1 updated; model-variants table.
- [ ] R7 (REV-08): obligation `must_fix`. Cohort table; RT-CGM devices and era.
- [ ] S2 (REV-09): obligation `should_fix`. INT8 calibration pool of the shipped model stated.
- [ ] S3 (REV-10): obligation `should_fix`. Abstract shortened.
- [ ] S4 (REV-11): obligation `should_fix`. Headline labelled as a prespecified secondary; null primary stated.
- [ ] S5 (REV-12): obligation `should_fix`. Delta encoding labelled post-hoc and held-out-selected.
- [ ] R8 (REV-13): obligation `must_fix`. "Only on the majority class" corrected.
- [ ] R9 (REV-14): obligation `must_fix`. Confirmatory-read count corrected.
- [ ] R10 (REV-15): obligation `must_fix`. Bound on the null knobs corrected.
- [ ] S6 (REV-16): obligation `should_fix`. "Of 200 windows" wording.
- [ ] R11 (REV-17): obligation `must_fix`. 250.9 mg/dL ceiling in §7.7.
- [ ] R12 (REV-18): obligation `must_fix`. Per-cohort clipped-window shares.
- [ ] S7 (REV-19): obligation `should_fix`. Device-timing sentence and count.
- [ ] S8 (REV-20): obligation `should_fix`. Presentation fixes (history, typesetting, Figure 3, reference conventions).
- [ ] R13 (REV-21): obligation `must_fix`. Neutral rate-band wording.
- [ ] S9 (REV-22): obligation `should_fix`. Budget-by-strategy interaction reported.
- [ ] S10 (REV-23): obligation `should_fix`. recency_collision.json and round-four design versioned or cited.
- [ ] S11 (REV-24): obligation `should_fix`. Weights licence and ethics statement.
- [ ] R14 (REV-25): obligation `must_fix`. Undefined-draw fractions and per-class support in Table 11.
- [ ] R15 (REV-26): obligation `must_fix`. Small-cluster-valid inference, or no intervals.
- [ ] R16 (REV-27): obligation `must_fix`. Round five attributed to the bundled candidate.
- [ ] R17 (REV-28): obligation `must_fix`. Take-home stated as input-range coverage.
- [ ] S12 (REV-29): obligation `should_fix`. Protocol-to-report table; seed and draw variation; sequential 1,000 vs 200 interval.
- [ ] S13 (REV-30): obligation `should_fix`. Held-out read ledger.
- [ ] S14 (REV-31): obligation `should_fix`. Operator inventory and size for confirmation and delta artifacts.
- [ ] S15 (REV-32): obligation `should_fix`. Widest single-tier interval.
- [ ] S16 (REV-33): obligation `should_fix`. Per-class F1 and precision for round one.
- [ ] R18 (REV-34): obligation `must_fix`. Round-four quantity renamed; "ceiling" and "headroom" removed.
- [ ] R19 (REV-35): obligation `must_fix`. Label-noise conclusion limited to the tested noise model.
- [ ] S17 (REV-36): obligation `should_fix`. Per-artifact hashes, confusion counts, batch size.
- [ ] S18 (REV-37): obligation `should_fix`. Rate-error-grid and time-gain evaluation in §2.3.
- [ ] S19 (REV-38): obligation `should_fix`. Kistkins et al. quotation.
- [ ] S20 (REV-39): obligation `should_fix`. Censoring codes at 39 and 401 mg/dL.
- [ ] S21 (REV-40): obligation `should_fix`. Effect of calibration inside the INT8 artifact; dtype map; backend.
- [ ] S22 (REV-41): obligation `should_fix`. RT-CGM cadence audit; the 96 subjects.
- [ ] S23 (REV-42): obligation `should_fix`. Sweep-method citations.
- [ ] S24 (REV-43): obligation `should_fix`. Known-result framing for soups and regression.
- [ ] S25 (REV-44): obligation `should_fix`. Fixed input range from the sensor range discussed.
- [ ] S26 (REV-45): obligation `should_fix`. Resolution null framed as predicted.
- [ ] S27 (REV-46): obligation `should_fix`. Both delta-encoding resolutions reported.
- [ ] S28 (REV-47): obligation `should_fix`. Error profile translated for clinical readers.
- [ ] S29 (REV-48): obligation `consider`. Answers to R3 questions Q1, Q4, Q5 and Q6.
- [ ] S30 (REV-49): obligation `should_fix`. Regime and population limitations; RT-CGM naming.
- [ ] S31 (REV-50): obligation `should_fix`. Terminology for clinical readers.
- [ ] S32 (REV-51): obligation `should_fix`. Weinstock case-control design and blinded wear.
- [ ] S33 (REV-52): obligation `should_fix`. Intended-use statement and model card.
- [ ] S34 (REV-53): obligation `should_fix`. Measured absence of an INT8 benefit; "trade-offs" scoped.
- [ ] S35 (REV-54): obligation `should_fix`. PEDAP per-class changes.
- [ ] S36 (REV-55): obligation `should_fix`. §8.1 mechanism statements reconciled.
- [ ] S37 (REV-56): obligation `should_fix`. Quantitative bound replaces "nothing measurable".
- [ ] S38 (REV-57): obligation `should_fix`. Participant range and "never-read" corrected.
- [ ] S39 (REV-58): obligation `consider`. Gate-override rule.
- [ ] R20 (REV-59): obligation `must_fix`. Null contrasts scoped to the saturated regime.
- [ ] R21 (REV-60): obligation `must_fix`. Round-five costs reported beside the gain.

### Response Letter Template

Please respond item by item, in the REV order above, using the format in `templates/revision_response_template.md` (academic-paper-reviewer). The response must include:
1. A response and revision description for each Required Revision, R1–R21.
2. A response for each Suggested Revision, S1–S39: either the change made or the reason for not making it.
3. Change markup for every change in the revised manuscript.
4. A cross-reference table of new section and page locations.

Where an item records an arbitrated disagreement (D1–D7), address the arbitrated requirement and acknowledge the dissenting position. Author triage (`will_address`, `wont_address`, `not_on_point`) is collected separately in the author-adjudication sidecar; a decline stays visible and unresolved. The revised manuscript will be re-reviewed (Major Revision).

---

## Part 3: Reviewer Report Summary (Appendix)

| Reviewer | Role | Recommendation | Confidence |
|---|---|---|---|
| Journal-Fit Reviewer (EIC) | Action editor for an applied health-ML proceedings venue (ML4H/CHIL class) | Major revision (explicit signal) | Per-finding 3–5; no report-level score |
| Reviewer 1 | Biostatistician: clustered resampling, confirmatory ML evaluation | No verdict stated (dimension scores only) | Per-finding 3–5 |
| Reviewer 2 | On-device PTQ and CGM-forecasting benchmark researcher | No verdict stated (dimension scores only) | Per-finding 4–5 |
| Reviewer 3 | Diabetes-technology clinician-scientist | No verdict stated (dimension scores only) | Per-finding 3–5 |
| Devil's Advocate | Fixed adversarial seat | N/A, findings only | N/A, per finding only |

### Journal-Fit Review Report Summary
- Recommendation: major revision (explicit signal). Scores: D5 block; D6 block (repairable). Confidence: per-finding 3–5.
- Key point: The calibration half is a careful case study with a transferable rule. Its single contribution is diluted by the bundled sweep program, its novelty is not positioned against PTQ calibration-data and on-device health time-series work, and the conclusion contradicts the record twice (the eight-hundredths bound; one read per round).

### Reviewer 1 (Methodology) Summary
- Recommendation: no verdict stated. Scores: D1 block (repairable); D3 block (repairable). Confidence: per-finding 3–5.
- Key point: The core contrast re-derives from the released JSON and holds seed by seed (13.05–14.14 points; SD 0.47), and three GRIM receipts are consistent. The second-cohort "confirmation", the six-cluster intervals and the single-seed sweep certifications are stated more strongly than their inference allows.

### Reviewer 2 (Domain) Summary
- Recommendation: no verdict stated. Score: D2 block (repairable). Confidence: per-finding 4–5.
- Key point: The quantizer configuration and range arithmetic are exact. The "toolchain-default" label contradicts the documentation the paper cites, and the calibration-data literature is not engaged, which changes what the headline means. Twelve minor accuracy and provenance corrections follow.

### Reviewer 3 (Perspective) Summary
- Recommendation: no verdict stated. Score: D4 warn ("None of my block conditions fires"). Confidence: per-finding 3–5.
- Key point: The scope and safety language is better than is typical for the genre, and a clinical reader understands the calibration findings as written. The shipped model's error profile and its cohorts still need translating, and the label-noise conclusion needs limits.

### Devil's Advocate Summary
- Recommendation: N/A, findings only. Score: D3 block (repairable). CRITICAL table empty; MAJOR findings M1–M3.
- Key challenge: There is no unresolved Critical challenge. The strongest counter-argument is that the paper carefully measures a failure the field already knows to avoid:
  - the nulls compare arms with identical input ranges;
  - the external support is thin;
  - the round-five win sits inside seed noise;
  - no benefit side of INT8 is measured.

---

## Attachment: Acronym Check (advisory, #849)

_Run on a plain-text conversion of paper/main.tex (the script accepts only .md/.txt); line numbers refer to that conversion. Script output, not a reviewer finding; no reply needed._

### Acronym check (advisory; no reply needed)
Coverage: body, English abstract (partial)
Not in this input: Chinese abstract.
Not checked:
- Body, line 448: PT2E (the initials before its parentheses do not spell it)
- Body, line 1487: KiB (the initials before its parentheses do not spell it)
- Body, line 1651: INT8 (the initials before its parentheses do not spell it)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 38 | Not defined | CGM | 32 |
| Body | 178 | Not defined | iOS | 17 |
| Body | 209 | Not defined | RMSE | 1 |
| Body | 217 | Not defined | CNN | 6 |
| Body | 402 | Not defined | ReLU | 1 |
| Body | 446 | Not defined | LiteRT | 13 |
| Body | 495 | Not defined | SHA | 6 |
| Body | 522 | Not defined | CPU | 3 |
| Body | 533 | Not defined | API | 3 |
| Body | 585 | Not defined | QDQ | 5 |
| Body | 603 | Not defined | UI | 1 |
| Body | 604 | Not defined | S908E | 1 |
| Body | 604 | Not defined | SM | 1 |
| Body | 1089 | Not defined | CE | 3 |
| Body | 1114 | Not defined | T1D | 2 |
| Body | 1117 | Not defined | PEDAP | 5 |
| Body | 1144 | Not defined | JVM | 2 |
| Body | 1144 | Not defined | MODELS | 1 |
| Body | 1172 | Not defined | RT | 8 |
| Body | 1180 | Not defined | TCN | 4 |
| Body | 1253 | Not defined | CORN | 2 |
| Body | 1510 | Not defined | PTQ | 2 |
| Body | 1708 | Not defined | BY | 1 |
| Body | 1708 | Not defined | CC | 1 |
| Body | 1708 | Not defined | NC | 1 |
| EN abstract | 9 | Not defined | iOS | 1 |
| EN abstract | 12 | Not defined | CNN | 1 |
| EN abstract | 14 | Not defined | INT8 | 2 |
| EN abstract | 27 | Not defined | TCN | 1 |
| EN abstract | 29 | Not defined | RT | 1 |
