# Editorial Decision Package

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from a candidate or prose-named profile. `PROFILE_MEASURED` stays unavailable until a closed profile artifact and replay validator bind the exact target fields to the completed panel's `execution_topology_sha256`. Every seat card also declares `NOT_CALIBRATED`.

## Manuscript Information

- **Title**: GlucoEdge: Quantization-Calibration Coverage and Staged Model Selection in On-Device Glucose Trend Classification
- **Manuscript ID**: none (pre-submission round)
- **Decision Date**: 2026-10-18
- **Review Round**: fresh pre-submission round on an already-revised manuscript (contract stage note)
- **Contract**: `reviewer/reviewer_full/v2`, baseline `v3.20.0`, `panel_size` 5, contract file SHA-256 `f69078c34094b83f5589b97b348811d31e1bdd7649531516d28f6282df1feef1`
- **Manuscript read as data**: `paper/main.tex`, SHA-256 `d7aee27c5aa8fdc41c2127dc25cd74763b19c3e04c03e98ff5b76af2f357151b`
- **Review target**: `criteria_binding_unavailable`. All five seats' Phase 1 commitments carry this disclosure, and the EIC and R3 Phase 2 cards restate it. This package makes no venue-fit or venue-alignment claim.

## Review Panel Provenance (#540/#740)

- **Typed artifact**: `paper/analysis/panel-review-2026-10-18/provenance.json` (`review-panel-provenance/1.0`, mode `reviewer_full`, contract `reviewer/reviewer_full/v2`). Carrier state `valid`: the dispatching layer built it from `provenance_input.json` and replay-validated it (`review-panel-provenance: PASS`).
- **Artifact SHA-256**: `3650481576f001ff75fa9f5bf53c338c3f640fe2e2e3890cfa14e1c5a61f18f0` (exact raw bytes)
- **Panel ID**: `glucoedge-main-tex-round-3-fresh-panel`
- **Normalized manifest SHA-256**: `837e6409f84df443c734a1b748c09ab6005bd2b7a1ccda84e8609d47941d1d9f`
- **Execution topology SHA-256**: `c259b3f71e07a5edf63a2840650cfafb0d9380f68f5533b18fdd78f82f6e2603`
- **Fresh-context scope**: `within_panel_attempt_only` (does not compare retries or prior rounds)

| Seat | Role ID | Actor type | Context ID | Peer outputs visible | Model family | Provider | Human reviewer ID |
|---|---|---|---|---|---|---|---|
| EIC | eic | model | subagent-seat-eic | false | claude-opus-5-5 | anthropic | null |
| R1 | methodology | model | subagent-seat-r1 | false | claude-opus-5-5 | anthropic | null |
| R2 | domain | model | subagent-seat-r2 | false | claude-opus-5-5 | anthropic | null |
| R3 | perspective | model | subagent-seat-r3 | false | claude-opus-5-5 | anthropic | null |
| DA | da | model | subagent-seat-da | false | claude-opus-5-5 | anthropic | null |

| Provenance axis | Status (`true` / `false` / `unknown`) |
|---|---|
| Role-separated | true |
| Within-panel invocation-context separation | true |
| Blind to peer outputs | true |
| Model-family distinct | false |
| Provider distinct | false |
| Human-reviewer distinct | false |

- **Binary independence claim**: Not computed (artifact `independence_claim: not_computed_from_personas`). The recorded axes show role separation, separate invocation contexts within this panel attempt, and blindness to peer outputs. They also show that every seat ran on one model family and one provider, with no human reviewer. The panel is not described as independent.
- **Correlated-error disclosure** (required, reason `same_model_family`): All model-executed review seats used one model family; role separation does not remove correlated-error risk.
- **Known coverage gap (Phase 0)**: no seat is a CGM clinician or regulatory-science specialist. Clinical interpretation is a partial blind spot this round. The DA lists the missing stakeholder perspectives; the panel did not adjudicate them.

---

## Mechanical Decision Audit (Sprint Contract Synthesizer Protocol)

All five Phase 2 cards are usable, and each declares its dispatch role once (`eic`, `methodology`, `domain`, `perspective`, `da`). No seat dissents on its scoring plan. No eligible seat abstained.

### Step 1: Role-scoped scoring matrix

| Dimension | Priority | Eligible roles | Assessed eligible scores | Audit verdict |
|---|---|---|---|---|
| D1 methodology_rigor | mandatory | methodology | methodology = warn | warn |
| D2 domain_accuracy | mandatory | domain | domain = block (repairable) | block |
| D3 argumentative_coherence | mandatory | da, methodology | da = block (repairable); methodology = warn | block |
| D4 cross_disciplinary_relevance | high | perspective | perspective = warn | warn |
| D5 writing_and_structure | normal | eic | eic = warn | warn |
| D6 venue_fit_and_contribution | mandatory | eic | eic = block (repairable) | block |

No seat declared a fatal block, so no verdict renders as `block(fatal)`. Every dimension has at least one assessed eligible seat.

### Step 2: Failure conditions

| Condition | Severity | Quantifier | Expression | Per-dimension evaluation | Fired |
|---|---|---|---|---|---|
| F1 | 95 | any | any mandatory dimension has a fatal block | D1, D2, D3, D6: no fatal score | false |
| F2 | 90 | any | any mandatory dimension scores 'block' | D2 true (1 of 1); D3 true (1 of 2); D6 true (1 of 1) | true |
| F3 | 70 | majority | two or more mandatory dimensions score 'warn' or worse | D1 n=1, owner warn: true; D2 n=1: true; D3 n=2, both seats warn or worse: true; D6 n=1: true. 4 dimensions, at least 2 needed | true |
| F4 | 60 | any | any high-priority dimension scores 'block' | D4 n=1, warn: false | false |
| F5 | 40 | any | any dimension scores 'warn' or worse | all six dimensions true | true |
| F0 | 10 | all | every dimension scores 'pass' | D1 not pass: false | false |

### Step 3: Precedence and emission

The fired conditions are F2 (90), F3 (70) and F5 (40). F2 has the highest severity and there is no tie, so its action applies. The DA's `#### CRITICAL` table has no rows, so the adjudication list is empty and no rejection rationale is owed. The accept-path consistency marker does not apply, because the decision is not accept.

```
dimension_verdicts: [D1=warn, D2=block, D3=block, D4=warn, D5=warn, D6=block]
fired_conditions: [F2, F3, F5]
da_critical_adjudications: []
editorial_decision=major_revision
```

Cross-model blind decision check (#518): `ARS_CROSS_MODEL` is not set, so the check was not run.

### Schema 6 criterion adapter (sprint scale, copied without translation)

`judgement_scale: sprint_contract` for every cell. "ineligible" marks a structural `not_assessed`.

| Dimension | eic | methodology | domain | perspective | da |
|---|---|---|---|---|---|
| D1 | not_assessed (ineligible) | warn | not_assessed (ineligible) | not_assessed (ineligible) | not_assessed (ineligible) |
| D2 | not_assessed (ineligible) | not_assessed (ineligible) | block, repairable | not_assessed (ineligible) | not_assessed (ineligible) |
| D3 | not_assessed (ineligible) | warn | not_assessed (ineligible) | not_assessed (ineligible) | block, repairable |
| D4 | not_assessed (ineligible) | not_assessed (ineligible) | not_assessed (ineligible) | warn | not_assessed (ineligible) |
| D5 | warn | not_assessed (ineligible) | not_assessed (ineligible) | not_assessed (ineligible) | not_assessed (ineligible) |
| D6 | block, repairable | not_assessed (ineligible) | not_assessed (ineligible) | not_assessed (ineligible) | not_assessed (ineligible) |

---

## Synthesis Working Record

### Step 1a: Reviewer summary matrix

| Item | Journal-Fit Reviewer (EIC) | R1 (Methodology) | R2 (Domain) | R3 (Perspective) | DA |
|---|---|---|---|---|---|
| Overall recommendation | Major Revision | Major Revision (stated as advisory) | Major Revision | Minor Revision (perspective seat only) | N/A: findings only |
| Confidence / scope disclosure | 4. Did not audit statistics or the quantization toolchain | 4. Recomputed from versioned records; no retraining; subset analyses not checked | 4. Not a clinician; took the quantization pipeline as described | 4. Toolchain probe on a random-weight toy model; no device timing | per finding only |
| Contract scores | D5 warn; D6 block (repairable) | D1 warn; D3 warn | D2 block (repairable) | D4 warn | D3 block (repairable) |
| Key strengths | Read-ledger governance (S1); large effect attributed by mechanism (S2); candid self-scoping (S3); recomputable counts (S4); GlucoBench prefix-row pitfall (S5); consistent non-clinical scoping (S6) | RQ1 holds for the recipe (S1); interventional mechanism controls (S2); cluster-aware inference (S3); adaptivity documented per look (S4); recomputable counts (S5); round-five confirmation on disjoint data (S6) | Label kept distinct from device arrows (S1); sensor-code awareness (S2); error profile beyond recall (S3); honest RQ1 positioning (S4); cohort limits stated (S5); published counts (S6); precise non-clinical semantics (S7) | Mechanism controls (S1); per-operator disclosure (S2); terminology for three communities (S3); benchmark hygiene (S4); client quantization arithmetic specified (S5); transferable CI lesson (S6) | Clip-only control; read ledger and deviation notes; exact numeric bookkeeping |
| Key weaknesses | see Step 1b (W1-W12: 3 Major, 9 Minor) | see Step 1b (W1-W9: 2 Major, 7 Minor) | see Step 1b (W1-W7: 2 Major, 5 Minor) | see Step 1b (W1-W6: 1 Major, 5 Minor) | 0 CRITICAL; 4 MAJOR (M1-M4); 7 minor (m1-m7) |
| Questions for authors | 5 | 6 | 5 | 5 | none |
| Minor issues outside the W list | 0 (all listed as W4-W12) | 0 | 5 | 3 | 7 (m1-m7) |

### Step 1b: Weakness sub-claim inventory

Only non-silent positions are listed. Any of the four non-DA seats not shown for a sub-claim is `not-mentioned`, which is silence, not opposition. DA rows are recorded for visibility and do not enter the consensus count. Severity and confidence are transported from each card's per-finding tags; no fallback tags were needed.

| sub_claim_id | parent_weakness | reviewer_id | position | evidence_pointer | severity | confidence |
|---|---|---|---|---|---|---|
| SC-1 | EIC W1: RQ1's increment over established guidance is thin (the governing rule is conceded; the new part is the overlapping-window trap, its size and the clipping mechanism; nulls confined to a saturated regime) | EIC | raised | text: Section 2.4 "That calibration data must represent the deployment inputs is therefore established guidance, not a finding of this paper." | major | 4 |
| SC-2 | EIC W1: the frozen primary contrast was untestable and the positive headline is a secondary contrast the plan marks descriptive | EIC | raised | text: Section 2.4 "That calibration data must represent the deployment inputs is therefore established guidance, not a finding of this paper." | major | 4 |
| SC-2 | R1 W4: confirmatory content of RQ1 is empty; "prespecified" overstates the headline's status; "the issue is labelling" | R1 | disputed (severity) | table: Table 16, rows H1 and H2 ("Untestable as posed") | minor | 4 |
| SC-2 | DA m1: abstract calls the primary contrast null | DA | corroborated (outside count) | text: Abstract "range stratification, the primary contrast, was null" | minor | 4 |
| SC-3 | EIC W2: two papers in one; RQ1 and RQ2 share no claim, and RQ2 fills about half the main text and most of the abstract | EIC | raised | text: Section 9.3 "What the rounds established is narrower than five rounds of selection suggest." | major | 4 |
| SC-4 | EIC W2: RQ2's lesson is presented as general methodology although it rests on a post hoc rejection and is familiar (also Detailed Comments, Structural Coherence) | EIC | raised | text: Section 9.3 "What the rounds established is narrower than five rounds of selection suggest." | major | 4 |
| SC-4 | R2 W1: the lesson is endpoint-relative and "should not be presented as general methodological evidence" | R2 | corroborated | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | major | 4 |
| SC-4 | DA M1: the lesson assumes, without arguing, that directional recall is the right target | DA | corroborated (outside count) | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | major | 4 |
| SC-5 | EIC W2: round five's ship decision is a +0.20-point choice between two artifacts (four factors confounded, one seed per side, two gates failed on both Weinstock components) | EIC | raised | text: Section 9.3 "What the rounds established is narrower than five rounds of selection suggest." | major | 4 |
| SC-5 | DA M3: the gated gain carries no training-seed variance, and cohorts with cleaner provenance point the other way | DA | corroborated (outside count) | text: Appendix B "but that rule could not fail" and "which would distort the rate labels of round five's primary cohort" | major | 4 |
| SC-6 | EIC W3: AI involvement is undisclosed, including an AI-generated review credited as "external"; no AI-use statement | EIC | raised | text: Appendix D "An external review published between rounds four and five" | major | 5 |
| SC-6 | DA M4: same, grounded in the COPE position statement on authorship and AI tools | DA | corroborated (outside count) | text: Appendix D "An external review published between rounds four and five" | major | 4 |
| SC-7 | EIC W4: the abstract spends its space on round history and omits the accuracy illusion | EIC | raised | text: Abstract "Round five's seed-4 artifact (flatten head, cosine schedule, validation-chosen epoch), now bundled in the research applications" | minor | 4 |
| SC-8 | EIC W5: excess length; the round-five caveat recurs almost verbatim in six places | EIC | raised | text: Section 12 "a fifth of round three's 1.1-point float seed spread on Weinstock held-out (its own RT-CGM seed spread was not measured)" | minor | 4 |
| SC-9 | EIC W6: following RQ2 requires moving between four locations | EIC | raised | table: Table 9 (confirmatory reads), caption identifying champions only by reference to appendix Table 12 | minor | 4 |
| SC-10 | EIC W7: "in 8-bit integers" in the abstract and contribution bullets overstates the converted graph | EIC | raised | text: Abstract "For a 2,909-parameter convolutional network in 8-bit integers (INT8)" | minor | 3 |
| SC-10 | R3 W1: the minimum remedy rescopes the abstract's "network in 8-bit integers (INT8)" and contribution bullet 4 | R3 | disputed (severity) | text: §7.9 "The artifact is also less integer than its name" | major | 4 |
| SC-10 | DA m2: the abstract overstates how integer the graph is | DA | corroborated (outside count) | text: Section 7.9 "The artifact is also less integer than its name" | minor | 4 |
| SC-11 | EIC W7: the deployment-cost findings are properties of this converter configuration, and the paper does not say why the convolutions stay in float | EIC | raised | text: Abstract "For a 2,909-parameter convolutional network in 8-bit integers (INT8)" | minor | 3 |
| SC-11 | R3 W1: the float convolutions come from LiteRT Torch 0.9.1's conv1d lowering; the INT8 deployment conclusions inherit it | R3 | disputed (severity) | text: §7.9 "The artifact is also less integer than its name" | major | 4 |
| SC-12 | EIC W8: prior work on compression's hidden class-level damage is uncited | EIC | raised | absence: Sections 2 and 9.2 — expected prior work showing that compression harms minority classes or atypical examples while aggregate accuracy is preserved; checked Sections 2.3, 2.4, 9.1, 9.2 and all 68 bibliography entries | minor | 4 |
| SC-13 | EIC W9: the read ledger is not positioned against the holdout-reuse literature | EIC | raised | absence: Sections 2.2 and 9.3 — expected positioning against the adaptive data analysis and holdout-reuse literature; checked Sections 2.2, 6.2, 9.3, 10, Table 13 and all 68 bibliography entries | minor | 4 |
| SC-14 | EIC W10: no statement on health prediction-model reporting norms | EIC | raised | absence: Section 11 and Declarations — expected a reporting-guideline statement such as a TRIPOD+AI checklist or a stated reason it does not apply; checked Sections 11.1, 11.2, Declarations and Appendices A to D | minor | 3 |
| SC-15 | EIC W11: an unevaluated on-device language-model caption is described | EIC | raised | text: Section 5.3 "the captions were not evaluated" | minor | 4 |
| SC-16 | EIC W12: the headline figure shows means without variability | EIC | raised | figure: Figure 2 (per-class held-out recall) | minor | 4 |
| SC-17 | R1 W1: gate outcomes in rounds two and three are seed-determined, and seed robustness is reported for gate 1 only | R1 | raised | text: §8.2 "The effect did not hinge on the chosen seed" | major | 5 |
| SC-18 | R1 W2: round five's RT-CGM gain is a slow-to-fast reallocation with a macro-F1 loss of -1.65 [-1.75, -1.57]; "passed all gates" omits it | R1 | raised | text: §6.2 "The two cases that arose fit one rule" | major | 4 |
| SC-18 | R2 W1: the bundled candidate has lower QWK and lower falling_fast precision than the champion on all four components, RT-CGM included | R2 | corroborated | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | major | 4 |
| SC-18 | DA M2: round five was shipped despite an operating-point-shift diagnosis and a macro-F1 loss on every paired split | DA | corroborated (outside count) | text: Section 6.2 "No protocol stated when a gate-passing candidate may be overridden. The two cases that arose fit one rule" and Section 8.5 "consistent with an operating-point shift from the slow to the fast classes rather than better discrimination" | major | 4 |
| SC-19 | R1 W2: the reconciling override rule was stated after both outcomes and is phrased so that round five falls outside it | R1 | raised | text: §6.2 "The two cases that arose fit one rule" | major | 4 |
| SC-19 | DA M2: the post hoc rule applies one diagnosis in opposite directions | DA | corroborated (outside count) | text: Section 6.2 "No protocol stated when a gate-passing candidate may be overridden. The two cases that arose fit one rule" | major | 4 |
| SC-20 | R1 W3: the sequential arm is a single unreplicated selection | R1 | raised | text: §5.2 "(participant, segment, time) order, as the project's original conversion" | minor | 4 |
| SC-20 | DA m4: the 13.4-point magnitude comes from one deterministic ordering | DA | corroborated (outside count) | table: tab:contrasts, row uniform random vs. sequential, 200 | minor | 3 |
| SC-21 | R1 W5: decision thresholds and frozen predictions lack an operational basis; gate 1 is called "beats" | R1 | raised | text: §6.2 "beats the champion's INT8 model on the endpoint" | minor | 4 |
| SC-22 | R1 W6: the 20-cluster percentile bootstrap is not checked against a small-cluster method | R1 | raised | text: §8.3 "first fully positive interval on the directional endpoint" | minor | 4 |
| SC-23 | R1 W7: the round-three freeze cannot be verified from repository history | R1 | raised | dataset: git history of paper/results/calibration-coverage/protocol_sweep_r3.json (only commit a84b237, shared with accuracy_sweep_r3.json) | minor | 5 |
| SC-24 | R1 W8: the comparator champion was selected on the same held-out data | R1 | raised | text: §7.8 "rule advanced the higher-scoring variant, so the anchor variant was chosen" | minor | 3 |
| SC-24 | DA m7: champion selection on held-out data tilts rounds one and two toward the champion | DA | corroborated (outside count) | text: Section 7.8 "so the anchor variant was chosen on held-out data" | minor | 3 |
| SC-25 | R1 W9: the no-effect statements rely on point estimates | R1 | raised | text: §1 "budget, and the observer choice move recall by at most about a tenth of" | minor | 4 |
| SC-26 | R2 W1: QWK and per-class precision reverse round one's rejection; the class weights imply an unstated cost of about 24:1 | R2 | raised | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | major | 4 |
| SC-26 | DA M1: the rejected round-one candidate has higher precision in all four directional classes and the highest held-out macro-F1 | DA | corroborated (outside count) | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | major | 4 |
| SC-27 | R2 W1: the stable-logit offset scan listed in Table 13 has no reported result | R2 | raised | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | major | 4 |
| SC-27 | DA M2: the offset scan's outcome is not reported | DA | corroborated (outside count) | text: Section 6.2 "No protocol stated when a gate-passing candidate may be overridden. The two cases that arose fit one rule" | major | 4 |
| SC-28 | R2 W2: INT8 conversion of the bundled model raises its falling_fast call rate by 45-72%; the recall-based fidelity gate misses it; §9.3's attribution and §9.2's recommendation need correcting | R2 | raised | text: Section 9.2 "Directional (minority-class) recall or five-class macro recall exposes it immediately" | major | 4 |
| SC-29 | R2 W3: "worse directional classifier" overstates RQ1's degradation; much of the magnitude is an operating-point shift | R2 | raised | text: Section 9.2 "The sequential-arm INT8 model is a worse directional classifier with better accuracy" | minor | 4 |
| SC-29 | DA m3: the accuracy-illusion takeaway holds only where the float operating point under-predicts the majority class | DA | corroborated (outside count) | text: Section 9.2 "any pipeline that gates a quantized model on raw accuracy would have promoted the broken artifact" | minor | 3 |
| SC-30 | R2 W4: baselines are compared at mismatched operating points; no forecaster-derived baseline | R2 | raised | text: Section 7.5 "On held-out participants the five-seed float CNN reaches 0.487 directional recall against 0.278 for the recent-slope baseline" | minor | 4 |
| SC-31 | R2 W5: the RT-CGM description contains verifiable errors (composition, wear mode, cadence) | R2 | raised | table: Table 11, RT-CGM row (n, Age and CGM; wear columns) | minor | 4 |
| SC-32 | R2 W6: the label-noise ceiling and differential sensor noise are not quantified | R2 | raised | text: Section 8.4 "White noise of this size is also gentler than published CGM error models" | minor | 3 |
| SC-33 | R2 W7: literature on operating points, label shift and ordinal evaluation is missing | R2 | raised | absence: Section 2.3 and the bibliography — expected literature on post-hoc operating-point adjustment, label shift and ordinal evaluation metrics; checked Sections 2.3, 9.2, 9.3, 10 and all 68 bibliography entries | minor | 4 |
| SC-34 | R3 W2: the microbenchmark measures per-call overhead at timer resolution | R3 | raised | table: Table 19 (tab:devicebench), Appendix C.6 | minor | 4 |
| SC-35 | R3 W3: the fixed-range remedy does not transfer to the delta+anchor encoding | R3 | raised | text: §9.1 "fixing its range from the sensor's reporting limits" | minor | 4 |
| SC-35 | DA m6: the bundled artifact clips 2.8% of RT-CGM windows, in tension with RQ1's own rule | DA | corroborated (outside count) | text: Section 8.5 "That gap includes any cost of input clipping" | minor | 4 |
| SC-36 | R3 W4: Android parity coverage of the post-July bundled artifacts is overstated | R3 | raised | text: §8.2 "after golden parity passed in the Android unit tests on the Java virtual" | minor | 4 |
| SC-37 | R3 W5: golden-vector coverage is thin for a "bit-exact" claim | R3 | raised | text: §C.7 "selected from each float-predicted class and including values above the" | minor | 4 |
| SC-38 | R3 W6: the INT8 deployment target is unstated, and the measured phones favour float | R3 | raised | text: §5.3 "every complete window locally with the float artifact by default or INT8" | minor | 4 |
| SC-39 | DA M3: the ship decision rests on one gated cohort whose label validity the paper doubts, and the summaries leave the doubt out | DA | raised (DA-tracked) | text: Appendix B "but that rule could not fail" and "which would distort the rate labels of round five's primary cohort" | major | 4 |
| SC-40 | DA m5: the noise reference for round five's +0.20 is not comparable with it | DA | raised (DA-tracked) | text: Introduction "a fifth of round three's 1.1-point float seed spread on Weinstock held-out" | minor | 4 |

### Step 1c: Surface-form parity

Each sub-claim above was assessed against its manuscript anchor, not against its phrasing. Every quoted anchor was found in `paper/main.tex`, and the Declarations section was read to confirm the absence behind SC-6. No sub-claim was marked unevaluable. Technical precision (R3's operator counts, R2's QWK values, R1's receipts) was not counted as corroboration; each was treated as evidence to be read against the manuscript.

### Step 2: Consensus counts (denominator: the four non-DA seats)

| Sub-claim | agree | conflict | silent | Disposition |
|---|---|---|---|---|
| SC-4 | 2 (EIC, R2) | 0 | 2 (R1, R3) | corroborated finding |
| SC-18 | 2 (R1, R2) | 0 | 2 (EIC, R3) | corroborated finding |
| SC-2 | 1 (EIC) | 1 (R1, severity) | 2 | SPLIT |
| SC-10, SC-11 | 1 (EIC) | 1 (R3, severity) | 2 | SPLIT |
| SC-1, SC-3, SC-5 to SC-9, SC-12 to SC-17, SC-19 to SC-38 | 1 each | 0 | 3 each | single-reviewer findings |
| SC-39, SC-40 | 0 | 0 | 4 | DA-tracked (outside the consensus count) |

No weakness sub-claim reaches CONSENSUS-4 or CONSENSUS-3. That follows from the panel design: each seat covers a separate remit, and silence is not disagreement.

---

## Part 1: Editorial Decision Letter

Dear Author,

Thank you for submitting your manuscript titled "GlucoEdge: Quantization-Calibration Coverage and Staged Model Selection in On-Device Glucose Trend Classification" for pre-submission review. No target venue was bound for this round (`criteria_binding_unavailable`), so this letter makes no venue-fit judgement. Five role-separated review seats reviewed the manuscript, including a Journal-Fit Reviewer role. Their execution provenance was recorded and replay-validated, and is reported above. The seats were role-separated, each ran in its own invocation context, and none saw another seat's output. All five used one model family, so correlated errors cannot be ruled out, and the panel's provenance is not reduced to a binary independence claim.

### Decision: Major Revision

### Consensus Analysis

#### Points of Agreement

**On strengths (not roadmap items):**
- All four scoring seats and the DA judge the RQ1 result robust and correctly attributed to its mechanism. R1 reproduced +13.38 [+10.29, +16.88] from the published counts and found the effect at every seed (13.05 to 14.14, SD 0.47). The clip-only control accounts for 13.14 of the 13.35 points. R2 found that the direction survives a chance-corrected check at every seed (EIC S2, R1 S1-S2, R2 W3 and Summary, R3 S1, DA strengths).
- All four scoring seats and the DA found the headline numbers consistent across the abstract, body, tables and result files (EIC criterion row on numeric consistency, R1 verification, R2 S3, R3 S4, DA score rationale).
- Three seats single out the published per-participant counts as an unusually strong reproducibility asset (EIC S4, R1 S5, R2 S6; R3 silent).

**On weaknesses:**
- No weakness reached [CONSENSUS-4] or [CONSENSUS-3].
- **Corroborated by two seats (SC-4):** the RQ2 "main lesson" is presented as general methodology beyond what the evidence supports. The EIC calls it "a single-case observation with a post hoc decision at its centre", and R2 shows that it is endpoint-relative. The DA (M1) concurs. R1 and R3 are silent.
- **Corroborated by two seats (SC-18):** round five's primary RT-CGM gain is a reallocation from slow to fast classes, with lower class-balanced performance. R1 measures it as macro-F1 -1.65 [-1.75, -1.57]; R2 measures lower QWK and falling_fast precision on all four components. The "passed all gates" summaries leave this out. The DA (M2) concurs. The EIC and R3 are silent.

#### Points of Disagreement

**Disagreement 1: evidential status of the RQ1 headline (SC-2)**
- **EIC view (W1, Major)**: the primary contrast was untestable as posed, and the positive headline is a secondary contrast, added by amendment, that the plan marks descriptive. Together with the conceded rule, this narrows RQ1's contribution.
- **R1 view (W4, Minor)**: the confirmatory content is indeed empty and "prespecified" overstates the headline's status, but "the effect's credibility does not depend on this... The issue is labelling."
- **DA (m1)**: the abstract calls the primary contrast "null", but the body classifies it as untestable as posed.
- **Disagreement type**: severity disagreement.
- **Editor's Resolution**: Minor. The required action is labelling, carried as S1. The EIC's separate contribution sub-claim, SC-1, is unaffected and stays Major (R1).
- **Resolution Rationale**: the arbitration falls to the editorial synthesizer, because the Journal-Fit Reviewer is a party to this split. The confirmatory-versus-exploratory status of a statistical result belongs to the methodology remit (expertise first), and R1 holds that criterion under D1. R1 also supplies the evidence: an exact reproduction of the interval, per-seed effects, and the clip-only control. Both seats agree that the problem exists. They differ only on how much it weighs, and the EIC's contribution concern stands on its own §2.4 anchor.

**Disagreement 2: the hybrid float/int8 graph and the INT8 deployment claims (SC-10, SC-11)**
- **EIC view (W7, Minor, confidence 3)**: the wording "in 8-bit integers" overstates the graph, and the deployment findings are properties of the converter configuration. The EIC adds that "whether the hybrid graph comes from configuration is for the systems seat."
- **R3 view (W1, Major)**: inspection of the committed artifacts, plus a probe that reuses the repository's `convert_int8`, shows that LiteRT Torch 0.9.1 lowers `aten.conv1d` with a separate float bias add. This prevents quantized-convolution fusion, whereas a Conv2d(1x3) re-expression produces a 9-operator integer graph. "INT8 is slower" and "no measured speed benefit" therefore describe this lowering path, not 8-bit inference of the model, and integer-only targets cannot run the graph.
- **DA (m2)**: the abstract overstates how integer the graph is.
- **Disagreement type**: severity disagreement.
- **Editor's Resolution**: Major, carried as R5.
- **Resolution Rationale**: expertise first, because toolchain behaviour is R3's remit and the EIC explicitly deferred to that seat. Evidence first also favours R3, whose artifact inspection turns "cause unexplained" into a scope limit that applies to every INT8 deployment statement. R3's own scope limits carry over: the probe used a random-weight model, not the study checkpoints, and nothing was timed on a device. R3 also reports that the RQ1 accuracy findings very likely transfer (argmax agreed on all 5,000 probe inputs), so this resolution concerns the deployment-cost claims, not RQ1's accuracy result.

**Remedy coordination (not SPLITs; author autonomy applies)**
- **Cutting RQ2 versus reporting more on it.** The EIC (W2) asks for RQ2 to be moved out or compressed. R1 (W1, W2), R2 (W1) and the DA (M1-M3) ask for more RQ2 reporting. These requests are compatible. If RQ2 stays in this manuscript in any form, R6-R9 and R11 apply in full; their tables may go to a supplement, consistent with EIC W5. If RQ2 moves to a separate report, those items move with it, and this manuscript drops RQ2's claims from the title, abstract and Conclusion (EIC, Title & Abstract). R3 (the RQ2 lesson) and R10 (§9.2 and §9.3, about the bundled artifact) apply to whichever document carries those passages.
- **Choosing RQ1's broadening evidence.** EIC W1 lists three options. If the author picks the fixed-input-range remedy, R3 W3 constrains it (S22): the remedy targets absolute inputs, and the deployed delta+anchor encoding needs either a float input interface or two graph inputs. R3 W1 describes the Conv2d(1xk) full-integer reconversion as cheap, which makes it a candidate for the "fully integer graph" option.
- **Rewriting the abstract.** EIC W4 (S2) asks to lead with the accuracy illusion. R2 W3 (S16) asks to frame it as loss of fidelity to the float model under the declared endpoint, with the operating-point condition from DA m3. These fit together in one rewrite.
- **R3 S6 and R2 W2 do not conflict.** R3 credits the minority-recall gate for catching RQ1's damage, which R2 also accepts. R2 W2 concerns a different shift, in the bundled pair, that R3 did not examine.

#### Devil's Advocate Critical Issues

The DA's CRITICAL table is empty, so there is no DA-CRITICAL item to adjudicate and nothing in this category can block or change the decision. For visibility, the four DA MAJOR findings map as follows:
- M1 to SC-4 and SC-26 (corroborated by EIC W2 and R2 W1).
- M2 to SC-18, SC-19 and SC-27 (corroborated by R1 W2 and R2 W1).
- M3 to SC-5 (corroborated by EIC W2) and SC-39 (DA-only, carried as R11).
- M4 to SC-6 (corroborated by EIC W3).

The DA graded D3 as if RQ2 were central and noted that, if RQ2 were secondary, the same findings would describe overreach instead. The manuscript as it stands presents RQ2 as central: it takes half the main text and most of the abstract (EIC W2, W4), so the DA's premise holds. If the author removes or compresses RQ2 (R2), the same findings apply at that reduced scope.

### Decision Rationale

The decision follows mechanically from contract `reviewer/reviewer_full/v2`. Three mandatory dimensions carry a repairable block: domain accuracy (D2, R2), argumentative coherence (D3, DA, with R1 at warn), and venue fit and contribution (D6, EIC). Condition F2 (severity 90) therefore fires and sets Major Revision. F3 and F5 also fire, at lower severity. F1 does not fire because no seat declared a fatal block, and F4 does not fire because the one high-priority dimension, D4, scored warn. The DA raised no CRITICAL item.

The reviewer cards agree with the arithmetic. Three of the four scoring seats recommend Major Revision. The fourth, R3, recommends Minor Revision from its deployment remit. That is the "Minor, Major, Major, Major" row of the decision standards. Every seat that examined RQ1 found its central result robust and correctly localized.

The blocks come from three sources:
- The RQ2 lesson and the round-five adoption are stated more firmly than the per-seed, per-class and class-balanced evidence supports (EIC W2; R1 W1-W2; R2 W1-W2; DA M1-M3).
- RQ1's contribution is narrow relative to guidance the paper itself calls established (EIC W1).
- AI involvement is undisclosed, including an AI-generated review credited as "external" (EIC W3, DA M4).

The decision is not Reject because every block is declared repairable and the seats name concrete remedies. R1 and R2 state that their RQ2 analyses need no retraining, only the published counts or cached outputs. The decision is not Minor Revision because three mandatory-dimension blocks remain and the required changes reach the framing of both research questions, not only clarification.

The panel had no clinical or regulatory seat. The DA's stakeholder questions, about over-calling during stable glucose and about the regulatory position of distributed apps, are recorded but were not adjudicated. The absence of such a seat should not be read as clearance.

### Blocking Issues (0-3, immutable source order)

These are the three required items whose resolution bears on the most block-scored dimensions. All eleven required items must still be addressed.

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| R3 | The RQ2 "main lesson" is stated as general methodology beyond its single-case, endpoint-relative evidence (bears on D6, D2, D3) | EIC, R2 (DA M1) | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | REV-SC-4 |
| R4 | AI involvement is undisclosed, and an AI-generated review is credited as "external" (bears on D6) | EIC (DA M4) | text: Appendix D "An external review published between rounds four and five" | REV-SC-6 |
| R7 | Round five's "passed all gates" adoption omits a slow-to-fast reallocation with class-balanced loss on the primary cohort (bears on D2, D3) | R1, R2 (DA M2) | text: §6.2 "The two cases that arose fit one rule" | REV-SC-18 |

### Closing

We encourage you to consider the reviewers' comments carefully and to submit a substantially revised manuscript. The revised manuscript will be reviewed again, with each required item checked against its acceptance criterion.

---

## Part 2: Revision Roadmap

Obligation rule: `obligation_class` follows the transported severity, or the arbitrated severity for a SPLIT. Major gives `must_fix` and minor gives `should_fix`. `consider` is used only where the source seat made the remedy conditional on an author choice, and for editorial minor issues that carry no severity. Rows are in immutable source order: seat order EIC, R1, R2, R3, DA, then finding ordinal, then sub-claim ordinal. This order is not a work order.

Machine core: the closed `revision-roadmap/1.0` artifact is **not emitted** in this package. No block manifest is bound, so `block_manifest_sha256` and the exact `proposed_targets[]` block IDs cannot be filled without fabricating them. Targets below are given as section, table and claim locators. The base draft SHA-256 is recorded under Manuscript Information. With a bound manifest, `scripts/revision_roadmap.py` builds the machine core from these rows.

> The `Sub-Claim(s)` column carries the Step 1b `sub_claim_id`(s) each item traces to.

### Required Revisions (Must Fix)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| R1 | Reframe RQ1 as a pitfall-and-audit contribution and add one broadening analysis from the EIC's options | SC-1 | major | text: Section 2.4 "That calibration data must represent the deployment inputs is therefore established guidance, not a finding of this paper." | 4: generalist ML-methodology editor; contribution judgement | EIC | must_fix | re_analysis: abstract, §1, §9.1, plus one broadening analysis | claim_scope_unsupported: claim "RQ1 contribution" |
| R2 | Resolve the two-papers structure: move RQ2 out, or compress it to about one page as an evaluation-governance case study; describe round five's bundling as a two-artifact choice | SC-3, SC-5 | major | text: Section 9.3 "What the rounds established is narrower than five rounds of selection suggest." | 4: structural and contribution judgement | EIC (DA M3 corroborates SC-5) | must_fix | section: RQ2 (§6.2, §8, §9.3), abstract, title | editorial_conformance_unmet: manuscript "RQ1/RQ2 structure" |
| R3 | Restate the RQ2 "main lesson" as endpoint disagreement in a single case under a declared preference, with the round-one decision labelled post hoc | SC-4 | major | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | 4 (EIC: contribution judgement); 4 (R2: recomputed from published counts) | EIC, R2 (DA M1) | must_fix | sentence: §9.3, abstract, Conclusion | claim_scope_unsupported: claim "§9.3 sweep lesson" |
| R4 | Add an AI-use statement; describe the "external review" accurately (its nature, the labels it read, which proposals were adopted); explain or rename the panel and re-review ledger artifacts | SC-6 | major | text: Appendix D "An external review published between rounds four and five" | 5: commit message and trailers read directly | EIC (DA M4) | must_fix | section: Declarations, Appendix D, Limitation 5, Table 13 | reporting_requirement_unmet: section "Declarations" |
| R5 | State the cause of the hybrid graph (conv1d lowering, converter version) and scope every INT8 latency and "integer" statement to it; describe the artifact as hybrid in the abstract | SC-10, SC-11 | major (arbitrated; EIC minor, R3 major) | text: §7.9 "The artifact is also less integer than its name" | 4: artifact inspection and toolchain probe, not on the study checkpoints (R3); 3 (EIC) | EIC, R3 (DA m2) | must_fix | sentence: abstract, contribution bullet 4, §7.9, Limitation 8, Conclusion | claim_scope_unsupported: claim "INT8 deployment statements" |
| R6 | Report per-seed gate outcomes and seed-averaged gates 2 and 3 for rounds two and three; reword round two as an artifact-level pass the recipe does not reproduce | SC-17 | major | text: §8.2 "The effect did not hinge on the chosen seed" | 5: recomputed with the repository's bootstrap and frozen seeds | R1 | must_fix | re_analysis: §8.2, §8.3, Discussion, Conclusion (published counts, no retraining) | claim_scope_unsupported: claim "round two 'most consistent result'" |
| R7 | Report RT-CGM champion-versus-candidate per-class recall and precision and the macro-F1 interval; state the slow-to-fast reallocation wherever "passed all gates" appears | SC-18 | major | text: §6.2 "The two cases that arose fit one rule" | 4 (R1: recomputed from counts); 4 (R2: recomputed from counts) | R1, R2 (DA M2) | must_fix | re_analysis: §8.5, Table 22, abstract, Conclusion | claim_scope_unsupported: claim "'passed all gates'" |
| R8 | Label the override rule post hoc; justify it on a principle that does not reference the two outcomes, or present it as a preference ranking directional recall first | SC-19 | major | text: §6.2 "The two cases that arose fit one rule" | 4: recomputed figures; the neutrality judgement is interpretive | R1 (DA M2) | must_fix | sentence: §6.2, §9.3 | interpretive_ambiguity_remains: section "§6.2" |
| R9 | State the cost structure the endpoint and class weights imply; report QWK (or a declared cost-matrix metric) and per-class precision beside every confirmatory directional-recall read; report the stable-logit offset-scan result | SC-26, SC-27 | major | text: Section 9.3 "The sweep program's main lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing" | 4: recomputed from published counts; no logits available | R2 (DA M1, M2) | must_fix | re_analysis: Tables 9 and 20, §9.3 | evidence_gap_remains: table "confirmatory reads (Tables 9, 20)" |
| R10 | Report float and INT8 profiles of the bundled pair side by side; correct §9.3's attribution of over-calling; qualify §9.2's recall-based fidelity recommendation | SC-28 | major | text: Section 9.2 "Directional (minority-class) recall or five-class macro recall exposes it immediately" | 4: computed from published float and INT8 counts | R2 | must_fix | re_analysis: Tables 22 and 23, §9.2, §9.3 | claim_scope_unsupported: claim "§9.2 recommendation; §9.3 attribution" |
| R11 | State the Appendix B label-validity doubt about RT-CGM wherever the round-five result is summarized; either run the five-seed RT-CGM comparison or present the bundling as a choice not backed by evidence of improvement | SC-39 | major | text: Appendix B "but that rule could not fail" and "which would distort the rate labels of round five's primary cohort" | 4: appendix statements and audit verdicts checked | DA | must_fix | sentence: abstract, Introduction, Conclusion | claim_scope_unsupported: claim "round-five summaries" |

### Required Item Details

> **Ordinal contract (#576/#670):** `R<n>` is a transport reference, never a work rank. Numbering follows the roadmap's source-traceability order filtered to `obligation_class == must_fix`.

**R1: Reframe RQ1's contribution**
- **Problem**: the rule RQ1 supports is conceded as established guidance (§2.4). What is new is narrow: the contiguous-selection trap for overlapping windows, its size, and the clipping mechanism. The nulls are confined to a saturated regime.
- **Source**: EIC W1 (a driver of the D6 block).
- **Requirement**: reframe RQ1 as a pitfall-and-audit contribution: report the calibration set's input range against the deployment range, and gate on minority-class metrics. Then add one piece of broadening evidence from the EIC's list: a regime where random draws under-cover the range, a fully integer graph, or the fixed-input-range remedy (constrained by S22).
- **Acceptance criteria**: the abstract, Introduction and §9.1 state the contribution as the pitfall and its audit rule, and one broadening analysis from the EIC's listed options is reported with its scope.

**R2: Resolve the two-papers structure**
- **Problem**: RQ1 and RQ2 share a model family and a split but not a claim. RQ2 takes about half the main text and most of the abstract. Round five's ship decision is a +0.20-point choice between two artifacts, with four factors confounded, one seed per side, and two gates failed on both Weinstock components.
- **Source**: EIC W2. DA M3 corroborates the narrow evidential basis.
- **Requirement**: move RQ2 to a separate report or a supplement, or compress it to about one page as an evaluation-governance case study positioned against holdout-reuse work (S6). In either form, describe round five's bundling as a choice between two artifacts, not as evidence of a design improvement.
- **Acceptance criteria**: the main text carries RQ2 only in compressed form, or not at all with the title adjusted, and the abstract and Conclusion describe the round-five adoption as a choice between two artifacts.

**R3: Restate the RQ2 "main lesson"**
- **Problem**: §9.3 presents the endpoint lesson as general methodology. The EIC reads it as a single-case observation with a post hoc decision at its centre. R2 shows that its verdicts depend on an unstated cost choice.
- **Source**: EIC W2 and Detailed Comments; R2 W1; DA M1.
- **Requirement**: restate the lesson as two endpoints that disagree in one case, settled by a stated preference whose cost is shown, with the round-one decision labelled post hoc.
- **Acceptance criteria**: §9.3, the abstract and the Conclusion no longer present the sweep lesson as general methodology, and the restated lesson names both endpoints, the declared preference between them, and the post hoc status of the round-one decision.

**R4: AI-use disclosure**
- **Problem**: Appendix D, Limitation 5 and the read ledger credit an "external review". According to the repository record, both seats cite, an AI agent produced it. That review read Weinstock held-out and temporal labels and proposed round five's flatten head, schedule and epoch selection. The Declarations cover funding, competing interests, CRediT and licences, but contain no AI-use statement.
- **Source**: EIC W3; DA M4 (COPE position statement on authorship and AI tools).
- **Requirement**: add an AI-use statement covering code, analysis, writing and the simulated reviews. Replace "external review" with an accurate description that states what labels it read and which of its proposals were adopted. Explain or rename the panel and re-review entries in the read ledger.
- **Acceptance criteria**: the Declarations contain an AI-use statement, no passage calls the AI-generated review "external", and Appendix D and Table 13 state what it read and which of its proposals were adopted.

**R5: Scope the INT8 deployment claims to the conv1d lowering**
- **Problem**: the abstract describes "a network in 8-bit integers", but only the final fully connected layer runs in int8. R3 traced the float convolutions to LiteRT Torch 0.9.1's conv1d lowering. The latency and benefit conclusions therefore describe that lowering, and integer-only targets cannot run the graph.
- **Source**: EIC W7; R3 W1; DA m2. A SPLIT, arbitrated to Major.
- **Requirement**: state the cause and the converter version. Scope every INT8 latency and "integer" statement to this lowering (abstract, contribution bullet 4, §7.9, Limitation 8, Conclusion), and describe the artifact as hybrid in the abstract. Optional, per R3's stronger remedy: reconvert the study checkpoints in Conv2d(1xk) form with the same 20 selections, report agreement with the published INT8 predictions, and time the result in the existing benchmark.
- **Acceptance criteria**: the abstract describes the artifact as hybrid, §7.9 names the lowering cause and the converter version, and no deployment-cost statement is phrased as a property of 8-bit inference of the model.

**R6: Seed-determined gate outcomes**
- **Problem**: in round two, only the chosen seed passes all three gates, and the five-seed mean stable change (-2.76 points) fails gate 2. In round three, only a non-chosen seed passes. "The effect did not hinge on the chosen seed" holds for the gate-1 point estimate only.
- **Source**: R1 W1.
- **Requirement**: add a per-seed gate table for every round whose seeds were evaluated. Report seed-averaged versions of gates 2 and 3 beside the artifact-level ones. Reword the round-two subsection, the Discussion and the Conclusion to describe an artifact-level pass that the recipe does not reproduce.
- **Acceptance criteria**: a per-seed gate table for rounds two and three is reported, and every round-two claim separates the artifact-level pass from the recipe-level result.

**R7: Round five's RT-CGM reallocation**
- **Problem**: on RT-CGM, falling_fast recall rises 9.7 points while falling recall drops 8.8. Falling_fast precision falls from 0.278 to 0.214, and macro-F1 changes by -1.65 [-1.75, -1.57] (R1). QWK and falling_fast precision are lower on all four components (R2). The summaries say "passed all gates" without any of this.
- **Source**: R1 W2; R2 W1; DA M2.
- **Requirement**: report the RT-CGM per-class recall and precision for champion and candidate, and the macro-F1 interval. State in the abstract and Conclusion that the gain reflects reallocation from slow to fast classes.
- **Acceptance criteria**: the RT-CGM per-class comparison and the macro-F1 interval appear in the results, and every "passed all gates" summary sentence states the reallocation and the macro-F1 change.

**R8: The post hoc override rule**
- **Problem**: the rule that "fit[s]" both cases was stated after both outcomes were known. It is phrased so that round five's primary-component macro-F1 loss falls outside it, whereas an outcome-neutral rule would have overridden both rounds.
- **Source**: R1 W2; DA M2.
- **Requirement**: label the rule post hoc. If the intended principle is that macro-F1 was pre-declared as non-gating, justify the rule on that basis. Before any future read, freeze an override or harm-bound rule on a named secondary metric.
- **Acceptance criteria**: §6.2 labels the rule post hoc, and either justifies it on a stated principle that does not refer to the two outcomes or presents it as a preference that ranks directional recall first.

**R9: Ordinal and precision reporting, the cost structure, and the offset scan**
- **Problem**: recomputed QWK and precision reverse round one's rejection (held-out QWK 0.439 against 0.385) and favour the champion over the round-five candidate on every component. The inverse-frequency weights imply an error-cost ratio of about 24:1 that the paper never states. Table 13 lists a stable-logit offset scan whose result is never reported.
- **Source**: R2 W1; DA M1, M2.
- **Requirement**: state the implied cost structure. Report QWK, or an R-EGA-style cost matrix, and per-class precision beside every confirmatory directional-recall read, and report the offset scan's outcome. R2's stronger option is a matched-operating-point comparison by post-hoc logit adjustment, plus per-class AUROC and AUPRC from cached logits.
- **Acceptance criteria**: every confirmatory directional-recall read has QWK or a declared cost-matrix metric and per-class precision beside it, the implied cost ratio is stated, and the offset-scan result is reported.

**R10: INT8 conversion of the bundled model**
- **Problem**: conversion raises the falling_fast call rate (held-out 11.0% to 17.8%, temporal 8.5% to 14.6%, RT-CGM 5.3% to 7.6%) and lowers precision. Its QWK cost equals that of the headline calibration failure, yet directional recall moves by 0.5 points or less, so gate 3 passes. §9.3 attributes the over-calling only to the class weights and the endpoint, and §9.2 says recall "exposes it immediately".
- **Source**: R2 W2.
- **Requirement**: show float and INT8 profiles side by side in Tables 22 and 23. Correct the §9.3 attribution and qualify the §9.2 recommendation. For future gates, gate INT8 fidelity on per-class prediction rates or window-level agreement with the float model.
- **Acceptance criteria**: Tables 22 and 23 show float and INT8 profiles for the bundled pair, §9.3 names conversion as a contributor to over-calling, and §9.2 states that directional recall alone did not detect this shift.

**R11: RT-CGM label validity in the round-five summaries**
- **Problem**: Appendix B readmits RT-CGM by a rule that "could not fail" and reports a collinear-triple excess that "would distort the rate labels of round five's primary cohort". The abstract and Conclusion describe RT-CGM only as never evaluated.
- **Source**: DA M3. R2 W5 (S18) bears on the same cohort.
- **Requirement**: state the label-validity doubt wherever the round-five result is summarized. Then either run the five-seed RT-CGM comparison or present the bundling as a choice that is not backed by evidence of improvement.
- **Acceptance criteria**: the abstract, Introduction and Conclusion mention the Appendix B label-validity doubt wherever they cite the RT-CGM gain.

### Suggested Revisions (Should Fix / Consider)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| S1 | State in the abstract and Conclusion that the RQ1 headline is a secondary contrast marked descriptive in the plan and previously observed on these participants; replace "null" with "untestable as posed"; note in §6.1 that the coverage audit could have served as a pre-freeze manipulation check | SC-2 | minor (arbitrated; EIC major, R1 minor) | table: Table 16, rows H1 and H2 ("Untestable as posed") | 4: protocol file, plan commit and revision history (R1); 4 (EIC) | EIC, R1 (DA m1) | should_fix | sentence: abstract, Conclusion, §6.1 | claim_scope_unsupported: claim "abstract 'prespecified' / 'null'" |
| S2 | Restructure the abstract to lead with the trap, its size, the accuracy illusion and the mechanism, giving RQ2 at most two sentences (framed per S16) | SC-7 | minor | text: Abstract "Round five's seed-4 artifact (flatten head, cosine schedule, validation-chosen epoch), now bundled in the research applications" | 4: editorial reading of the abstract against the body | EIC | should_fix | sentence: abstract | reader_traceability_reduced: section "Abstract" |
| S3 | State each caveat once, in Limitations; move the bundling history, historical device observations and per-round narrative to a supplement; shorten contribution bullet 6 | SC-8 | minor | text: Section 12 "a fifth of round three's 1.1-point float seed spread on Weinstock held-out (its own RT-CGM seed spread was not measured)" | 4: word and table counts measured | EIC | should_fix | section: main text | editorial_conformance_unmet: manuscript "main text length" |
| S4 | If RQ2 stays, add a variant column or a champion-lineage table to the main text | SC-9 | minor | table: Table 9 (confirmatory reads), caption identifying champions only by reference to appendix Table 12 | 4: navigation check of the rendered PDF | EIC | consider | section: §8, Table 9 | reader_traceability_reduced: table "Table 9" |
| S5 | Cite and position against prior work on compression's class-level damage (Hooker et al., arXiv:1911.05248) | SC-12 | minor | absence: Sections 2 and 9.2 — expected prior work showing that compression harms minority classes or atypical examples while aggregate accuracy is preserved; checked Sections 2.3, 2.4, 9.1, 9.2 and all 68 bibliography entries | 4: bibliography searched | EIC | should_fix | sentence: §2.4, §9.2 | interpretive_ambiguity_remains: section "Related Work" |
| S6 | Position the read ledger against the holdout-reuse literature (Dwork et al. 2015; Blum and Hardt 2015) | SC-13 | minor | absence: Sections 2.2 and 9.3 — expected positioning against the adaptive data analysis and holdout-reuse literature; checked Sections 2.2, 6.2, 9.3, 10, Table 13 and all 68 bibliography entries | 4: bibliography and text searched | EIC | should_fix | sentence: §2.2, §9.3 | interpretive_ambiguity_remains: section "Related Work" |
| S7 | Add a reporting-guideline statement (a TRIPOD+AI-style checklist, or a stated reason it does not apply) | SC-14 | minor | absence: Section 11 and Declarations — expected a reporting-guideline statement such as a TRIPOD+AI checklist or a stated reason it does not apply; checked Sections 11.1, 11.2, Declarations and Appendices A to D | 3: applicability to a non-clinical prototype is debatable | EIC | should_fix | sentence: §11 | reporting_requirement_unmet: section "§11" |
| S8 | Remove the unevaluated "Explain" caption from the paper, or report a minimal content audit | SC-15 | minor | text: Section 5.3 "the captions were not evaluated" | 4: scope judgement | EIC | should_fix | sentence: §5.3 | interpretive_ambiguity_remains: section "§5.3" |
| S9 | Add seed-level points or error bars to Figure 2 | SC-16 | minor | figure: Figure 2 (per-class held-out recall) | 4: figure inspection | EIC | should_fix | other (figure): Figure 2 | reporting_requirement_unmet: figure "Figure 2" |
| S10 | Report the distribution of the maximum input over all contiguous 200-window blocks (label-free); optionally add a random-offset contiguous arm | SC-20 | minor | text: §5.2 "(participant, segment, time) order, as the project's original conversion" | 4: design reading | R1 (DA m4) | should_fix | re_analysis: §7 | claim_scope_unsupported: claim "13.4-point magnitude as a property of the practice" |
| S11 | Give a rationale for the -0.5, 2 and 0.5-point thresholds; give gates 2 and 3 interval forms or call them heuristics; describe gate 1 as superiority in point and non-inferiority in interval | SC-21 | minor | text: §6.2 "beats the champion's INT8 model on the endpoint" | 4: protocol files and recomputed seed variability | R1 | should_fix | sentence: §6.2 | interpretive_ambiguity_remains: section "§6.2" |
| S12 | Report a small-cluster-robust companion (sign-flip test or wild cluster bootstrap-t) for the 20-cluster contrasts, and qualify "first fully positive interval" | SC-22 | minor | text: §8.3 "first fully positive interval on the directional endpoint" | 4: jackknife and exact sign-flip enumeration on the published counts | R1 | should_fix | re_analysis: §8.3 | claim_scope_unsupported: claim "first fully positive interval" |
| S13 | Publish the pre-squash round-three commit or a tag, note the squash in Table 13, and use external timestamps for future protocols | SC-23 | minor | dataset: git history of paper/results/calibration-coverage/protocol_sweep_r3.json (only commit a84b237, shared with accuracy_sweep_r3.json) | 5: git log on both files | R1 | should_fix | other (repository): Table 13 and repository tag | method_reproducibility_unresolved: dataset "protocol_sweep_r3.json" |
| S14 | Note in Limitation 5 that the comparator champion was selected on the same held-out data, or add a non-selected reference comparator | SC-24 | minor | text: §7.8 "rule advanced the higher-scoring variant, so the anchor variant was chosen" | 3: bias direction follows from selection; size not estimated | R1 (DA m7) | should_fix | sentence: Limitation 5 | interpretive_ambiguity_remains: section "Limitations" |
| S15 | State the bound the intervals support (about +/-0.2 points) and give the observer arm an interval | SC-25 | minor | text: §1 "budget, and the observer choice move recall by at most about a tenth of" | 4: Table 6 intervals | R1 | should_fix | re_analysis: §1, Table 6 | claim_scope_unsupported: claim "nulls 'at most about a tenth of a point'" |
| S16 | Reframe the "worse directional classifier", "damaged" and "accuracy illusion" wording around fidelity to the float model under the declared cost frame, with DA m3's operating-point condition; report QWK or prediction agreement for the RQ1 arms | SC-29 | minor | text: Section 9.2 "The sequential-arm INT8 model is a worse directional classifier with better accuracy" | 4: per-seed recomputation from published counts | R2 (DA m3) | should_fix | re_analysis: §9.2, Conclusion | claim_scope_unsupported: claim "worse directional classifier" |
| S17 | Tune the slope and logistic-regression thresholds or offsets on validation to the CNN's stable recall; add a thresholded 15-minute GlucoBench forecaster; tune the logistic-regression C | SC-30 | minor | text: Section 7.5 "On held-out participants the five-seed float CNN reaches 0.487 directional recall against 0.278 for the recent-slope baseline" | 4: recomputed from published baseline counts | R2 | should_fix | re_analysis: §7.5 | evidence_gap_remains: table "baselines" |
| S18 | Correct the RT-CGM row and caption (composition, mixed wear, 10-minute cadence of the 96 unscored subjects); report whether the round-five contrast holds within the HbA1c of 7.0% or higher main cohort | SC-31 | minor | table: Table 11, RT-CGM row (n, Age and CGM; wear columns) | 4: registry record and review read directly; 96-subject mapping inferred | R2 | should_fix | re_analysis: Table 11, Appendix B | reporting_requirement_unmet: table "Table 11" |
| S19 | Simulate label flip rates per boundary with a Facchinetti-type sensor-error model, and report the per-cohort share of near-boundary windows | SC-32 | minor | text: Section 8.4 "White noise of this size is also gentler than published CGM error models" | 3: direction standard; size unmeasured | R2 | should_fix | re_analysis: §8.4 | evidence_gap_remains: section "§8.4" |
| S20 | Add the missing literature: operating-point adjustment, label shift, ordinal metrics, precision-recall evaluation under imbalance, cost-sensitive thresholds, and the Cinar et al. (2026) survey | SC-33 | minor | absence: Section 2.3 and the bibliography — expected literature on post-hoc operating-point adjustment, label shift and ordinal evaluation metrics; checked Sections 2.3, 9.2, 9.3, 10 and all 68 bibliography entries | 4: bibliography read in full | R2 | should_fix | section: §2.3 | interpretive_ambiguity_remains: section "Related Work" |
| S21 | State the timer resolution and label the ~3 µs figure as hot-loop dispatch cost; either profile per operator and report one cold call at app cadence, or drop the INT8-versus-float latency comparison from the contributions | SC-34 | minor | table: Table 19 (tab:devicebench), Appendix C.6 | 4: recomputed from released JSON; test sources read | R3 | should_fix | sentence: Appendix C.6, §7.9 | claim_scope_unsupported: table "Table 19" |
| S22 | State that the §9.1 remedy targets absolute inputs; note that the deployed encoding still clips 2.8% of RT-CGM windows; discuss a float input interface or a two-input graph, marked untested | SC-35 | minor | text: §9.1 "fixing its range from the sensor's reporting limits" | 4: quantization parameters read from the bundled artifact | R3 (DA m6) | should_fix | sentence: §9.1 | claim_scope_unsupported: claim "§9.1 fixed-range remedy" |
| S23 | Reword the §8.2 and §A.1 parity statements to say what actually ran; either run the instrumented suite on an arm64 device for the current pair or scope "three clients agree" (§9.2) to the historical artifacts | SC-36 | minor | text: §8.2 "after golden parity passed in the Android unit tests on the Java virtual" | 4: test sources, CI workflow, README and manifest inspected | R3 | should_fix | sentence: §8.2, §A.1, §9.2 | claim_scope_unsupported: claim "Android golden parity" |
| S24 | Add boundary-stress and saturating golden vectors, report the minimum distance to a rounding boundary, and state the tie rule | SC-37 | minor | text: §C.7 "selected from each float-predicted class and including values above the" | 4: goldens re-encoded and run through the committed artifacts | R3 | should_fix | other (test_assets): golden vectors, §C.7 | method_reproducibility_unresolved: section "§C.7" |
| S25 | Name the intended INT8 deployment target, state that float is the sensible phone default, and position the coverage lesson for full-integer pipelines | SC-38 | minor | text: §5.3 "every complete window locally with the float artifact by default or INT8" | 4: app defaults confirmed in both clients | R3 | should_fix | section: §5.3 or §9.1 (one paragraph) | interpretive_ambiguity_remains: section "§5.3" |
| S26 | Replace or qualify the noise reference for round five's +0.20 (a TCN float seed spread on Weinstock, not a paired INT8 RT-CGM difference), together with S3 | SC-40 | minor | text: Introduction "a fifth of round three's 1.1-point float seed spread on Weinstock held-out" | 4: quantities compared directly | DA | should_fix | sentence: every recurrence (EIC W5 lists six) | claim_scope_unsupported: claim "+0.20 against seed spread" |

### Editorial minor issues (channel `editorial`, no severity)

| Transport ref | Item | Source | Obligation class |
|---|---|---|---|
| E1 | Table 22: write "edges adapted from" for the time-in-range bands | R2 Minor Issues 1 | consider |
| E2 | Table 22 and §9.3: do not average a falling_fast recall computed from a single window into the below-70 band; report band recall only over classes with adequate support | R2 Minor Issues 2 | consider |
| E3 | Report class prevalence per cohort beside precision (PEDAP and RT-CGM prevalences are absent) | R2 Minor Issues 3 | consider |
| E4 | §2.3: tighten "recalls of 0.85-0.98" against Kistkins et al.'s Table 2 (0.78 and 0.79 cases) | R2 Minor Issues 4 | consider |
| E5 | §3: say that the five classes match some device arrow taxonomies more closely than others | R2 Minor Issues 5 | consider |
| E6 | §5.1 and §7.9: give the INT8 output quantization parameters | R3 Minor Issues 1 | consider |
| E7 | §5.3: one sentence on the `libLiteRt.so` `pickFirsts` collision between LiteRT and LiteRT-LM | R3 Minor Issues 2 | consider |
| E8 | §C.6: state the 41.67 ns timer tick beside "within 0.05 µs" (overlaps S21) | R3 Minor Issues 3 | consider |

### Source-Traceability Checklist

> This list follows source order and does not suggest a work order. Author triage is collected later as a separate, explicit sidecar.

- [ ] R1 — obligation `must_fix`: reframe RQ1 as a pitfall-and-audit contribution with one broadening analysis
- [ ] R2 — obligation `must_fix`: move RQ2 out or compress it; describe the bundling as a two-artifact choice
- [ ] R3 — obligation `must_fix`: restate the RQ2 lesson as single-case endpoint disagreement under a declared preference
- [ ] R4 — obligation `must_fix`: add an AI-use statement and correct the "external review" attribution
- [ ] R5 — obligation `must_fix`: state the conv1d lowering cause and scope the INT8 claims; call the artifact hybrid
- [ ] R6 — obligation `must_fix`: per-seed and seed-averaged gate reporting for rounds two and three
- [ ] R7 — obligation `must_fix`: RT-CGM per-class and macro-F1 comparison; state the reallocation in the summaries
- [ ] R8 — obligation `must_fix`: label the override rule post hoc and justify or restate it
- [ ] R9 — obligation `must_fix`: QWK and precision beside confirmatory reads; state the cost ratio; report the offset scan
- [ ] R10 — obligation `must_fix`: float and INT8 profiles for the bundled pair; correct §9.3 and qualify §9.2
- [ ] R11 — obligation `must_fix`: carry the RT-CGM label-validity doubt into every round-five summary
- [ ] S1 — obligation `should_fix`: label the RQ1 headline's descriptive, previously observed status; "untestable as posed"
- [ ] S2 — obligation `should_fix`: restructure the abstract
- [ ] S3 — obligation `should_fix`: shorten the main text and state each caveat once
- [ ] S4 — obligation `consider`: champion-lineage table if RQ2 stays
- [ ] S5 — obligation `should_fix`: cite the precedent on compression's class-level damage
- [ ] S6 — obligation `should_fix`: position against holdout reuse
- [ ] S7 — obligation `should_fix`: reporting-guideline statement
- [ ] S8 — obligation `should_fix`: remove or audit the Explain caption
- [ ] S9 — obligation `should_fix`: show variability in Figure 2
- [ ] S10 — obligation `should_fix`: distribution of the contiguous-block maximum
- [ ] S11 — obligation `should_fix`: threshold rationale and accurate gate-1 wording
- [ ] S12 — obligation `should_fix`: small-cluster-robust companion
- [ ] S13 — obligation `should_fix`: verifiable round-three freeze
- [ ] S14 — obligation `should_fix`: disclose champion selection on held-out data
- [ ] S15 — obligation `should_fix`: interval-based no-effect bounds and an observer interval
- [ ] S16 — obligation `should_fix`: fidelity framing of RQ1's degradation
- [ ] S17 — obligation `should_fix`: baselines matched on operating point, plus a forecaster baseline
- [ ] S18 — obligation `should_fix`: correct the RT-CGM description; main-cohort check
- [ ] S19 — obligation `should_fix`: label-noise ceiling simulation
- [ ] S20 — obligation `should_fix`: imbalance and ordinal-evaluation literature
- [ ] S21 — obligation `should_fix`: rescope the microbenchmark
- [ ] S22 — obligation `should_fix`: scope the fixed-range remedy to absolute inputs
- [ ] S23 — obligation `should_fix`: accurate Android parity statements
- [ ] S24 — obligation `should_fix`: boundary-stress golden vectors and tie rule
- [ ] S25 — obligation `should_fix`: name the INT8 deployment target
- [ ] S26 — obligation `should_fix`: comparable noise reference for +0.20
- [ ] E1-E8 — obligation `consider`: editorial minor issues

### Response Letter Template

Please respond to every item (R1-R11, S1-S26, E1-E8) in the format of `templates/revision_response_template.md`. Answer each seat's Questions for Authors (EIC 5, R1 6, R2 5, R3 5), and mark all changes in the revised manuscript. A declined item stays visible, with its reason.

---

## Part 3: Reviewer Report Summary (Appendix)

### Journal-Fit Review Report Summary
- Recommendation: Major Revision | Confidence: 4 (scope: contribution, positioning and disclosure; statistics and toolchain not audited)
- Key Point: the problem is contribution, not correctness. RQ1's rule is conceded as established, RQ2 is a case study whose general lesson rests on a post hoc decision, and AI involvement, including an AI-generated review credited as external, is undisclosed. A focused, reframed RQ1 paper could clear the bar.

### Reviewer 1 (Methodology) Summary
- Recommendation: Major Revision (stated as advisory) | Confidence: 4 (scope: recomputed from versioned records; no retraining; subset analyses not checked)
- Key Point: the RQ1 effect holds for the recipe as well as the fitted artifacts, and R1 reproduced it exactly. RQ2's gate outcomes are seed-determined, and round five's gain repeats the operating-point pattern behind round one's override, under a rule fitted after the fact. Four arithmetic receipts are attached: three consistent, one not computable.

### Reviewer 2 (Domain) Summary
- Recommendation: Major Revision | Confidence: 4 (scope: evaluation metrics and external validation; not a clinician)
- Key Point: chance-corrected ordinal agreement and per-class precision, recomputed from the published counts, reverse the round-one rejection and the round-five adoption. They also show that INT8 conversion shifts the bundled model's falling_fast operating point in a way the recall-based gate cannot see. RQ1's direction survives.

### Reviewer 3 (Perspective) Summary
- Recommendation: Minor Revision (perspective seat only) | Confidence: 4 (scope: toolchain probe on a random-weight model; no device timing)
- Key Point: the hybrid float/int8 graph comes from conv1d lowering in LiteRT Torch 0.9.1, so the INT8 deployment conclusions describe that lowering path. The RQ1 accuracy findings very likely transfer. The benchmark, the remedy, Android parity and golden-vector coverage claims need rescoping.

### Devil's Advocate Summary
- Recommendation: N/A — findings only
- Key Challenge: no CRITICAL challenge was raised. The strongest MAJOR challenge (M1-M3) is that RQ2's lesson and ship logic depend on an unargued endpoint premise and a post hoc, asymmetric override, and that the shipped artifact rests on a cohort whose label validity the paper itself doubts. M4 concerns the undisclosed AI-generated review. The DA also records two non-defect observations the author may use: the delta experiment partly separates the two accounts of class asymmetry, and every flatten-head seed beats every pooled-head seed on validation.

## Attachment: Acronym Check (advisory, #849)

_Run on a plain-text copy of paper/main.tex (the script accepts only .md/.txt); line numbers refer to main.tex. Script output, not a reviewer finding; no reply needed._

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.
Not checked:
- Body, line 53: INT8 (the initials before its parentheses do not spell it)
- Body, line 106: PTQ (the initials before its parentheses do not spell it)
- Body, line 533: ReLU (the initials before its parentheses do not spell it)
- Body, line 733: QDQ (the initials before its parentheses do not spell it)
- Body, line 831: CE (the initials before its parentheses do not spell it)
- Body, line 1473: CORN (the initials before its parentheses do not spell it)
- Body, line 1950: CRediT (the initials before its parentheses do not spell it)
- Body, line 2071: KiB (the initials before its parentheses do not spell it)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 48 | Not defined | iOS | 21 |
| Body | 74 | Defined again | CGM | 95 |
| Body | 151 | Not defined | RT | 57 |
| Body | 203 | Not defined | PEDAP | 28 |
| Body | 297 | Not defined | CG | 2 |
| Body | 297 | Not defined | EGA | 2 |
| Body | 350 | Not defined | LiteRT | 18 |
| Body | 426 | Not defined | HIGH | 4 |
| Body | 426 | Not defined | LOW | 4 |
| Body | 649 | Not defined | SHA | 13 |
| Body | 668 | Not defined | API | 3 |
| Body | 670 | Not defined | CPU | 4 |
| Body | 677 | Not defined | LM | 1 |
| Body | 818 | Not defined | BG | 3 |
| Body | 818 | Not defined | CITY | 3 |
| Body | 818 | Not defined | DCLP3 | 4 |
| Body | 819 | Not defined | SENCE | 3 |
| Body | 819 | Not defined | WISDM | 3 |
| Body | 1252 | Not defined | MODELS | 3 |
| Body | 1857 | Not defined | BY | 3 |
| Body | 1857 | Not defined | CC | 3 |
| Body | 1857 | Not defined | NC | 3 |
| Body | 1859 | Not defined | DLCP3 | 2 |
| Body | 1863 | Not defined | SEVEN | 3 |
| Body | 1880 | Not defined | CARD | 1 |
| Body | 1880 | Not defined | MODEL | 1 |
| Body | 1952 | Not defined | MIT | 1 |
| Body | 2024 | Not defined | S908E | 1 |
| Body | 2024 | Not defined | SM | 1 |
| Body | 2116 | Not defined | RBG | 1 |
| Body | 2131 | Not defined | GAP | 4 |
