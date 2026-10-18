contract_role: methodology

## Dimension Scores

### D1: methodology_rigor
score: warn
trigger: "seed or draw variability is reported for some headline numbers but not all"

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: warn
trigger: "a disclosed post-hoc deviation from the declared plan is not carried through to how firmly the conclusion is stated"

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

### Reviewer Identity
Peer Reviewer 1 (methodology). I am a statistician working on the methodology of machine-learning experiments: crossed seed, data-order and sampling-draw effects, variance decomposition of benchmark results, selective inference after model selection, and holdout reuse. The configuration card is the field analyst's reading. I checked each factual claim in it against the manuscript and the versioned records, and the corrections are listed under Verification performed.

### Overall Recommendation
Advisory to the synthesizer, not a panel decision: Major Revision. The RQ1 result is robust and I verified it independently. Two RQ2 findings need rewriting and additional reporting. Both can be done from the published per-participant counts, with no new training and no new label reads.

### Confidence Score
4. Confidence is an uncertainty and scope disclosure only. I recomputed the main numbers from the versioned records. I did not rerun training or conversion. Subset analyses (sensitivity tiers, the RT-CGM 40-400 check) cannot be recomputed from the published counts, so I did not check them.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
RQ1 is a well-built computational experiment. It has a factorial calibration design, five training seeds, three sampling draws, a paired participant-cluster bootstrap with stated cluster counts, and two input-grid mechanism controls. The headline is that contiguous first-N selection costs 13.38 points [+10.29, +16.88]. I reproduced that interval exactly from the published counts. The effect appears at every seed (13.05 to 14.14, SD 0.47), so it holds for the recipe as well as for the fitted artifacts. Its inferential status is weaker than the word "prespecified" suggests: H1 and H2 were untestable as posed; the headline is a secondary contrast that the study plan marks descriptive; and the July revision had already seen the same loss in the same participants. The magnitude and the clip-only control make the effect credible anyway.

RQ2 is weaker. The program's ship decisions are artifact-level, and I recomputed gate outcomes for every candidate seed. In round two only the chosen seed clears all three gates. At the recipe level round two fails gate 2 by its own threshold, and in round three a non-chosen seed would have passed. Round five's primary RT-CGM gain has the slow-to-fast reallocation pattern that the paper treats as non-improvement elsewhere, and RT-CGM macro-F1 falls by -1.65 [-1.75, -1.57] points. The rule that reconciles round five with round one's override was written after both outcomes were known. Disclosure throughout is exemplary. These findings concern how firmly RQ2's conclusions are stated, not concealment.

### Verification performed
- **Exact reproductions from `participant_counts_*_r1.json.gz`**, using the repository's `paired_cluster_bootstrap_counts`, the frozen seeds, and a scratch script outside the repository:
  - uniform minus sequential at 200, held-out: +13.38 [+10.29, +16.88];
  - round-five RT-CGM: +0.20 [+0.11, +0.29];
  - per-seed effects 13.07, 13.56, 13.05, 13.08 and 14.14 (SD 0.47);
  - float directional-recall seed SD 1.34 points;
  - uniform draw SDs within a seed 0.02 to 0.21;
  - Dubosson temporal participant contributions (+1.62 for one participant, -0.04 to +0.16 for the rest), with exact sign-flip p = 6/64;
  - Hall held-out p = 16/64;
  - round one: macro-F1 gain +4.92 to +5.68 at every seed (mean +5.25), and directional-recall loss -5.67 to -7.91 at every seed (chosen seed -7.91 [-8.87, -6.97]). The basis for round one's override therefore holds at the recipe level.
- **Internal consistency checks.** All of the following agree:
  - The abstract matches Table 6.
  - Table 5 accuracy (0.507/0.570/0.496) and macro recall (0.494/0.412/0.492) recompute from Figure 1's per-class recalls and Table 8's held-out supports.
  - Every interaction row in Table 6 equals the difference of its two budget rows. Uniform minus sequential at 1,000 equals 13.38 + 0.02 - 13.35.
  - The Table 7 components sum to the 13.35-point loss.
  - Table 22's directional means equal Table 9's INT8 entries (0.520 held-out, 0.555 RT-CGM).
  - The 63% movement-follow share, the 1.7-5.2 over-call ratios and the 0.138/0.151 reversal shares recompute from Tables 22-23.
  - The size reductions are 31.0% and 70.3%.
  - The Table 3 and Table 14 counts sum correctly.
  - The tier sizes 8,682 + 2,010 equal 57,068 - 46,376.
  I found no inconsistency.
- **Protocol timing in git.**
  - Calibration study: plan `d63cc8f` (00:49) and protocol `803df2f` (00:55) precede results `9dd88bb` (02:16), all on 2026-10-10.
  - Sweep rounds one, two and five: the protocol commit precedes the result commit by 2 to 16 minutes.
  - Round three: see W7.
  - The evaluation guard `protocol_is_committed` exists in `experiments/accuracy_sweep.py`.
- **Corrections to the configuration card.**
  - Gate 1 is not a pure non-inferiority rule. The frozen protocols require a positive point estimate and an interval lower bound above -0.5 points.
  - The read ledger has 15 rows, and 12 of them touch Weinstock held-out or temporal labels.
  - The Dubosson sign-flip dominance is not driven by window count: the dominant participant has 141 windows, the same as four others.

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D1 split integrity and leakage control | contract D1; stage notes (leakage control) | MEETS | see S4; Table 3 | Row-ownership scoring, disjoint scored keys, gap mask, provenance audit | Held-out reuse is extensive but disclosed (12 of 15 ledger rows) | yes, it bounds RQ2's evidential weight |
| D1 uncertainty for clustered dependent data | contract D1; stage notes | PARTLY_MEETS | see S3, W6 | Cluster bootstrap with stated cluster counts; exact test at 6 clusters; 20-cluster percentile intervals used for gates without a small-cluster check | No gate decision changes in my recomputation | yes (RQ2 wording) |
| D1 effects against seed, draw and cluster variability | contract hint 3 | PARTLY_MEETS | see S1, W1 | RQ1 exceeds the bar; RQ2 gates are artifact-level and seed-determined | Round-five non-chosen seeds were not published | yes |
| D1 confirmatory versus exploratory separation | contract hint 2 | PARTLY_MEETS | see S4, W4 | Labels are present throughout; the RQ1 headline is a descriptive secondary on previously scored participants | none identified | yes (labelling only) |
| D1 multiplicity and selective inference | stage notes | PARTLY_MEETS | see W5, W8 | Multiplicity cannot threaten a 10-point lower bound; RQ2 margins are unjustified and the champion was selected on held-out data | Bias direction known, magnitude not estimated | no for RQ1, yes for RQ2 |
| D1 numeric consistency across surfaces | contract hint 1 | MEETS | see S5 | Every cross-check above agrees | Subset analyses not recomputable from counts | no |
| D1 reproducibility affordances | contract D1 | MEETS | see S5, W7 | Counts reproduce published intervals; checkpoints withheld under data terms, with a stated reason | Round-three freeze not verifiable from history | no |
| D1 statistical reporting adequacy (Step 4a) | statistical_reporting_standards.md | MEETS | see S3 | Effects reported in points with CIs, cluster counts, an undefined-draw rule, and exact p values with attainable minimum | Power analysis not applicable to fixed public cohorts | no |
| D3 decision rules applied as declared | contract D3 | PARTLY_MEETS | see W2 | The override rule is post hoc and fitted to two cases | Interpretive | yes |
| D3 claim strength matches evidence, RQ1 | contract D3 | MEETS | see S2 | Mechanism control and seed robustness support the stated scope | none identified | no |
| D3 claim strength matches evidence, RQ2 | contract D3 | PARTLY_MEETS | see W1, W2 | Round two's "most consistent result" and round five's "passed all gates" omit recipe-level and per-class facts | none identified | yes |

### S1: The RQ1 effect is established for the recipe, not only for one artifact
**Evidence Anchor**: text: §7.3 "The effect appears at every training seed (13.05 to 14.14"
The paper separates seed and draw variation from the conditional interval and reports both. I verified that the per-seed effects have SD 0.47 against an effect of 13.4, with draw SDs of at most 0.21. A seed-inclusive interval would therefore change nothing material. This is the right way to report a conditional bootstrap.

### S2: Mechanism controls give an interventional attribution
**Evidence Anchor**: table: Table 7, sequential-200 row (float-clip +13.14, float-QDQ +13.12, QDQ-INT8 +0.23)
The clip-only and QDQ controls each manipulate one factor on the float model. That separates range truncation from grid resolution and internal quantization without any extra training. The claim "clipping, not resolution" is supported at the right strength.

### S3: Inference respects the dependence structure, including the small-cluster case
**Evidence Anchor**: text: §7.7 "with six clusters cluster-based inference"
Windows are never resampled individually, cluster counts are stated, and the six-cluster components get an exact sign-flip test with its discrete attainable minimum disclosed. I reproduced both p values exactly. The additive-contribution statistic is valid under the null that each participant's contribution is symmetric about zero. The replacement of the frozen percentile intervals is disclosed as such.

### S4: Adaptivity is documented at the level of individual looks
**Evidence Anchor**: table: Table 13 (read ledger)
The read ledger, the protocol-to-report map (Table 16) and the deviation notes make every look at protected labels auditable. That is what allowed me to grade each claim by its actual analysis path, not by its label.

### S5: Published per-participant counts allow independent recomputation
**Evidence Anchor**: dataset: paper/results/calibration-coverage/participant_counts_glucobench_r1.json.gz and participant_counts_diadata_r1.json.gz
The counts reproduce the published full-split intervals. They also allowed me to run analyses the authors did not report (W1, W2, W6) without access to the underlying data. Few applied health-ML papers offer a reproducibility affordance this strong.

### S6: Round five confirms on data disjoint from selection
**Evidence Anchor**: text: §6.2 "Round five moved its primary endpoint to RT-CGM, a cohort no round had"
Selection (Weinstock validation seeds and epochs) and confirmation (RT-CGM) use disjoint data. The +0.20 estimate for the chosen artifact therefore carries no winner's-curse bias, and the paper correctly calls the interval conditional on two artifacts, with no training-seed variation.

### W1: Gate outcomes in rounds two and three are seed-determined, and seed robustness is reported for gate 1 only
**Severity**: Major
**Evidence Anchor**: text: §8.2 "The effect did not hinge on the chosen seed"
**Confidence**: 5 — recomputed from the published counts with the repository's bootstrap function, the frozen per-round seeds, and the gate wording in `protocol_sweep_r2.json`/`_r5.json`

All five candidate seeds of rounds two and three were evaluated on held-out data. The manuscript reports the non-chosen seeds only for the endpoint: round two's five-seed mean is +0.55 with none below the champion, and round three's is +0.05. I applied the three frozen gates to each seed's uniform-200 INT8 artifact on the primary held-out component.
- **Round two.** Only seed 1, the chosen seed, passes all three gates.
  - Seeds 0, 2 and 3 fail gate 2: INT8 stable recall is -4.08, -4.32 and -2.81 points against the 2-point limit.
  - Seed 3 also fails gate 3: its INT8 model sits 0.65 points above its own float.
  - Seed 4 fails gate 1: +0.00 [-0.56, +0.65].
  - The five-seed mean stable change is -2.76 points (seed SD 1.48), so the recipe fails gate 2.
- **Round three.** Only seed 4 passes all three gates: +0.34 [-0.43, +1.14], stable -1.85, float gap 0.29. The chosen seed 2 fails gate 2.

In both rounds one seed in five clears the gates, so the validation argmax over seeds decided which round produced a bundled model. "The effect did not hinge on the chosen seed" is true of the gate-1 point estimate and not true of the ship decision. The Discussion calls round two "the most consistent result of the program" but omits a recipe-level stable-recall cost that the program's own gate 2 rejects.

Why it matters: RQ2's account of what the rounds established rests on the ship and no-ship outcomes. At the recipe level the gates do not distinguish round two from round three. Gates 2 and 3 are point rules whose thresholds (2 and 0.5 points) are on the order of the between-seed SD.

Remedy: none of the following needs new training or new reads.
- Add a per-seed gate table for every round whose seeds were evaluated.
- Report seed-averaged versions of gates 2 and 3 beside the artifact-level ones.
- Reword the round-two subsection, Discussion and Conclusion to describe the promotion as an artifact-level pass that the recipe does not reproduce.
- If future rounds are meant to support design conclusions, freeze their gates at the recipe level, for example on a seed mean with a seed-inclusive interval.

### W2: Round five's primary gain repeats the pattern behind round one's override, and the reconciling rule is fitted after the fact
**Severity**: Major
**Evidence Anchor**: text: §6.2 "The two cases that arose fit one rule"
**Confidence**: 4 — the per-class figures and intervals are recomputed from `participant_counts_diadata_r1.json.gz`; the neutrality judgement about the rule is interpretive

**Facts from the RT-CGM counts** (primary component, candidate minus champion INT8):
- Recall: falling_fast +9.7 points (0.576 to 0.673), falling -8.8 (0.582 to 0.494), rising -2.6, rising_fast +2.5. These net to the +0.20 endpoint gain.
- falling_fast precision falls from 0.278 to 0.214, and its predicted share rises from 5.0% to 7.6%.
- Macro-F1 on the same component changes by -1.65 points [-1.75, -1.57] (frozen seed 20261015). The macro-F1 interval is far from zero and its magnitude is about eight times the endpoint gain.
- The Weinstock held-out component shows the same reallocation (falling_fast +8.0, falling -8.8).

**What the manuscript reports.** For PEDAP it diagnoses exactly this pattern as "an operating-point shift from the slow to the fast classes rather than better discrimination". For round one it gives the mirror-image diagnosis. For RT-CGM it shows only the candidate's per-class profile (Table 22) and gives macro-F1 without an interval.

**Why this is a coherence problem.**
- The sweep program's stated main lesson is that a gated endpoint can pass while the model moves its operating point rather than discriminating better. Round five is a second instance on the primary component, yet the decision went the other way.
- The reconciling rule was stated after both outcomes were known: override only for a directional loss on the primary component, not for losses on non-gating components. It is phrased so that round five's primary-component macro-F1 loss falls outside it, because that loss is neither directional nor on a non-gating component.
- An outcome-neutral alternative, such as "override when a reported class-balanced secondary falls on the primary component with an interval excluding zero", would have overridden both rounds. The rule "never override after a read" would have shipped round one.
- A principled distinction is available. Round five's protocol pre-declared macro-F1 as non-gating, while round one's protocol said nothing about directional recall. The manuscript does not make that argument; it frames the rule around components instead.

Why it matters: the ship decision is a stated outcome of RQ2. The abstract and Conclusion say the artifact "passed all gates" without saying that the primary gain is a slow-to-fast reallocation accompanied by a clear class-balanced loss.

Remedy:
- Report the RT-CGM champion-versus-candidate per-class recall and precision, and the macro-F1 interval.
- State in the abstract and Conclusion that the RT-CGM gain reflects reallocation between slow and fast classes.
- Label the override rule as post hoc. If the pre-declared versus undeclared distinction is the intended principle, justify the rule on that basis.
- Before any future read, freeze an override or harm-bound rule with a margin on a named secondary metric.

### W3: The sequential arm is a single unreplicated selection
**Severity**: Minor
**Evidence Anchor**: text: §5.2 "(participant, segment, time) order, as the project's original conversion"
**Confidence**: 4 — design reading; the remedy is label-free

Sequential selection is deterministic, so the 13.4-point magnitude is the effect of one contiguous block: two participants with a 251 mg/dL maximum. The paper's interval conditions on that block. The Discussion's general reading, that contiguous windows "stop below the population's tail", rests on this one block plus the small cohorts. The headline magnitude is therefore not a property of the practice. A different sort order or start offset would give a different ceiling and a different loss.

Remedy, at no read cost: report the distribution of the maximum input over all contiguous 200-window blocks of the pool (or over random start offsets). This is label-free. Optionally, add a random-offset contiguous arm to estimate the expected cost of the practice.

### W4: The confirmatory content of RQ1 is empty, and the abstract's "prespecified" overstates the headline's status
**Severity**: Minor
**Evidence Anchor**: table: Table 16, rows H1 and H2 ("Untestable as posed")
**Confidence**: 4 — based on the protocol file, the study plan commit, and the manuscript's revision history

The primary contrast failed its manipulation: both arms received the same input range. That failure was checkable before the freeze without labels, because the study records each calibration set's input extremes (Table 4). The positive headline is a secondary contrast added by Amendment 2, which the plan marks descriptive. Appendix A shows that the July revision had already observed the sequential-calibration loss on the same participants. The abstract says "prespecified secondary contrast" but not "previously observed on these participants" or "descriptive per the plan". The effect's credibility does not depend on this, because the magnitude and the clip-only control carry it. The issue is labelling.

Remedy:
- Say both facts in the abstract and Conclusion.
- Note in §6.1 that the coverage audit could have served as a pre-freeze manipulation check.
- Make such a check a redesign trigger in future protocols.

### W5: Decision thresholds and frozen predictions lack an operational basis
**Severity**: Minor
**Evidence Anchor**: text: §6.2 "beats the champion's INT8 model on the endpoint"
**Confidence**: 4 — protocol files and recomputed seed variability

Gate 1 combines a positive point estimate with a non-inferiority bound at -0.5 points, but the text calls it "beats". The -0.5 margin, gate 2's 2 points and gate 3's 0.5 points have no stated derivation. They are smaller than or comparable to the seed variability: float directional-recall seed SD is 1.34, and INT8 stable-recall seed SD in round two is 1.48. Gates 2 and 3 are point rules with no uncertainty. That is how the delta ship criterion can pass at 0.4955 while its interval straddles 0.5. The frozen "near-zero" predictions (Hall, delta encoding) had no margin and so could not fail; the paper acknowledges this.

Remedy:
- State a rationale for each threshold, for example a smallest effect of practical interest or a multiple of the seed SD.
- Either give gates 2 and 3 interval forms or call them heuristics.
- Describe gate 1 as a superiority-in-point, non-inferiority-in-interval rule.

### W6: The 20-cluster percentile bootstrap is not checked against a small-cluster method
**Severity**: Minor
**Evidence Anchor**: text: §8.3 "first fully positive interval on the directional endpoint"
**Confidence**: 4 — leave-one-participant-out jackknife and exact enumeration of 2^20 sign patterns on the published counts

The paper treats six clusters as too few for the bootstrap but uses percentile intervals at 20 clusters for gate decisions. For round three's chosen seed, +0.77 [+0.09, +1.49], the jackknife with t(19) gives [+0.01, +1.54]. The exact sign-flip test on additive participant contributions, the paper's own six-cluster method, gives p = 0.099, and one participant contributes 51% of the effect. Round two's jackknife interval, [-0.18, +1.37], leaves gate 1 unchanged. No decision flips, but "first fully positive interval" is fragile.

Remedy: report a small-cluster-robust companion (a sign-flip test or a wild cluster bootstrap-t) for the 20-cluster contrasts, and qualify the wording.

### W7: The round-three freeze cannot be verified from repository history
**Severity**: Minor
**Evidence Anchor**: dataset: git history of paper/results/calibration-coverage/protocol_sweep_r3.json (only commit a84b237, shared with accuracy_sweep_r3.json)
**Confidence**: 5 — `git log --all --follow` on both files

Round three was squash-merged. The protocol file first appears in the same commit as its results and the paper subsection, and no branch preserves the pre-squash freeze commit. The "frozen" entry in Table 13 rests on the squash message and on a code guard whose earlier pass a squash cannot evidence. Every other freeze also rests on author-controlled commit timestamps.

Remedy:
- Publish the pre-squash commit or a tag, and note the squash in Table 13.
- For future protocols, use an external timestamp before the read, such as an OSF or Zenodo deposit or a pushed, signed tag.

### W8: The champion used as comparator was selected on the same held-out data
**Severity**: Minor
**Evidence Anchor**: text: §7.8 "rule advanced the higher-scoring variant, so the anchor variant was chosen"
**Confidence**: 3 — the direction of bias follows from selection; I did not estimate its size

Rounds one and two compare candidates with the delta pair, which was chosen on held-out float recall (0.5088 against 0.4969). Rounds three to five compare with the round-two winner, which was promoted on the round-two held-out read. Selection inflates the champion's held-out score, which biases held-out gate 1 against candidates in rounds one to three. That direction is conservative for promotions, but it means pass and reject outcomes on the reused split are not clean performance comparisons. The paper notes only the delta variant's selection-affected margin over the absolute model.

Remedy: add one sentence to Limitation 5, or add a non-selected reference comparator on held-out data.

### W9: The no-effect statements rely on point estimates
**Severity**: Minor
**Evidence Anchor**: text: §1 "budget, and the observer choice move recall by at most about a tenth of"
**Confidence**: 4 — Table 6 intervals

"At most about a tenth of a point" describes point estimates. The matching 95% intervals reach -0.16 to +0.18 points (Table 6), and the observer comparison has no interval.

Remedy: state the bound the intervals support (within about +/-0.2 points, conditional on the five seeds and three draws), and give the observer arm an interval.

### Detailed Comments

#### Research Questions and Hypotheses
RQ1 is answerable and was answered, though by a secondary contrast (W4). As executed, RQ2 is a case study of one selection program. The paper says so for round five but not for rounds two and three (W1).

#### Research Design
The factorial crosses five seeds with selections shared across seeds, and the 200-window budget is nested in the 1,000-window one. This is efficient and correctly analysed as paired. The sequential factor level has no replication (W3). The sweep's staged greedy design is sound for engineering, but its gates operate on single artifacts (W1, W5).

#### Sampling Strategy
The held-out component has 20 participants and the second cohorts have six each. The paper handles the six-cluster case well and the 20-cluster case less carefully (W6). RT-CGM's 355 clusters are ample. The validity of its labels is a matter for R2 and R3: the collinear fraction of 0.305 and possible sample-and-hold effects bear on round five's primary endpoint.

#### Data Collection
Provenance masking, row ownership and the gap-fill rule are careful and documented. The coded LOW and HIGH values receive a sensitivity check.

#### Analysis Methods
The paired cluster bootstrap averages over seeds and draws inside each bootstrap draw, and the dropping of undefined draws is prespecified. The sign-flip statistic is valid for its symmetric null. Replacing the frozen percentile intervals with it after the read is a change of procedure, but it is disclosed and does not change the conclusion.

#### Results Presentation
Results are complete, including nulls and failures. The two omissions that matter are per-seed gate outcomes (W1) and RT-CGM per-class changes (W2).

#### Reproducibility
Strong (S5), with the round-three verification gap noted in W7.

#### Methodological Fallacies Detected
- Winner's curse appears in the "first fully positive interval" for a chosen seed (W6), alongside the five-seed mean of +0.05, which the paper itself discloses.
- Forking paths appear in the post-hoc override rule (W2).
- No Simpson-type reversal was found between the components.

### Questions for Authors
1. Do the authors regard round two's promotion as evidence about the DiaData-only recipe, given that only one of five seeds clears the gates and the seed-mean stable change is -2.76 points?
2. Is the override rule meant to rest on the pre-declared non-gating status of macro-F1? If so, would a primary-component macro-F1 loss with an interval excluding zero have triggered an override had macro-F1 not been pre-declared?
3. What is the distribution of the maximum input over all contiguous 200-window blocks in the pool?
4. Were calibration-set input ranges inspected before the freeze?
5. How were the -0.5, 2 and 0.5-point thresholds chosen?
6. Can the pre-squash round-three freeze commit be published?

## Arithmetic Receipts

### AR1
procedure_id: grim
evidence_anchor: table: Table 8, Dubosson temporal row, Inference column p=0.094
reported_inputs: exact two-sided sign-flip p = 0.094 at three decimals; 6 participants; all 2^6 = 64 sign patterns enumerated (Section 7.7)
assumptions: the statistic is an absolute pooled difference, so each pattern and its global negation tie and attainable p values are k/32 for k = 1..32, consistent with the paper's stated minimum attainable p of 0.031; round-to-nearest at the printed precision, as the paper's own 13.38 to 13.4 and 16.88 to 16.9 show
derivation: k = 3 gives 3/32 = 0.09375, which rounds to 0.094; independently, enumerating the 64 patterns over the published per-participant contributions gives exactly 6/64 = 0.09375
derived_value_or_range: 3/32 = 0.09375
comparison_rule: consistent if some attainable k/32 lies in the half-open rounding interval of the reported value
rounding_interval: [0.0935, 0.0945)
nearest_achievable: 3/32 = 0.09375 below and 4/32 = 0.125 above the reported 0.094
status: consistent

### AR2
procedure_id: grim
evidence_anchor: table: Table 8, Hall held-out row, Inference column p=0.25
reported_inputs: exact two-sided sign-flip p = 0.25 at two decimals; 6 participants; 64 sign patterns (Section 7.7)
assumptions: attainable p values are k/32 for k = 1..32 because each pattern ties with its global negation; round-to-nearest at the printed precision, as the paper's own rounding shows
derivation: k = 8 gives 8/32 = 0.25 exactly; independently, enumeration over the published contributions gives 16/64 = 0.25
derived_value_or_range: 8/32 = 0.25
comparison_rule: consistent if some attainable k/32 lies in the half-open rounding interval of the reported value
rounding_interval: [0.245, 0.255)
nearest_achievable: 7/32 = 0.21875 and 9/32 = 0.28125 straddle the reported value, which equals the attainable 8/32
status: consistent

### AR3
procedure_id: grim
evidence_anchor: text: §7.7 "has its sign flipped in all"
reported_inputs: smallest attainable two-sided p = 0.031 at three decimals; 2^6 = 64 sign patterns
assumptions: the observed pattern and its global negation always reach the observed absolute statistic, so the smallest attainable p is 2/64 = 1/32; round-to-nearest at the printed precision
derivation: 1/32 = 0.03125, which rounds to 0.031 at three decimals
derived_value_or_range: 1/32 = 0.03125
comparison_rule: consistent if the minimum attainable proportion lies in the half-open rounding interval of the reported value
rounding_interval: [0.0305, 0.0315)
nearest_achievable: no attainable value lies below 1/32 = 0.03125; the next attainable value is 2/32 = 0.0625
status: consistent

### AR4
procedure_id: p_from_test_statistic
evidence_anchor: text: §7.7 "sign-flip test gives"
reported_inputs: exact two-sided sign-flip p = 0.094 for the Dubosson designated contrast; no t, z, F or chi-square statistic and no degrees of freedom are reported
assumptions: none beyond the paper's statement that the test enumerates all 64 sign patterns
derivation: there is no named parametric statistic and df to invert; the reported p is a permutation proportion, whose reachability is checked in AR1
derived_value_or_range: none, because the procedure is outside its boundary for a permutation p value
comparison_rule: not applied
tail_convention: two-tailed
status: not_computable
not_computable_reason: nonstandard_p_procedure
