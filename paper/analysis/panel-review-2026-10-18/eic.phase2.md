contract_role: eic

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: warn
trigger: "excess length relative to contribution, redundancy across sections"

Headline numbers agree across the abstract, Introduction, results tables, Figure 2 and Conclusion; I checked more than twenty of them and found no mismatch. The problems are proportion and redundancy. The main text runs to about 15,000 words with 9 main-text tables and 14 appendix tables. The round-five comparison is restated almost word for word in six places. The abstract gives most of its space to RQ2's round history and leaves out the accuracy illusion. RQ2 can only be followed by moving between Section 6.2, Section 8, Table 12 and Appendix D. None of this misleads a reader, but all of it slows one down (W4, W5, W6, W12).

### D6: venue_fit_and_contribution
score: block
trigger: "not clearly distinct from established practice or rests mainly on post-hoc or descriptive findings presented as confirmatory, so a general ML-methodology readership would not find a practice-changing lesson without substantial reframing or added evidence"
block_class: repairable

RQ1's take-home rule is, in the paper's own words, established guidance (W1). Its confirmatory primary contrast was untestable as posed, and its headline is a secondary contrast that the study plan marks descriptive. RQ2's general lesson rests on a post hoc rejection and is familiar to this readership (W2). The provenance disclosure is also incomplete (W3). The core survives: the RQ1 effect is large, mechanistically attributed and honestly scoped. That is why the block is repairable rather than fatal. No clinical-validity claim is made, so the fatal trigger does not apply.

## Review Body

### Reviewer Identity
Senior handling editor working field-general (no target venue configured). Has long cross-venue experience adjudicating empirical ML methodology papers (reproducibility studies, negative-result and lessons-learned reports, evaluation-protocol papers) and cross-listed handling experience with digital-health ML. The question I ask: does a careful but narrow empirical study give a broad ML-evaluation readership something it would change its practice for? No review target is bound (`criteria_binding_unavailable`), so this card makes no venue-alignment claim.

### Overall Recommendation
Major Revision

### Confidence Score
4. My judgements on contribution, positioning and disclosure are within my competence. I did not audit the statistical machinery or the quantization toolchain, and I take those from the manuscript as stated. Confidence is a scope disclosure only.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D6: contribution distinct from established practice | Contract D6; Phase 1 D6 plan | PARTLY_MEETS | text: Section 2.4 (established guidance sentence); table: Table 16 | The measured magnitude, the mechanism and the accuracy illusion are new for this setting; the rule they support is conceded as established | Judged against a field-general readership; no venue criteria bound | yes, it sets the D6 block |
| D6: one transferable finding stated at evidence strength | Contract D6; Card #1 focus 1 | PARTLY_MEETS | text: Section 9.2; text: Abstract | Claims are carefully calibrated, but the most practice-relevant point (accuracy rises while minority recall collapses) is missing from the abstract and buried in Section 9.2 | none identified | yes |
| D6: coherence of the contribution across RQ1 and RQ2 | Contract D6; Card #1 focus 1 | DOES_NOT_MEET | text: Section 9.3 | The two RQs share a model family and a split but not a claim; RQ2 is self-described as narrower than it looks | An author may defend RQ2 as a governance case study; that would need reframing | yes |
| D6: disclosure and positioning | Contract D6; Card #1 focus 3 | DOES_NOT_MEET | text: Appendix D; absence: Declarations | An AI-agent review that proposed round five's design is labelled external; there is no AI-use statement; two relevant literatures are uncited | The commit history establishes the nature of the review; the manuscript does not | yes |
| D6: no dependence on clinical-validity claims | Contract D6 fatal trigger; contract stage notes | MEETS | text: Section 11.2 | Research-use scope is explicit and consistent; a model card lists out-of-scope uses | none identified | yes, it rules out the fatal class |
| D5: organisation and proportion | Contract D5; Phase 1 D5 plan | PARTLY_MEETS | text: Section 12; table: Table 9 | Complete and correctly cross-referenced, but long and repetitive; RQ2 requires several jumps | none identified | yes, it sets the D5 warn |
| D5: numeric consistency across abstract, body, tables and figures | Contract additional_measurement_hints | MEETS | table: Tables 5, 6, 9, 22, 23; text: Abstract | More than twenty headline and derived quantities checked, including the 63% movement-label share and the 5.2x over-calling, recomputed from Table 23 | Spot check, not exhaustive | no |
| D5: standalone figures and tables | Contract D5 | PARTLY_MEETS | figure: Figure 2 | Tables are well captioned; the headline figure shows means without seed or draw variability | none identified | no |
| D1 to D4 | Contract eligible_roles | NOT_ASSESSED | — | Outside this seat's remit | Owned by other seats | no |

### Summary Assessment
The manuscript reports two studies of one compact CGM trend classifier. RQ1 is a pre-registered factorial study of how post-training-quantization calibration windows are selected. Taking the first N overlapping, time-ordered windows (contiguous selection) capped the INT8 input range at 250.9 mg/dL. That cost 13.4 points of held-out directional recall while raw accuracy rose. A clip-only control places the loss at the input edge. Once the input range was covered, no other calibration choice moved recall by more than about a tenth of a point. RQ2 documents five staged, gated model-selection rounds. They ended with a new bundled artifact whose margin is +0.20 points on one cohort and is negative or tied on the others. Execution and disclosure are well above what applied health-ML submissions typically show, and the reported numbers are internally consistent.

The editorial difficulty is contribution, not correctness. By the paper's own statement, the rule RQ1 supports is established guidance. The confirmatory primary contrast could not test its hypothesis. RQ2 is a case study whose general lesson rests on a post hoc decision and is familiar to an ML-evaluation readership. At about 15,000 main-text words, the transferable finding is buried under round-by-round history. The manuscript also credits an AI-generated review as "external" and carries no AI-use statement. A focused, reframed RQ1 paper, with RQ2 cut or published separately, could clear the bar.

### Strengths

### S1: Evaluation governance that a methodology readership could adopt
The read ledger lists every analysis that touched protected labels, each tied to a commit. The evaluation code refuses to run before the protocol is committed. A candidate whose artifacts are SHA-identical to the champion's spends no read. Together these are a concrete, reusable template for controlling held-out reuse in iterative development, which is rarer than it should be.
- **Evidence Anchor**: table: Table 13 (read ledger, Appendix B)

### S2: A large effect, attributed by mechanism
The 13.4-point contrast appears at every training seed (SD 0.47 points). The clip-only control reproduces 13.14 of the 13.35-point loss, and the effect survives every frozen sensitivity tier. The headline is not a seed or draw artifact.
- **Evidence Anchor**: table: Table 7 (mechanism controls)

### S3: Candid self-scoping of what did not work
The primary contrast is labelled "untestable as posed". Round five is described as a choice between two artifacts, not evidence for a design. Frozen predictions that failed are reported as failed. This is the right register for a lessons-learned paper.
- **Evidence Anchor**: table: Table 16 (protocol-to-report map)

### S4: Recomputable results without data redistribution
The published per-participant confusion counts reproduce the full-split intervals without retraining. Artifact hashes, frozen protocols and an archived release complete the record.
- **Evidence Anchor**: text: Section 11.1 "recompute the 25 full-split intervals"

### S5: A reusable pitfall for GlucoBench users
The finding that context-prefix rows let 13.6% of aggregate test windows score validation-owned targets, and let 35.1% of validation windows duplicate training windows, applies to anyone who builds window classifiers on this benchmark's formatter output. The corrected ownership rule is stated precisely.
- **Evidence Anchor**: text: Section 4.2 "Earlier revisions let every prefix row start a window"

### S6: Consistent non-clinical scoping
The research-use restriction, the displayed in-app notice and the model card's out-of-scope uses are consistent. No result is framed as clinical validity.
- **Evidence Anchor**: text: Section 11.2 "This work makes no diagnosis, monitoring, treatment, or dosing claim."

### Weaknesses

### W1: RQ1's increment over established guidance is thin, and the paper concedes the rule
The paper's own positioning states that representative calibration data is established guidance, and Section 9.1 says the study measures a departure from it. Of the frozen hypotheses, H1 and H2 were untestable as posed and H3 and H4 were partly supported. The positive headline is a secondary contrast, added by amendment, that the study plan marks descriptive. The nulls on stratification, budget and observer are confined to a saturated regime. What is new for a broad readership is therefore narrow: the contiguous-selection trap for overlapping, time-ordered windows; its size; and the clipping mechanism.

Improvement: reframe RQ1 as a pitfall-and-audit paper. Make the contribution the specific trap and an audit rule practitioners can apply: report the calibration set's input range against the deployment range, and gate quantized models on minority-class metrics. Add one piece of evidence that broadens it. Options are a regime where random draws under-cover the range (the small cohorts already show that such regimes exist), a fully integer graph, or the fixed-input-range remedy that Section 9.1 proposes but does not test.
- **Severity**: Major
- **Evidence Anchor**: text: Section 2.4 "That calibration data must represent the deployment inputs is therefore established guidance, not a finding of this paper."
- **Confidence**: 4 (generalist ML-methodology editor; contribution judgement, not a technical audit)

### W2: Two papers in one, and the sweep half adds little transferable knowledge
RQ1 and RQ2 share a model family and a split but not a claim. RQ2 takes about half the main text and most of the abstract. Its stated lesson is that a pre-registered endpoint can pass every gate and still optimize the wrong thing. That lesson rests on round one's post hoc rejection and is familiar to an ML-evaluation readership. The final ship decision is, by the paper's own account, a +0.20-point choice between two artifacts. Four factors are confounded, there is one seed per side, and the candidate fails two gates on both Weinstock components. The Discussion concedes the point.

Improvement: move RQ2 to a separate report or supplement. If it stays, compress it to about one page and reframe it as a case study in evaluation governance (S1, plus the drift of the relative stable-recall floor). Position it against the holdout-reuse literature (W9).
- **Severity**: Major
- **Evidence Anchor**: text: Section 9.3 "What the rounds established is narrower than five rounds of selection suggest."
- **Confidence**: 4 (structural and contribution judgement; I did not audit RQ2's statistics)

### W3: AI involvement is undisclosed, including an AI-generated review credited as "external"
Appendix D, Limitations item 5 and the read ledger credit an "external review" with two things. It ran an offset scan that read Weinstock held-out and temporal labels, and it proposed round five's flatten head, learning-rate schedule and epoch selection, which produced the bundled artifact. The manuscript does not say what this review was. The repository record does. The message of commit `3755d55` says the review was "produced by a separate agent", and the commit carries an AI co-author trailer. The read-ledger rows "This revision" and "Re-review follow-up", and the `panel_revision_r1.json` record, likewise come from simulated review rounds (commit subjects `f746bf4`, `201e69b`). The Declarations contain no AI-use statement, although 208 of the repository's 251 commits carry an AI co-author trailer.

This changes no number. It does change how a reader weighs two things: where round five's design came from, and how much independent scrutiny the work has received. Current editorial policies make disclosure a precondition of acceptance.

Improvement: add an AI-use statement covering code, analysis, writing and the simulated reviews. Replace "external review" with an accurate description, state what labels it read, and say which of its proposals were adopted. Explain or rename the panel and re-review artifacts.
- **Severity**: Major
- **Evidence Anchor**: text: Appendix D "An external review published between rounds four and five"
- **Confidence**: 5 (the commit message and trailers were read directly from the repository history)

### W4: The abstract spends its space on round history and omits the accuracy illusion
The abstract runs to about 220 words and more than a dozen quantities. Most of its second half concerns RQ2: seed identity, head, schedule, gate outcomes and macro-F1 pairs. It never says that raw accuracy rose while directional recall collapsed. That is the paper's most practice-relevant point, the reason any accuracy-gated pipeline would have shipped the broken artifact, and it is stated only in Section 9.2. Improvement: lead with the trap, its size, the accuracy illusion and the mechanism, and give RQ2 at most two sentences.
- **Severity**: Minor
- **Evidence Anchor**: text: Abstract "Round five's seed-4 artifact (flatten head, cosine schedule, validation-chosen epoch), now bundled in the research applications"
- **Confidence**: 4 (editorial reading of the abstract against the body)

### W5: Excess length and repeated caveats
The main text is about 15,000 words over 19 two-column pages, with 23 tables in all. The round-five comparison, with its seed-spread caveat, recurs almost verbatim in the abstract, the Introduction, Section 8.5, Section 9.3, Limitations item 5 and the Conclusion. The sixth contribution bullet in the Introduction is a single sentence of about 140 words. Hedging this dense hides the claims it is meant to protect. Improvement: state each caveat once, in Limitations, and refer back to it. Move the bundling history, historical device observations and per-round narrative to a supplement. Aim for a main text roughly half its current length.
- **Severity**: Minor
- **Evidence Anchor**: text: Section 12 "a fifth of round three's 1.1-point float seed spread on Weinstock held-out (its own RT-CGM seed spread was not measured)"
- **Confidence**: 4 (word and table counts measured on the rendered PDF and source)

### W6: RQ2 can only be followed by moving between four locations
Following any round requires the protocol in Section 6.2 and the outcomes in Section 8. It also requires appendix Table 12 to identify the champion and candidate artifacts, and Appendix D for selection details. The main confirmatory table names its champions by reference to the appendix. Improvement: if RQ2 stays, put a compact variant column or a small champion-lineage table in the main text.
- **Severity**: Minor
- **Evidence Anchor**: table: Table 9 (confirmatory reads), caption identifying champions only by reference to appendix Table 12
- **Confidence**: 4 (navigation check of the rendered PDF)

### W7: "In 8-bit integers" overstates the converted graph
The abstract describes the study model as a network in 8-bit integers. Sections 5.1 and 7.9, however, report that both convolutions, their bias additions and the pooling run in float32 on dequantized tensors, with only the final fully connected layer executing in int8. The deployment-cost findings (no latency benefit, cost-free fix) are therefore properties of this converter configuration. The manuscript does not explain why the convolutions stay in float. The title itself does not say INT8, so the wording problem is confined to the abstract and the contribution bullets. Improvement: describe the artifact as hybrid in the abstract, scope the deployment claims to the toolchain configuration, and state why the convolutions stay in float.
- **Severity**: Minor
- **Evidence Anchor**: text: Abstract "For a 2,909-parameter convolutional network in 8-bit integers (INT8)"
- **Confidence**: 3 (framing judgement; whether the hybrid graph comes from configuration is for the systems seat)

### W8: Prior work on compression's hidden class-level damage is uncited
The accuracy illusion is the paper's most portable lesson. It has a well-known precedent: compression can preserve top-line accuracy while concentrating damage on underrepresented classes and atypical examples (Hooker et al., "What Do Compressed Deep Neural Networks Forget?", arXiv:1911.05248). Without that citation the lesson reads as newer than it is. With it, the paper could present its result as a time-series, input-range instance of a known failure mode. Improvement: cite and position against this line of work.
- **Severity**: Minor
- **Evidence Anchor**: absence: Sections 2 and 9.2 — expected prior work showing that compression harms minority classes or atypical examples while aggregate accuracy is preserved; checked Sections 2.3, 2.4, 9.1, 9.2 and all 68 bibliography entries
- **Confidence**: 4 (bibliography searched; precedent confirmed by a web literature check)

### W9: The read ledger is not positioned against the holdout-reuse literature
The governance machinery (S1) is a manual control for adaptive data analysis. The paper cites leakage and benchmark-variance work but none of the holdout-reuse literature, for example Dwork et al., "The reusable holdout" (Science, 2015), or Blum and Hardt's Ladder mechanism (ICML 2015). That is where a methodology readership would place the contribution. Improvement: a short paragraph in Related Work and in the RQ2 discussion.
- **Severity**: Minor
- **Evidence Anchor**: absence: Sections 2.2 and 9.3 — expected positioning against the adaptive data analysis and holdout-reuse literature; checked Sections 2.2, 6.2, 9.3, 10, Table 13 and all 68 bibliography entries
- **Confidence**: 4 (bibliography and text searched)

### W10: No statement on health prediction-model reporting norms
For the digital-health readership, the manuscript points to a repository model card but gives no reporting-guideline statement. A TRIPOD+AI-style checklist would do, or a brief reason why such guidance does not apply to a non-clinical research classifier. Improvement: add one sentence and, if applicable, a supplementary checklist.
- **Severity**: Minor
- **Evidence Anchor**: absence: Section 11 and Declarations — expected a reporting-guideline statement such as a TRIPOD+AI checklist or a stated reason it does not apply; checked Sections 11.1, 11.2, Declarations and Appendices A to D
- **Confidence**: 3 (cross-listed editorial experience; applicability to a non-clinical prototype is debatable)

### W11: An unevaluated on-device language-model caption is described in a paper that makes no claim about it
Section 5.3 describes an optional "Explain" caption produced by on-device language models in both applications, and states that it was not evaluated. No result depends on it. In a health-adjacent paper, however, an unevaluated generative text feature invites questions the manuscript cannot answer. Improvement: remove it from the paper, or report a minimal content audit.
- **Severity**: Minor
- **Evidence Anchor**: text: Section 5.3 "the captions were not evaluated"
- **Confidence**: 4 (scope judgement)

### W12: The headline figure shows means without variability
Figure 2 shows five-seed mean recalls per class with no seed or draw spread, although Section 7.3 reports an SD of 0.47 points across seeds and the falling_fast class carries most of the effect. Improvement: add seed-level points or error bars.
- **Severity**: Minor
- **Evidence Anchor**: figure: Figure 2 (per-class held-out recall)
- **Confidence**: 4 (figure inspection)

### Detailed Comments

#### Journal Fit
No target venue is bound, so I judge fit against a field-general ML-methodology readership, with digital-health ML as a secondary audience, and make no venue-alignment claim. In its current form the manuscript reads as a comprehensive project record rather than an article. A methodology readership would value the RQ1 pitfall and the governance machinery. A digital-health readership would value the cross-cohort error profile, but the paper scopes that profile explicitly as descriptive.

#### Originality
The originality comes from measurement and mechanism, not from a new principle: the contiguous-selection trap for overlapping windows, its size, the clip-only attribution, and the accuracy illusion in a time-series setting. Two precedents are uncited (W8, W9), and the paper itself concedes the governing rule (W1). Its strongest original asset for this readership, the read-ledger governance, is presented as housekeeping.

#### Significance
The significance of a pitfall paper depends on how common the pitfall is. The only evidence that this mistake is "easy to make" is the project's own pipeline. Showing that public conversion examples or tutorials draw calibration data from an unshuffled stream would raise significance more than any further RQ2 round. Within its scope, the effect is large enough to matter wherever it occurs.

#### Structural Coherence
Title, research questions, results and Conclusion are consistent, and the paper under-promises rather than over-promises. The incoherence lies between the two halves (W2), not within either. The Conclusion calls the sweep lesson "methodological". It is better described as a single-case observation with a post hoc decision at its centre.

#### Title & Abstract
The title accurately names both halves. If RQ2 is cut, the title should drop "Staged Model Selection". The abstract needs restructuring (W4) and more accurate wording about the converted graph (W7).

#### Conclusion
The Conclusion's numbers match the body. It repeats the round-five caveats from earlier sections almost verbatim (W5), and its final paragraph appropriately keeps the second-cohort evidence exploratory.

### Questions for Authors
1. What single finding should a reader outside CGM take away? Would you accept moving RQ2 to a separate report or supplement?
2. Is there evidence that contiguous or unshuffled calibration selection occurs in public PTQ pipelines or tutorials for time-series models?
3. What produced the "external review" credited in Appendix D, what inputs and labels did it see, and which of its proposals were adopted? What role did AI tools play in the code, the analysis, the writing and the earlier review rounds?
4. Why do the convolutions stay in float in the converted graph, and would the RQ1 effect and its nulls hold in a fully integer graph?
5. Can range-aware selection be tested where uniform draws under-cover the range? Without that, H1 remains open rather than null.

### Minor Issues
Every presentation and positioning issue I found is listed above as a Minor entry (W4 to W12). I have nothing further at the level of typos or formatting: cross-references, numbering and citation formatting are uniform.
