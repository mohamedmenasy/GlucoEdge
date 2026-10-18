## Contract Paraphrase

D1 (methodology_rigor, mandatory, owned by the methodology seat). From an adversarial stance, this dimension asks whether the study design could survive a hostile replication attempt: whether data handling admits leakage between fitting, selection and evaluation material, whether reported uncertainty respects the dependence structure of clustered data rather than treating correlated samples as independent, whether multiplicity is acknowledged when many comparisons are run, and whether the released artifacts would actually let an outsider regenerate the headline numbers. I do not score it, but any weakness I find here that also breaks the argument's chain will be raised as a challenge.

D2 (domain_accuracy, mandatory, owned by the domain seat). This dimension tests whether claims sit inside current domain evidence, whether prior work is portrayed fairly rather than as a straw man, and whether domain terminology and cited results are stated correctly. Adversarially, the attack surface is misrepresented baselines, overstated novelty relative to the literature, and drift in technical terms. I do not score it; I will flag domain errors only where they undermine the central argument.

D3 (argumentative_coherence, mandatory, owned by my seat and shared with methodology). This is my scored dimension. It asks whether the core thesis holds together end to end: whether each headline claim is actually entailed by the evidence presented, whether the framing in summary sections matches what the analyses can support, whether exploratory or post-hoc findings are allowed to carry confirmatory weight, whether numbers that should agree across sections in fact agree, and whether the argument relies on a fallacy (selective reporting, equivocation, circular selection, inference from interval sign alone when effect size is negligible against seed and cluster variability). A thesis that survives only when its strongest counter-argument goes unstated fails here.

D4 (cross_disciplinary_relevance, high, owned by the perspective seat). This dimension asks whether a reader from an adjacent field could follow the definitions and framing, and whether claims that reach across disciplines are backed rather than asserted. Adversarially, the risk is jargon that hides assumptions and implications that overreach into fields the evidence does not touch. I do not score it.

D5 (writing_and_structure, normal, owned by the EIC seat). This dimension covers organisation, clarity, figure and table quality, and venue conventions. The adversarial angle is whether structure buries limitations or presentation makes weak evidence look stronger than it is. I do not score it.

D6 (venue_fit_and_contribution, mandatory, owned by the EIC seat). This dimension asks whether the work fits the configured venue and makes an original, significant contribution for its readers. Adversarially, the question is whether the contribution reduces to something already known or trivially derivable once the strongest prior-work comparison is drawn. I do not score it.

## Scoring Plan

### D3: argumentative_coherence
dimension_id: D3
what_to_look_for: Trace each headline claim in the abstract and conclusion back to a specific analysis, check that confirmatory and exploratory results are labelled and weighted differently, verify numeric agreement across abstract, body, tables, figures and cited result files, and test whether effect sizes remain meaningful against reported seed, draw and cluster-level variability.
what_triggers_block: At least one central claim is not supported by the evidence as presented (for example it rests on a post-hoc or descriptive result framed as confirmatory, or on an interval sign whose effect is negligible relative to reported variability), or a headline number disagrees with its source table or result file, such that the thesis needs substantive rewording or new analysis to stand.
what_triggers_warn: The core thesis holds, but secondary claims overreach their evidence, hedging is inconsistent between summary and body sections, a plausible counter-argument goes unaddressed, or minor numeric or wording inconsistencies weaken but do not break the argument chain.
what_triggers_fatal: The central thesis is internally contradicted by the manuscript's own reported results, or depends on a structural fallacy (such as selecting and evaluating on the same material, or a circular argument), so that no rewording without discarding the core contribution can make it coherent.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
