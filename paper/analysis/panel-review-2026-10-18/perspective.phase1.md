## Contract Paraphrase

D1 (methodology_rigor). This mandatory dimension asks whether the study design, data handling, statistical reporting and reproducibility support are strong enough for empirical-ML peer review. That covers leakage control, keeping confirmatory and exploratory analyses apart, uncertainty estimates that respect clustered and dependent data, multiplicity handling, and whether released artifacts let someone rerun the work. The methodology reviewer owns and scores it. From the perspective seat, I only care whether the reproducibility support would carry over to a practitioner on another toolchain or device. That observation can inform my own dimension, but I will not score D1.

D2 (domain_accuracy). This mandatory dimension asks whether the claims match current evidence in the field, whether prior work is described fairly, and whether domain terms and reported results are free of factual errors. The domain reviewer owns it. Seen from outside, it is the base that any cross-field reading relies on: if a term or a cited result is wrong, adjacent readers will carry the error with them. I will not score it.

D3 (argumentative_coherence). This mandatory dimension asks whether the central thesis is internally consistent, whether the evidence actually supports what is claimed, and whether any fallacy weakens the main argument. The devil's-advocate seat owns it, and the methodology seat is also eligible. From my angle it matters most where an argument crosses a discipline boundary, for example when a result measured in one setting is presented as a conclusion for another. Stress-testing the thesis is not my job, and I will not score it.

D4 (cross_disciplinary_relevance). This high-priority dimension is the one my role owns. It asks whether the framing, definitions and implications can be understood and used by readers in neighbouring fields, and whether claims that reach across disciplines are backed by evidence and not just asserted. For a systems-oriented health-sensing paper, those neighbouring readers include on-device runtime and compiler engineers, quantization practitioners, wearable and health-informatics researchers, and people who benchmark inference across platforms. The test is whether each of these groups can tell what was measured, under which toolchain and runtime conditions, what transfers to their setting, and what does not.

D5 (writing_and_structure). This normal-priority dimension covers how the manuscript is organised, how clear the exposition is, how good the figures and tables are, and whether it follows venue conventions. The editor-in-chief seat owns it. Clear structure helps adjacent-field readers, but I will not score presentation quality.

D6 (venue_fit_and_contribution). This mandatory dimension asks whether the manuscript suits the target venue and makes an original, significant contribution for that venue's readers. The editor-in-chief seat owns it. Whether the contribution is useful beyond its home field is related to my dimension, but the venue-fit and significance judgement is not mine, and I will not score it.

## Scoring Plan

### D4: cross_disciplinary_relevance
dimension_id: D4
what_to_look_for: Whether runtime, quantization, benchmarking and health-sensing readers can each follow the core definitions and setup. This includes whether the conversion path, runtime and kernel backend, toolchain versions and device or simulator conditions are stated precisely enough to transfer. It also includes whether every implication offered to an adjacent field (deployment readiness, cross-platform numerical agreement, on-device latency or footprint, wearable-data practice) has its own evidence and stated limits, and whether alternative readings of the systems results are acknowledged.
what_triggers_block: An interdisciplinary claim that carries the contribution is presented as established without evidence for it in the target field, such as deployment, cross-runtime parity or practical health-sensing relevance. Or a core term is undefined or misused in a way that would lead adjacent-field readers to misread the main result.
what_triggers_warn: Localized gaps that limit transfer without undermining the central claims. Examples are jargon from one field left unexplained for another, implications for adjacent fields that are stated but thinly supported or missing scope limits, and missing runtime, backend or version detail that a practitioner would need to compare results.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
