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
score: block
trigger: "A headline number in the abstract, conclusions or contribution list contradicts, or cannot be traced to, the table, figure or result file it summarizes beyond rounding at the stated precision"

### D6: venue_fit_and_contribution
score: block
block_class: repairable
trigger: "the closest prior on-device, quantized or health time-series work is missing so novelty cannot be checked"

## Review Body

**The submission.** The manuscript reports two studies built on one open-source Android/iOS CGM trend-classification prototype. The first is the one the title, the stated research question and the Discussion are written around: a controlled post-training-quantization study. On a provenance-masked GlucoBench/Weinstock protocol, 100 INT8 artifacts (four calibration-selection strategies, two budgets, three draws, five seeds) show that taking the first 200 time-ordered validation windows covers two participants, caps the input range near 251 mg/dL, and costs 13.4 points of held-out directional recall while raw accuracy rises. Any randomized selection restores float performance, clip-only controls place the loss at input clipping, and the fix costs nothing on device. The second study is a five-round, pre-registered model-improvement program (Sections 7.9 to 7.13). It ends with a 50,645-parameter flatten-head model shipping on a +0.20-point zero-shot gain. I read the full LaTeX source, all 17 rendered pages and the reference list. I also checked headline numbers against the versioned records in paper/results/calibration-coverage/ (protocol.json, aggregate_r1.json, confirmation_dubosson.json, confirmation_hall.json, delta_inputs_r1.json, accuracy_sweep_r5.json, device_benchmark_ios_r1.json). The source contains no instructions addressed to reviewers.

**First impression and fit.** The topic is squarely within applied health-ML evaluation and deployment work: the deployment pitfalls of quantized health time-series models, judged with class-balanced metrics. Field fit is therefore not the problem. No venue criteria are bound for this run, so I make no claim about any named venue. The problem is what the paper says it contributes. After one pass through the title, abstract, introduction and conclusion, I could not name a single takeaway. The title and research question promise a calibration-coverage and quantization study. The abstract gives its second half to the sweep program and closes on a model-selection result. The contribution list has five bullets of different kinds. The conclusion ends on two separate lessons: do not let window order pick calibration data, and pre-registered gates alone were not enough.

**Originality and significance.** The calibration half is a careful case study that isolates its mechanism and gives health time-series deployments a practical, transferable rule: draw calibration windows across participants at random, and gate quantized models on class-balanced metrics. What is new relative to established guidance is narrower than the framing suggests. Vendor guidance already asks for a representative calibration set, as Section 8.1 itself acknowledges, so "nominal sample count said nothing" restates that guidance's premise. The genuinely new parts are:

- the time-series-specific mechanism (overlapping, time-ordered windows make a nominal 200-window set span hours of one or two people);
- the class-asymmetric damage it causes (fast falls start high and get clipped);
- the accuracy illusion this produces;
- the nulls for stratification, budget and observer in this model class;
- delta encoding as a structural fix.

The manuscript never states its novelty in these terms, and it does not engage the literature a reader would need to check that novelty (W2). That gap is why D6 is scored block on the pre-committed positioning clause. The sweep half adds a useful methodological lesson (a pre-registered macro-F1 gate promoted a model whose gain came almost entirely from the majority class) and a catalogue of negative results. But its headline result is smaller than the paper's own reported seed spread (W3), and its presence dilutes the calibration contribution (W1).

**Structure and readability.** The Results section has 15 subsections in roughly chronological order. The sweep methods sit inside Results rather than in Methods or the Experimental Protocol. The calibration study's deployment-cost results (Section 7.15) are separated from its other results by three and a half pages of sweeps, and the Discussion covers only the calibration study. The rendered PDF runs to 17 two-column pages. It has compressed spacing, 20 tables (several with only two or three rows) and a 7.5 pt bibliography. Inside the body, analysis status is labelled with exemplary care (S1), so a careful reader can always recover the line between confirmatory and exploratory work. It is the abstract and the conclusion that blur that line (W7, W8).

**Dimension rationale.** D5 is block because the conclusion makes two statements that contradict the body or the result files beyond rounding, and the abstract repeats one of them:

- the bound "at most eight hundredths of a point" for stratification, budget and observer, although the observer moved range-coverage recall by 0.12 points in aggregate_r1.json (W9);
- the claim that each round spent one pre-registered confirmatory read, although round four spent none (W8).

Each of these is Minor in decision impact. The block follows from my pre-committed mechanical criterion. The remaining presentation issues (W6, W10 to W16, W18, W19) would each be warn-level on their own.

D6 is block, repairable. A genuine contribution exists, but it cannot be accepted as framed because its novelty is not positioned against the closest prior work (W2). Four other findings point the same way: the bundled sweep program (W1), the framing of the round-five result (W3), the undefined "toolchain-default" premise (W4) and the abstract's replication sentence (W5). Nothing here is fatal. The calibration study, the provenance-masked protocol and the open dual-platform artifacts would survive any reasonable revision.

**Recommendation signal (for the synthesizer, who decides): major revision.** The path is clear and mostly editorial:

- make the calibration-coverage study the single contribution;
- move the sweep program to an appendix or split it into a separate report;
- position the novelty against the calibration and health time-series literature;
- correct the handful of contradictions;
- update the system description to match what ships.

As an arXiv cs.LG preprint the work is already usable by a determined reader. These changes are what it needs to be ready for a peer-reviewed applied health-ML venue.

### W1: Two studies in one manuscript, with no single takeaway
**Severity**: Major
**Evidence Anchor**: text: §1 "With calibration settled, the work then asks which model and data changes improve directional recall"
**Confidence**: 5 — editorial structure judgment from the full source and rendered PDF

The research question in Section 1 is about calibration coverage. The title, Problem Formulation, Methods (Sections 4 to 6), Discussion and most of the Limitations are written for that study. The anchored sentence then opens a second study: five staged accuracy sweeps with their own endpoints, gates, external data (DiaData), architectures and confirmatory reads. Its methods appear only inside Results (Sections 7.9 to 7.13). Its outcome closes the abstract. It is absent from the title and the Discussion. The result is a 17-page, roughly 11,400-word paper in which the calibration study's own deployment-cost evidence (Section 7.15) arrives only after the sweeps.

Suggested fix: make the calibration-coverage study the paper; the current title already fits it. Keep the round-one lesson (a pre-registered macro-F1 gate promoted a majority-class gain) as one Discussion paragraph. Move the sweep program to an appendix or a separate report with its own methods section. Move client-engineering detail (the Section 5.3 emulator workaround and the rounding history) to an appendix too. If the author prefers to keep both studies, state two research questions up front, give the sweeps their own Methods and Discussion subsections, and retitle accordingly.

### W2: Novelty is not positioned against the closest prior work
**Severity**: Major
**Evidence Anchor**: absence: §2 Related Work and §8.1 — expected engagement with studies of PTQ calibration-data selection and with quantized on-device health or wearable time-series classifiers; checked §2.1 to §2.4, §8.1, and all 36 entries of the rendered reference list
**Confidence**: 4 — familiar with the PTQ calibration literature; verified the full reference list

Section 8.1 positions the paper's sharpest observation against vendor guidance and three PTQ method papers (clipping, adaptive rounding, small calibration sets). Section 2.4 cites generic mobile-inference benchmarks. Neither engages work on how calibration data are chosen or how much their composition matters, for example:

- Williams and Aletras (ACL 2024), on calibration-data effects in post-training quantization and pruning;
- the range-setting and calibration comparisons in Krishnamoorthi's 2018 and Nagel et al.'s 2021 quantization white papers;
- Wu et al.'s 2020 empirical evaluation of integer quantization;
- data-free calibration such as ZeroQ.

Nor does either section cite any quantized on-device health or wearable time-series classifier. Without that context a reader cannot separate what is new (the sliding-window concentration mechanism, class-asymmetric clipping, the accuracy illusion, delta-encoding robustness) from what that literature may already lead one to expect (the null effects of budget and observer at 8 bits for a very small network).

Suggested fix: add a related-work paragraph on calibration-data selection and one on quantized health time-series models. Then rewrite the novelty statement in Sections 1 and 8.1 around the time-series-specific mechanism and its class-level consequence, instead of "nominal sample count said nothing".

### W3: The sweep program's headline result is framed as a confirmed design improvement although it is smaller than the paper's own seed spread
**Severity**: Major
**Evidence Anchor**: text: §1 contribution list "a position-preserving flatten head confirmed zero-shot on a never-read cohort" versus §7.12 "the five-seed held-out spread (1.1 points) rivals every effect size the sweeps have chased"
**Confidence**: 4 — numbers taken from §7.12, §7.13, Table 18 and accuracy_sweep_r5.json

The round-five gain is +0.20 points (CI +0.11 to +0.29; delta 0.00202 in accuracy_sweep_r5.json). It compares one seed-4 candidate artifact with one champion artifact, and its interval is conditional on those two fitted models. Section 7.12 reports a five-seed held-out spread of 1.1 points, more than five times the certified gain. The body is candid that the gain is small. Even so, the contribution list says the flatten head was "confirmed", the conclusion says "a position-preserving head with a cosine schedule" was "promoted", and the abstract ends on it. What the read certifies is a pre-registered ship decision between two fixed artifacts. It does not show that position-preserving heads or the cosine recipe improve this task.

Suggested fix: describe round five as a ship decision under a pre-registered rule, and state beside the +0.20 figure that it is below the reported seed spread. Drop "confirmed" and "promoted" as design-level claims unless a multi-seed comparison of the two recipes on RT-CGM supports them.

### W4: The "toolchain-default" premise behind the significance claim is undefined and unsupported
**Severity**: Major
**Evidence Anchor**: text: §5.2 "the toolchain-default pattern that produced the shipped artifact"
**Confidence**: 3 — the manuscript cites no source for the default; toolchain documentation was outside this review's read scope

The abstract, contribution list, Sections 5.2, 7.8 and 8.1 and the conclusion all call time-ordered take-first-N selection the toolchain default. The practical significance of the main result rests on that claim: if it is the default, many deployments will hit this failure. Yet the manuscript never says which toolchain's default this is (the quantizer's, or the order of the GlucoBench data pipeline) and gives no source. Section 5.1 describes it as this pipeline's own choice ("The first 200 validation windows serve as calibration inputs"). Section 8.1 reports that official guidance asks for a representative calibration set, which sits uneasily with calling the non-representative pattern a default.

Suggested fix: either cite the official example code or documentation that feeds calibration data in stored order without shuffling, naming the toolchain and version, or rename the arm "time-ordered take-first-N selection" throughout and argue its prevalence separately.

### W5: The abstract overstates the two-cohort confirmation
**Severity**: Minor
**Evidence Anchor**: text: Abstract "The effect reproduces on two further cohorts"
**Confidence**: 4 — checked against Table 11 and the frozen predictions in confirmation_dubosson.json and confirmation_hall.json

The frozen prediction for hall was that the uniform-minus-sequential difference would be near zero. The prediction for dubosson was only that it would be positive. The designated components gave +2.0 points (CI -0.0 to +3.1) and +1.3 points (CI -0.1 to +3.3), against 13.4 on Weinstock. What held is a pattern scaled by the mechanism, not a reproduction of the effect. The same wording recurs in Limitations item 2 and in the conclusion ("confirmed on two further cohorts"). The hall prediction also assumed a ceiling near 240 mg/dL, but the realized sequential ceiling was 165 mg/dL (confirmation_hall.json).

Suggested fix: say that the pre-registered predictions held on their designated components, with effects of about 2 and 1 points. Add that hall's better-powered temporal component showed +4.3 points even though the hall prediction had been near zero.

### W6: Methods and Figure 1 describe a model that no longer ships
**Severity**: Minor
**Evidence Anchor**: figure: Figure 1, end-to-end workflow (rendered page 5)
**Confidence**: 4 — cross-read §4.3, Figure 1, Table 1, §7.9 to §7.13 and §10.1

Figure 1 and Section 4.3 present a 2,909-parameter TrendCNN trained on Weinstock as the system. Table 1 gives the evaluation scope as Weinstock held-out and temporal only. The model that the abstract says now ships in both applications is the 50,645-parameter flatten-head CNN from round five, with a cosine schedule and seed 4. As far as the sweep sections indicate, it was trained on the round-two DiaData-only mixture, which contains no Weinstock training windows. Its input encoding has to be inferred from scattered sentences, and the pool its uniform INT8 calibration set was drawn from is not stated.

Suggested fix: add a table of model variants (the calibration-study TrendCNN, the delta-encoded pair, and the round-two and round-five winners) giving architecture, parameter count, input encoding, loss, training data, calibration pool and artifact hash. Update Figure 1 and the Table 1 row to match.

### W7: The abstract is dense and does not mark what was confirmatory
**Severity**: Minor
**Evidence Anchor**: text: Abstract "difference-encoded inputs make quantization nearly calibration-proof"
**Confidence**: 4 — word count of the LaTeX abstract and claim-by-claim check against the body

At about 265 words and more than 20 numeric values, the abstract reads as a results digest of both studies. It has three specific problems:

- It does not say that the 13.4-point headline is a prespecified secondary contrast, added by a pre-label amendment, while the pre-registered primary contrast (range coverage versus balanced random) was null.
- It places the post-hoc delta-encoding result next to pre-registered confirmations without a label. Section 7.8 itself calls that experiment post-hoc on previously touched data.
- It says the round-one winner "gained only on the majority class", but Table 13 shows that rising recall also improved (0.365 versus 0.334).

Suggested fix: cut the abstract to about 200 words on the calibration study: question, design, primary and key secondary results with their status, mechanism, deployment cost and scope. Label any post-hoc result as post-hoc.

### W8: Abstract and conclusion say every sweep round spent one confirmatory read
**Severity**: Minor
**Evidence Anchor**: text: §11 Conclusion "Each round spent one pre-registered confirmatory read"
**Confidence**: 5 — direct contradiction with §7.12 and the Table 17 caption

Section 7.12 states that round four performed no confirmatory read, and the Table 17 caption says no held-out read was spent. The abstract repeats the claim ("each judged by one pre-registered read").

Suggested fix: say that four rounds spent one read each and that round four ended on an identity proof without a read.

### W9: The conclusion's bound on the null knobs is contradicted by the result file
**Severity**: Minor
**Evidence Anchor**: text: §11 Conclusion "stratification, budget, and observer choice moved it by at most eight hundredths of a point"
**Confidence**: 5 — recomputed from the observer-arm cells in aggregate_r1.json

For the randomized range-coverage arm, the min-max observer moved five-seed held-out directional recall from 0.48653 to 0.48771, a change of 0.12 points. That exceeds the stated bound, and Section 7.3's "0.488 vs. 0.487" already hints at it. The substance (the observer does not matter) is unaffected, but a quantitative bound in the conclusion should match the record. The same sentence reads "Any randomized selection of the same 200 windows", but the randomized sets are different windows, so it should say "of 200 windows".

Suggested fix: give the bound as about a tenth of a point, or state the observer comparison separately.

### W10: Section 7.7 pairs the historical ceiling with the corrected effect
**Severity**: Minor
**Evidence Anchor**: text: §7.7 "Weinstock's ceiling of 239.9 against a 401 maximum cost 13.4 points"
**Confidence**: 5 — Table 5 and the sequential_200 coverage record in aggregate_r1.json give 251 and 250.9

The 13.4-point effect belongs to the corrected-protocol sequential-200 set, whose ceiling is 250.9 mg/dL (Table 5). The 239.9 figure is the historical shipped artifact's ceiling (Table 4). The paragraph also argues dose-response in mg/dL gaps, and those gaps do not order the cohorts. Hall's gap (165 against 303) is larger than dubosson's (286 against 400), yet its primary effect is smaller. The share of clipped windows in each cohort would order them, and Section 7.4 already reports that share for Weinstock.

Suggested fix: use 250.9, and report the clipped-window share for each cohort as the explanatory quantity.

### W11: A protocol sentence still says no new artifact was device-timed
**Severity**: Minor
**Evidence Anchor**: text: §6 "none of the 110 new artifacts has been device-timed"
**Confidence**: 5 — contradicted by Table 19 and §7.15

Table 19, Section 7.15 and Limitations item 7 all report a prespecified nine-artifact benchmark of new artifacts on the iPhone. This sentence, and the neighbouring claim that the device observations predate the study, are therefore stale. Limitations item 7's count of the "remaining 101 matrix, confirmation-cohort, and delta-encoded artifacts" also mixes sets: 101 is 110 minus 9, a total that excludes the confirmation and delta artifacts.

Suggested fix: update Section 6 to describe both the historical observations and the new benchmark, and correct the count.

### W12: Revision history is narrated in the main text
**Severity**: Minor
**Evidence Anchor**: text: §10.1 "The September revision reused the July results"
**Confidence**: 4 — tallied revision references across §1, §4.3, §5.1, §5.3, §6, §7.14, §7.15, Table 20 and §10.1

The paper repeatedly tells the reader what an earlier revision did ("An earlier revision of this work could only motivate the question", the July Kotlin rounding note, Section 7.14's historical comparison, and Table 20's "uncorrected protocol era"). The provenance is valuable, but it interrupts the argument and makes the reader track three generations of models and protocols.

Suggested fix: move the version history, Section 7.14 and Table 20 into a short provenance appendix. Keep one sentence in Section 1 noting that this version corrects an earlier evaluation protocol.

### W13: Several tables are typeset below comfortable reading size
**Severity**: Minor
**Evidence Anchor**: table: Table 7, bracketed confidence intervals in tiny type (rendered page 8)
**Confidence**: 4 — inspected rendered pages 8 to 13

Table 7 sets its intervals in the smallest type size. Tables 1, 6, 7, 10, 11, 12, 14 and 19 are scaled down to column width. Tables 15 to 18 use script-size type with tight column spacing. The bibliography is set at 7.5 pt. The preamble comment shows that all of this was done to meet a page budget. Several of the sweep tables have only two or three rows.

Suggested fix: merge the four sweep confirmatory tables (14, 15, 16 and 18) into one, move secondary tables to an appendix, and restore normal type sizes.

### W14: Figure 3's legend is drawn over the bars
**Severity**: Minor
**Evidence Anchor**: figure: Figure 3, legend overlapping the falling and stable bars (rendered page 8)
**Confidence**: 4 — inspected the rendered figure

The legend has no fill and sits inside the plot area. Its "INT8 sequential 200" and "INT8 uniform 200" entries are printed across the bars of the falling and stable groups, including the tallest bar (0.648).

Suggested fix: move the legend above or outside the axes, or raise the axis maximum.

### W15: Wording drifts toward clinical framing
**Severity**: Minor
**Evidence Anchor**: text: §8.2 "a worse clinical-shape classifier with better accuracy"
**Confidence**: 4 — terminology check across §3, §7.12 and §8.2

Section 3 carefully presents the rate bands as conventions borrowed from CGM trend arrows rather than clinical rules. Section 7.12, however, calls them "the unchanged clinical thresholds" and decodes at "the clinical thresholds", and Section 8.2 uses the undefined term "clinical-shape classifier". For a research prototype that disclaims clinical use, these phrases should stay neutral.

Suggested fix: use "trend-arrow rate thresholds" and "directional (minority-class) recall" consistently.

### W16: The sweep sections keep switching reference conventions
**Severity**: Minor
**Evidence Anchor**: table: Table 17 caption, champion-reference convention note
**Confidence**: 4 — traced champion values across Tables 12 to 18 and §7.9 to §7.13

"Champion" refers to the delta-encoded pair in rounds one and two and to the round-two winner after that. It is quoted variously as a five-seed mean (0.509), a bundled single seed (0.517) and validation values (0.519, later 0.533) whose precision is explained only in the Table 17 caption. That caption needs several lines to explain why a model with an identical SHA-256 reads 0.6 points differently. Separately, Tables 6 and 9 label five-class macro recall "Macro", while the sweeps report macro-F1.

Suggested fix: name each reference model once in a table, state the precision and split behind every quoted champion number, and label the column "Macro recall".

### W17: A frozen secondary contrast is not reported
**Severity**: Minor
**Evidence Anchor**: absence: §6 secondary-contrast list and Table 7 — expected the budget-by-strategy interaction contrast that protocol.json lists among the frozen secondary contrasts; checked §6, Table 7, and the contrasts block of aggregate_r1.json
**Confidence**: 4 — compared protocol.json with §6, Table 7 and aggregate_r1.json

protocol.json lists four secondary contrasts, including a budget-by-strategy interaction. Section 6 lists three, and neither the paper nor aggregate_r1.json reports the interaction. Given the within-strategy budget contrasts it is very likely null, but a reader checking the pre-registration should not have to infer that.

Suggested fix: report the interaction, or state in Section 6 that it was dropped and why.

### W18: A cited diagnostic is missing from the versioned record
**Severity**: Minor
**Evidence Anchor**: absence: paper/results/ and paper/results/calibration-coverage/ — expected recency_collision.json, the diagnostic cited in §7.13 as the round-five motivation; checked both directory listings and the §10.1 inventory of versioned records
**Confidence**: 4 — listed both results directories

Section 10.1 says that each sweep round's frozen protocol and full record are versioned in the calibration-coverage directory. The file behind the round-five motivation is not there. Round four's pre-registered gates also have no protocol file; only the noise-ceiling record is published.

Suggested fix: version recency_collision.json and round four's frozen protocol, or give their repository paths.

### W19: Licence and ethics statements are incomplete for the shipped weights
**Severity**: Minor
**Evidence Anchor**: text: §10.2 "We do not claim an institutional ethics determination for this work."
**Confidence**: 3 — the licence of the distributed weights is not stated in the paper; the repository licence was outside my read scope

The bundled models are now trained on DiaData cohorts released under CC BY-NC 4.0, and the weights are distributed in a public repository. The paper does not say under which terms the derived weights are distributed or whether the non-commercial condition carries over. Health-informatics readers expect a short ethics statement in the conventional form: plainly saying that no review was sought and why (public, de-identified data used under the providers' terms).

Suggested fix: state the licence of the distributed weights and how it relates to the source terms, and replace the disclaimer with a one-sentence ethics statement in that form.

### S1: Analysis status is labelled with unusual care in the body
**Evidence Anchor**: text: §7.8 "this is a post-hoc experiment on previously touched held-out data"

Post-hoc, descriptive, frozen-but-run-later and confirmatory results are marked in the text and captions throughout (Sections 4.1, 7.6, 7.8 and 7.9, and Table 13). A careful reader can always recover what was pre-specified.

### S2: Mechanism controls isolate the cause
**Evidence Anchor**: table: Table 8, mechanism controls

The clip-only and quantize-dequantize controls show that input clipping reproduces essentially the whole 13.35-point loss, while the coarser input grid costs nothing measurable. That turns an observation into an explanation.

### S3: The accuracy illusion is a practically important result for this readership
**Evidence Anchor**: table: Table 6, held-out accuracy versus directional recall

Accuracy rises from 0.507 to 0.570 while directional recall falls from 0.487 to 0.354. A deployment pipeline that gates quantized models on accuracy would promote the broken artifact. This is the paper's most transferable lesson.

### S4: Deployment cost is measured with a paired, randomized on-device protocol
**Evidence Anchor**: table: Table 19, nine-artifact device benchmark

The benchmark uses thirty paired rounds with reshuffled artifact order, hash and parity checks before any timing, and thermal-state logging. The reported values match device_benchmark_ios_r1.json.

### S5: Headline numbers trace to versioned result files
**Evidence Anchor**: dataset: paper/results/calibration-coverage/aggregate_r1.json

The following match the versioned JSON records to the stated precision: the primary and secondary contrasts in Table 7, the sequential-200 coverage in Table 5, the round-five gate values, the delta ship-gate margin and the device means.

### S6: Scope is stated honestly
**Evidence Anchor**: text: §9 "statements about this 2,909-parameter CNN with per-tensor affine input quantization, not about PTQ in general"

The Limitations section names the single architecture, quantizer and primary cohort. It also states that the study reanalyses previously touched data and that the sweep validation numbers are selection-biased.

### S7: A rejected pre-registered winner is reported
**Evidence Anchor**: table: Table 13, post-hoc per-class recall of the rejected round-one winner

The author declined to ship a model that passed every gate because its gain came almost entirely from the majority class, then redesigned later gates around that lesson. This is a useful methodological contribution in its own right.

### S8: The provenance audit corrects a benchmark-level evaluation hazard
**Evidence Anchor**: table: Table 4, window provenance audit

Under naive windowing of the GlucoBench frames, 13.6% of aggregate test windows scored validation-owned targets and 35.1% of validation windows duplicated training windows. The paper shows this and provides masks that remove both problems, which anyone working with these formatted cohorts can reuse.

### S9: The research-prototype scope is explicit
**Evidence Anchor**: text: §10.2 "This work makes no diagnosis, monitoring, treatment, or dosing claim."

The abstract, the body and the conclusion consistently disclaim clinical use, and the application displays the restriction in its primary interface.
