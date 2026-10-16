contract_role: methodology
## Dimension Scores

### D1: methodology_rigor
score: block
trigger: "headline effects resting on a single training seed or a single resampling draw with no seed or draw variability shown"
block_class: repairable

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: block
trigger: "non-significance or overlapping intervals asserted as equivalence"
block_class: repairable

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

The core calibration-coverage study is methodologically stronger than most applied on-device ML work. Its evaluation is participant-disjoint and provenance-masked. It uses a paired participant-cluster bootstrap that never resamples overlapping windows, and its protocols were committed before the reads; I checked every protocol/result pair in git. Its mechanism controls separate clipping from grid resolution. I re-derived its headline contrasts from the released JSON files and they match the manuscript. The core result is also robust where the paper does not show it. From `aggregate_r1.json`, the per-seed uniform-minus-sequential effects are 13.07, 13.56, 13.05, 13.08 and 14.14 points (SD 0.47), so nothing below threatens the 13.4-point finding.

The problems are in the claims built around that core. (1) The second-cohort "confirmation" is described as a replication, but neither primary interval excludes zero, both primaries rest on six clusters, and the hall prediction is declared fulfilled from a non-significant interval with no margin (W1, W2). (2) The accuracy sweeps certify single-seed artifacts, with intervals conditional on both fitted models, while the abstract and conclusion draw recipe-level conclusions. The paper's own round-three record shows a chosen-seed effect of +0.77 points shrinking to +0.05 at the five-seed mean (W3, W4). (3) Several frozen-plan reporting rules were not followed, and several text and number statements contradict each other (W5 to W9).

D1 is scored block (repairable) on W3, with W2 in support. D3 is scored block (repairable) on W1, with W3 and W4 in support. Neither dimension meets my fatal trigger. No test partition contaminates the primary claim: DiaData's Weinstock-overlapping SHD cohort and the D1NAMO/dubosson cohort were excluded (`diadata_audit_r1.json`), and the calibration pool shares no participants with the 20 held-out participants.

Statistical-reporting judgement against the field's reporting criteria (not venue criteria):
- Cluster-level uncertainty for the Weinstock calibration contrasts: MEETS.
- Separate seed and draw variation, which the frozen plan required: DOES_NOT_MEET (derivable from the JSON but not reported).
- Interval validity for the six-cluster cohorts: DOES_NOT_MEET.
- Equivalence-type wording: PARTLY_MEETS. The tight main-study intervals do justify "nothing measurable"; hall's "near zero" has no margin.
- Selective reporting: PARTLY_MEETS (an omitted H2 contrast; full disclosure of deviations in the result files).

### S1: Cluster-aware, provenance-masked evaluation design
**Evidence Anchor**: text: §Experimental Protocol "Overlapping windows are never resampled individually."

The 20 held-out participants are disjoint from the 127-participant calibration pool (`split_audit.json`, `data_availability.json`). Scored keys are asserted disjoint across splits, the gap-fill mask removes interpolated targets, and the held-out and temporal components are never concatenated. The resampling unit is the participant (`experiments/analyze_calibration_coverage.py:59-100`), which is the correct unit for stride-one windows.

### S2: Pre-registration timing is verifiable in version control
**Evidence Anchor**: dataset: git history of paper/results/calibration-coverage/, protocol.json first committed in 803df2f (2026-10-10T00:55) before aggregate_r1.json in 9dd88bb (02:16)

Every frozen file precedes its result file:
- r1: 7d8fe19 before aa6127f.
- r2: 3cdb601 before eff6eac.
- r3: a8383b4 before e446f03 on the PR branch.
- r5: f9010b6 before 3278363.
- dubosson: fadb66e before bdeebf0.
- hall: 220f21b before 88655f0.
- delta: 1ba99ce before 207b7a1.
- Round-four abort rule: c8f3de6 before 0a70282.

On `main`, r3's protocol and results first appear together in squash commit a84b237. The paper should cite the pre-squash hash for r3, as it does f9010b6 for r5.

### S3: Mechanism controls isolate clipping from resolution
**Evidence Anchor**: table: tab:mechanism, Sequential 200 row (Float-clip +13.14, Float-QDQ +13.12, QDQ-INT8 +0.23)

The QDQ, clip-only and min-max observer controls give an interpretable split of the sequential-arm loss into input clipping, input grid and internal quantization. The effect holds seed by seed: float-minus-clip runs 12.65 to 13.87 points across the five seeds (`aggregate_r1.json`, mechanism.held_out.sequential_200_d0).

### S4: Post-hoc reads and decisions against interest are disclosed
**Evidence Anchor**: text: §Accuracy sweep "A pre-registered endpoint can pass every gate and still optimize the wrong thing"

The paper declines to ship a winner that passed every gate, states the delta ship-gate margin (0.004955 against 0.005), publishes deviation lists (`accuracy_sweep_r2.json`, `accuracy_sweep_r3.json`) and demotes reused Weinstock splits in round five. Transparency at this level is uncommon and lets the problems below be audited.

### S5: Paired, hash-pinned artifact comparisons with cross-runtime parity
**Evidence Anchor**: text: §Experimental Protocol "dequantized INT8 outputs are required to match bit-for-bit"

All contrasts are paired on identical windows, and float LiteRT is prediction-identical to PyTorch at every seed. The 110 INT8 artifacts have 110 distinct hashes, one operator inventory and one size (`artifacts_summary.json`). Golden vectors are checked in the Python, Kotlin and Swift runtimes.

### S6: Device benchmark protocol is sound
**Evidence Anchor**: table: tab:devicebench (nine-artifact device benchmark)

The benchmark uses:
- 30 rounds, with artifact order reshuffled each round.
- 100 warm-up and 200 timed calls per artifact per round.
- Thermal state recorded.
- Hash and parity checks before timing.
- A stated p95 estimator (`device_benchmark_ios_r1.json`).

This is adequate for the latency-neutrality claim, which the identical graphs already guarantee.

### S7: Reported headline numbers reproduce from the released aggregates
**Evidence Anchor**: dataset: paper/results/calibration-coverage/aggregate_r1.json contrasts.held_out (uniform200_vs_sequential200 delta 0.13382, ci95 [0.10292, 0.16883]; primary_range200_vs_balanced200 delta 0.00061, ci95 [-0.00016, 0.00132])

The following match the JSON files at the printed precision:
- Tables tab:calibmain, tab:contrasts, tab:sensitivity, tab:delta and tab:sweep to tab:sweepfive.
- The abstract's +0.20 [+0.11, +0.29], which matches `accuracy_sweep_r5.json` (0.00202 [0.00113, 0.00293]).
- The round-five stable and macro-F1 secondaries.

The exceptions are listed in W7 and W8.

### W1: The second-cohort "confirmation" is reported more strongly than its intervals allow
**Severity**: Major
**Evidence Anchor**: text: Abstract "The effect reproduces on two further cohorts"; §Second-cohort confirmation "Both predictions held, and the magnitudes line up with the mechanism"; "near zero as predicted"
**Confidence**: 5, values read directly from confirmation_dubosson.json and confirmation_hall.json

- **Neither primary interval excludes zero.**
  - Dubosson's designated contrast (temporal, six clusters) is 0.019809 with ci95 [-0.0000338, 0.030544], i.e. +1.98 [-0.003, +3.05] points, printed as "[-0.000, +0.031]".
  - Hall's primary (held-out, six clusters) is 0.013068 with ci95 [-0.001437, 0.033031].
- **Dubosson has no decision rule.** Its frozen prediction says only "POSITIVE", without saying whether the point sign or the interval decides. The paper silently uses the point sign.
- **Hall's "near zero" has no margin.** The frozen prediction is "NEAR ZERO" with no margin, yet an interval reaching +3.3 points, a quarter of the Weinstock effect, is declared "near zero as predicted". That reads absence of evidence as equivalence.
- **Hall's frozen premise was falsified.** The rationale ("~0% readings above 240: sequential selection cannot meaningfully under-cover the population range") is contradicted by the measured sequential ceiling of 165.0 mg/dL (mechanism block, cohort max 303.0). The 50-cluster temporal component shows +4.31 [+1.14, +7.34].
- **The two readings contradict each other.** If hall's near-zero prediction held, the effect did not "reproduce" there. If it reproduced, the hall prediction failed.
- **Undisclosed fragility.**
  - 1.2% of dubosson bootstrap draws had an undefined endpoint (undefined_fraction 0.012) and were dropped before the percentile step.
  - Directional support is tiny: 28 falling_fast windows in dubosson temporal, and 41 falling_fast and 50 rising_fast in hall held-out.

**Suggested fix**: State each frozen prediction's outcome separately and label the section exploratory:
- Dubosson: positive point estimate, interval includes zero.
- Hall: primary inconclusive, secondary positive, frozen premise contradicted by a 165 mg/dL ceiling.

Then replace "reproduces" and "Both predictions held" in the abstract, limitations and conclusion, and add per-class support and undefined-draw fractions to tab:confirmation. A confirmation claim needs a pre-registered margin and decision rule, on a cohort with enough participants and fast-class windows.

### W2: Percentile participant bootstrap applied at six clusters
**Severity**: Major
**Evidence Anchor**: text: §Second-cohort confirmation "its frozen protocol assigns the contrast to the temporal component (six clusters)"
**Confidence**: 4. Undercoverage of percentile pairs-cluster bootstraps at about six clusters is well established; I did not re-simulate these cohorts.

The same percentile pairs-cluster bootstrap (resample participants with replacement, take the 2.5/97.5 percentiles, discard undefined draws; `experiments/analyze_calibration_coverage.py:69-100`) is applied unchanged at six clusters (dubosson temporal; hall held-out). Six clusters admit only 462 distinct resamples. The bootstrap variance carries the (G-1)/G shrinkage, and the percentile interval behaves like a normal-quantile interval rather than one built on t with five degrees of freedom. These intervals should therefore be materially too narrow (Cameron, Gelbach and Miller 2008).

The frozen protocol recognised that one cluster is undefined but treated six as adequate. With fast-class support of 28 to 50 windows, probably concentrated in fewer participants still, the effective cluster count for the directional endpoint is lower. W1's claims rest on these intervals. Weinstock's 20 clusters matter much less, given how stable the 13.4-point effect is across seeds.

**Suggested fix**: For cohorts with fewer than about 15 participants, report the per-participant paired differences. Then either use a small-cluster-valid procedure (wild cluster bootstrap-t with Rademacher or Webb weights, or exact sign-flip randomisation over participants) or present those cohorts without intervals. State how undefined draws are handled.

### W3: Sweep confirmatory contrasts are single-seed artifact contrasts, while the abstract and conclusion make recipe claims
**Severity**: Major
**Evidence Anchor**: text: tab:sweepfive caption "candidate = flatten-head CNN, cosine recipe, seed 4, uniform INT8"
**Confidence**: 5, per-seed values from accuracy_sweep_r3.json and accuracy_sweep_r5.json

Each round compares one best-of-five validation seed against one champion artifact, which was itself a best-of-five seed. The bootstrap is conditional on both fitted models. That licenses an artifact-level ship decision. It does not license "a position-preserving flatten head passed all gates" (abstract) or "two changes were promoted" (conclusion).

The paper's own records show the gap:
- **Round three.** The chosen seed 2 beats the champion by +0.77 points held-out. Against the champion's 0.5234, the five per-seed held-out INT8 values are 0.5176, 0.5235, 0.5311, 0.5205 and 0.5268. The five-seed-mean effect is +0.05 points, with only 3 of 5 seeds ahead.
- **Round five.** Only seed 4 was evaluated on RT-CGM; `accuracy_sweep_r5.json` has no per-seed RT-CGM entries.
  - The flatten head's five-seed validation mean, 0.5382, is below the champion seed's validation float, 0.5391 (R_base seed 1). Only seed 4 (0.5409) exceeds it.
  - The +0.20-point effect is about a fifth of the 1.1-point five-seed held-out spread the paper itself reports in round four. That spread comes from a different recipe and endpoint, so the RT-CGM effect has no seed variability shown at all.

Rounds one and two are seed-robust: 5 of 5 seeds beat the champion, with five-seed-mean effects of +5.25 macro-F1 points and +0.55 directional points. So the missing analysis is feasible from cached checkpoints.

**Suggested fix**: Evaluate all five seeds of the candidate and champion recipes on RT-CGM. The confirmatory read is already spent, so treat this as descriptive. Report recipe-level differences with seed-and-participant uncertainty, either a two-stage bootstrap over seeds and participants or a mixed model with seed and participant effects. Otherwise, restate the abstract and conclusion at artifact level ("the seed-4 flatten-head artifact passed the gates").

### W4: Single-factor attributions where several factors co-vary
**Severity**: Minor
**Evidence Anchor**: text: Abstract "a position-preserving flatten head passed all gates zero-shot on a never-read cohort"; Conclusion "Coverage means covering participants; the other knobs were the wrong ones."
**Confidence**: 4, design read from protocol_sweep_r5.json and tab:coverage

**Round five.** The candidate differs from the champion in head, LR schedule, validation-chosen kept epoch and seed, yet the abstract credits the head alone. The conclusion names both head and schedule. The only separate evidence is the staged validation means (recipe +0.27 points, head +0.33 points), which the paper itself calls selection-biased.

**The conclusion's slogan.** "Coverage means covering participants" credits participants, but the paper's own controls point at input range. Sequential-1000 covers only 6 participants yet fully recovers (tab:contrasts, +0.05 [-0.05, +0.15]). Clip-only reproduces the loss.

**Suggested fix**: Attribute round five to the bundled candidate (head plus cosine schedule plus epoch selection) and say the factors are not separately identified at the confirmatory read. Rephrase the slogan as "cover the deployment input range (randomised selection does this)".

### W5: Departures from the frozen plan's reporting rules are not disclosed
**Severity**: Minor
**Evidence Anchor**: text: §Experimental Protocol "Secondary contrasts are range coverage versus uniform random, 1,000 versus 200 within each strategy, and uniform random versus sequential at each budget."
**Confidence**: 5, compared protocol.json and plan sections 1 and 4 against aggregate_r1.json and the manuscript

- **(i) Seed and draw variation is not reported.** Plan section 4 asks for intervals "plus separate model-seed and calibration-draw variation"; none is reported. From `aggregate_r1.json` I computed:
  - Per-seed uniform-minus-sequential effects: SD 0.47 points.
  - Within-seed draw SD: about 0.1 points.
  - Float directional recall per seed: 0.4717 to 0.5087 (SD 1.34 points).

  These are benign for the conclusions, but the plan required them.
- **(ii) The H2 interaction contrast is missing.** `protocol.json` lists a "budget-by-strategy interaction" (H2). It is not among the 7 contrasts in `aggregate_r1.json`, and the manuscript's list of secondaries drops it without comment. Sequential 1,000 vs 200 also has no interval, although "1,000 versus 200 within each strategy" is claimed. My point values: interaction -0.05 points; sequential +13.35 points.
- **(iii) A secondary contrast leads the abstract.** The plan marks secondary intervals descriptive and forbids promoting one to the primary result. The Amendment-2 uniform-vs-sequential contrast leads the abstract with its interval, unlabelled.
- **(iv) Hypotheses H1 to H4 are never stated.** The paper says they were frozen, but readers cannot see that H1 (range coverage helps) was not supported.

**Suggested fix**: Add a protocol-to-report table listing H1 to H4, the primary and every secondary, with status and location. Report seed and draw variation, give intervals for the H2 interaction and for sequential 1,000 vs 200, and label the headline contrast as a prespecified secondary.

### W6: Reuse of the Weinstock held-out split is under-counted, and "one pre-registered read" is overstated
**Severity**: Minor
**Evidence Anchor**: text: Abstract "Five staged accuracy sweeps followed, each judged by one pre-registered read."; Limitations "Weinstock held-out and temporal labels were read in rounds one to three"
**Confidence**: 4, sequence reconstructed from the result files and git history

Before round one, held-out labels had already driven decisions in three places:
- The main study.
- The sensitivity tiers.
- The delta-input experiment. There the variant was chosen on held-out data ("plain never reached conversion; anchor scored higher, 0.5088, and proceeded", `delta_inputs_r1.json`), and that variant then became the champion for rounds one and two. Its "appears to help" comparison (0.509 vs 0.487) is therefore selection-affected.

After that:
- Round one's no-ship decision came from a post-hoc read, which then set round two's endpoint.
- Round four performed no read, yet the conclusion says "Each round spent one pre-registered confirmatory read".

Each fact is disclosed somewhere, but the abstract and conclusion describe a cleaner discipline than the sequence supports. Rounds two and three are adaptive reuse of a single split.

**Suggested fix**: Add a held-out read ledger (date, analysis, decision taken). In the abstract, say that rounds one to three reused one split and that round one was decided post hoc. Describe the delta variant choice as held-out selection.

### W7: Device-timing and artifact-count statements contradict each other
**Severity**: Minor
**Evidence Anchor**: text: §Experimental Protocol "none of the 110 new artifacts has been device-timed"; Limitations "The remaining 101 matrix, confirmation-cohort, and delta-encoded artifacts were not device-timed"
**Confidence**: 5, counts from device_benchmark_ios_r1.json, artifacts_summary.json, delta_inputs_r1.json and confirmation_*.json

- **The protocol sentence is wrong.** Table tab:devicebench times 8 of the 110 INT8 artifacts.
- **"101" fits no count.** 110 minus 8 leaves 102 untimed study INT8 artifacts. Adding 20 dubosson, 20 hall and 20 delta INT8 artifacts gives 162.
- **The shared-inventory claim is only partly published.** The claim that these share "one operator inventory and byte size" is published only for the 110 study artifacts. The confirmation and delta records contain no size or operator fields.

**Suggested fix**: Correct both sentences, and publish the operator-inventory and size summary for the confirmation and delta artifacts.

### W8: Numeric misstatements in the text
**Severity**: Minor
**Evidence Anchor**: text: §Second-cohort confirmation "Weinstock's ceiling of 239.9 against a 401 maximum cost 13.4 points"; Conclusion "moved it by at most eight hundredths of a point"
**Confidence**: 5, values from tab:coverage, aggregate_r1.json and sensitivity_tiers_r1.json

- **(a) Wrong ceiling.** The 13.4-point contrast used the study's sequential-200 range of 0 to 250.9 (tab:coverage; the abstract says 251). 239.9 is the historical shipped artifact's ceiling.
- **(b) The observer moved recall by more than claimed.** The like-for-like comparison (range coverage 200, draw 101) moves held-out directional recall from 0.48653 to 0.48771. That is +0.12 points, more than "eight hundredths".
- **(c) A spliced interval.** The sensitivity-tier sentence gives temporal intervals "[+6.5, +14.7] at the widest", but no tier has that interval. It joins the gap-filled-endpoint lower bound (6.48, from [6.48, 14.56]) to the gap-filled-inputs upper bound (14.67, from [9.32, 14.67]).

**Suggested fix**: Use 250.9 (maximum 251) in the mechanism argument. Restate the bound as "about a tenth of a point at most". Quote the widest single-tier interval.

### W9: The round-one macro-F1 gain is decomposed using recall
**Severity**: Minor
**Evidence Anchor**: text: Abstract "gained only on the majority class and was rejected"; §Accuracy sweep "decomposes the macro-F1 gain by class"
**Confidence**: 4, from tab:sweeprecall and accuracy_sweep_r1.json post_hoc_descriptive_recall

Table tab:sweeprecall gives per-class recall, which cannot decompose a macro-F1 difference. F1 also moves with precision, and predicting stable more often raises minority-class precision. Even on recall, the winner gains on rising (0.3647 vs 0.3338, +3.1 points), so "only on the majority class" is wrong. The rejection itself stands, because it rests on the directional-recall drop from 0.517 to 0.438.

**Suggested fix**: Report per-class F1 and precision for this comparison, and justify the rejection by the directional endpoint alone.

### W10: The label-noise "ceiling" is not a bound on achievable performance
**Severity**: Minor
**Evidence Anchor**: text: §Accuracy sweep round four "the task is model- and data-limited, not label-noise-limited"
**Confidence**: 3, an inference question; the measurement is defined in exp0_sweep_r4.json

Experiment 0 compares two labels computed from the same future readings: the endpoint difference against a least-squares slope over g[t..t+3], plus noise-perturbed variants. An agreement of 0.954 shows the labels are stable to the choice of estimator and to modest sensor noise. It is not an upper bound on what the 12 past inputs can predict. So the "42 points" are not headroom, and the measurement cannot place the remaining error "in the model and data". Unobserved future drivers such as meals, insulin and activity are irreducible from these inputs.

**Suggested fix**: Call it a label-stability check and drop the ceiling and headroom wording, in round four and in the conclusion.

### W11: Interval recomputation depends on unpublished local artifacts
**Severity**: Minor
**Evidence Anchor**: text: §Reproducibility "so a fresh run may produce different numbers"
**Confidence**: 4, from the published file inventory of paper/results/calibration-coverage/

- **Recomputation needs local-only files.** Checkpoints, per-window predictions, window keys and per-artifact hashes are kept locally, and the environment record is stated not to be a complete lock. A third party cannot recompute any bootstrap interval without retraining.
- **Some sweep settings are missing from the manuscript.**
  - The sweeps' training batch size (frozen at 512 in each sweep's smoke record) appears nowhere; the manuscript states 64.
  - The sweep INT8 calibration pool ("from weinstock train only", `protocol_sweep_r5.json`) is not stated either.

**Suggested fix**:
- Publish per-artifact hashes.
- Where the source datasets' terms allow, also publish de-identified per-participant confusion counts so every interval can be recomputed; otherwise state the limitation explicitly.
- State the sweep batch size and calibration pool.

## Arithmetic Receipts

### AR1
procedure_id: grim
evidence_anchor: table: tab:baselines (held-out baselines), row Constant stable, Accuracy 0.727
reported_inputs: accuracy of the constant-stable classifier 0.727 reported to three decimals; analytic N 46,376 eligible held-out windows (tab:eligible, Held-out test, Eligible)
assumptions: constant-stable accuracy equals the share of stable windows among the 46,376 eligible held-out windows, a mean of binary indicators; three-decimal reporting with no stated tie rule
derivation: 0.727 x 46376 = 33715.352; counts 33715 and 33716 give 33715/46376 = 0.7269924 and 33716/46376 = 0.7270140, both inside the rounding interval, so the reported value is attainable
derived_value_or_range: attainable counts 33693 to 33738
rounding_interval: [0.7265, 0.7275)
nearest_achievable: 33715/46376 = 0.7269924 and 33716/46376 = 0.7270140
comparison_rule: consistent if at least one integer count k gives k/46376 inside the rounding interval; no attainable value lies on an interval endpoint, so the tie rule cannot change the verdict
status: consistent

### AR2
procedure_id: grim
evidence_anchor: figure: fig:range caption, share of training windows containing at least one reading above 239.9 mg/dL, reported as 24.4 percent
reported_inputs: proportion 24.4 percent reported to one decimal; analytic N 424,462 training windows (Dataset and preprocessing section; tab:distribution total)
assumptions: a mean of binary window indicators over the 424,462-window training frame that the same audit describes; one-decimal reporting with no stated tie rule
derivation: 0.244 x 424462 = 103568.728; counts 103568 and 103569 give 0.2439983 and 0.2440006, both inside the rounding interval
derived_value_or_range: attainable counts 103357 to 103780
rounding_interval: [0.2435, 0.2445)
nearest_achievable: 103568/424462 = 0.2439983 and 103569/424462 = 0.2440006
comparison_rule: consistent if at least one integer count k gives k/424462 inside the rounding interval; no attainable value lies on an endpoint
status: consistent

### AR3
procedure_id: grim
evidence_anchor: text: §Split and calibration provenance, value 26.3 percent, "validation windows contain at least one reading above that range"
reported_inputs: proportion 26.3 percent reported to one decimal; analytic N 55,289 validation windows (Dataset and preprocessing section; tab:eligible, Calibration pool, Frame)
assumptions: a mean of binary window indicators over the 55,289-window validation frame; one-decimal reporting with no stated tie rule
derivation: 0.263 x 55289 = 14541.007; counts 14541 and 14542 give 0.2629999 and 0.2630180, both inside the rounding interval
derived_value_or_range: attainable counts 14514 to 14568
rounding_interval: [0.2625, 0.2635)
nearest_achievable: 14541/55289 = 0.2629999 and 14542/55289 = 0.2630180
comparison_rule: consistent if at least one integer count k gives k/55289 inside the rounding interval; no attainable value lies on an endpoint
status: consistent
