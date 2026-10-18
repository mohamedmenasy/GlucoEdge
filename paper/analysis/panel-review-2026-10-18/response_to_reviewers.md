# Response to Reviewers

- **Reviewed manuscript:** "GlucoEdge: Quantization-Calibration Coverage and Staged Model Selection in On-Device Glucose Trend Classification" (`paper/main.tex` at commit `08a6f28`).
- **Revised manuscript:** same title, `paper/main.tex` on branch `docs/paper-panel-r3`. Change markup: `git diff 08a6f28 -- paper/main.tex`.
- **Review:** simulated five-seat panel, 2026-10-18, decision Major Revision (11 required items R1–R11, 26 suggested S1–S26, 8 editorial E1–E8). Decision letter and roadmap: `synthesis.md` in this directory.
- **Conventions.** "Where" uses the revised manuscript's numbering: §5.1 Conversion and quantizer; §6.2 Accuracy sweep protocol (`sec:sweepprotocol`); §7.9 Deployment costs and a fully integer graph (`sec:deploycost`); §8 Results: Staged Accuracy Sweeps (`sec:sweepresults`); §9.1–§9.3 Discussion; §10 Limitations; §11 Reproducibility and Ethics; §12 Conclusion; Declarations; App. B (Table 12, read ledger); App. D Additional Sweep Details (D.1 data and recipes, D.2 selection and protocol details, D.3–D.7 rounds one to five, D.8 per-seed gates, ordinal agreement and precision, D.11 error profile; Tables 19–26). Outcomes: *addressed* (the requested change was made), *partly addressed*, *not addressed in this pass*.

## Part 1. Author decisions

1. **RQ2 compressed, not moved out (R2).** RQ2 stays in this manuscript as a compact evaluation-governance case: about two pages of body text (§6.2, §8, §9.3) against roughly five before. The round narratives, the confirmatory-read table and every new RQ2 table are in Appendix D. The title is unchanged.
2. **Broadening analysis: a fully integer graph (R1, with R5's stronger option).** The five study checkpoints were re-expressed with Conv2d(1×3) kernels and an AvgPool2d(1×12), keeping their weights, and reconverted with the study's 20 calibration selections (`experiments/full_integer.py`, `full_integer_r1.json`). This is a post-hoc run on already-read labels and is listed in the read ledger. The full-integer artifacts were not timed on a device.
3. **R11 by rewording, not retraining.** The five-seed RT-CGM comparison was not run. The label-validity doubt is stated wherever the round-five result is summarized, and the bundling is described as a choice between two artifacts.
4. **New sweep figures come from the published counts only.** R6, R7, R9 and R10 are recomputed by `experiments/panel_r3.py` from `participant_counts_*_r1.json.gz`, with each round's frozen bootstrap seed (`panel_r3_r1.json`). The script reproduces the reviewers' recomputations exactly, for example round two's five-seed stable change of −2.76 points, RT-CGM macro-F1 −1.65 [−1.75, −1.57] and round one's QWK of 0.439 against 0.385.
5. **Scope of this pass.** This pass addresses the 11 required items. Suggested and editorial items are addressed only where the required changes covered them (Part 3).

## Part 2. Required revisions

### R1. Reframe RQ1 as a pitfall-and-audit contribution; add one broadening analysis (EIC W1). *Addressed.*
**Change.** The abstract, Introduction (third paragraph and the second contribution bullet) and §9.1 (second paragraph) now state the contribution as a measured pitfall plus an audit rule, not a new calibration method. The pitfall: contiguous selection of overlapping, time-ordered windows departs from the guidance without looking like it. The audit rule: compare the calibration set's input range with the deployment range, and gate INT8 artifacts on minority-class metrics. The broadening analysis is the fully integer graph (§7.9, new fourth contribution bullet, Limitation 2, Conclusion):
- Every reconverted INT8 artifact is a 6-operator graph whose tensors are all int8 or int32 (8,112 bytes; float 14,396).
- Input quantization parameters equal the hybrid artifacts' in 100 of 100 cases.
- The headline holds: +13.09 [+9.99, +16.61] held-out (hybrid +13.38) and +12.13 [+9.94, +14.15] temporal, at every seed (12.71 to 13.85). The clip-only control still accounts for 13.14 points.

**Scope reported.** The run is post hoc on already-read labels, with no device timing. Quantizing the convolutions raises the interior quantization cost (0.46 points at uniform-200, against 0.17), and agreement with the hybrid artifacts is 88–98%. Two shared-range contrasts now exclude zero, by at most 0.3 points:
- range coverage at 1,000 against 200 windows: −0.19 [−0.30, −0.07] held-out;
- range coverage against uniform random at 200: +0.13 [+0.03, +0.23].

The primary contrast stays near zero and is still untestable as posed, because its arms share one input range. The min–max observer arm was not repeated. The Introduction, §9.1 and the Conclusion now limit "at most about a tenth of a point" to the hybrid graph and give "up to 0.3 points" for stratification and budget in the integer graph.

### R2. Resolve the two-papers structure; describe the round-five bundling as a two-artifact choice (EIC W2; DA M3). *Addressed.*
**Change.**
- §6.2 is reduced to the design, gates and read rule. It is now positioned against the holdout-reuse literature (Dwork et al. 2015; Blum and Hardt 2015, both added to the bibliography; this also covers S6).
- §8 is a two-paragraph summary.
- §9.3 is three paragraphs.
- The data, recipe and conversion details and all five round narratives moved to Appendix D (D.1, D.3–D.7). Table 19 (confirmatory reads) moved there too.
- The abstract gives RQ2 two sentences. The abstract, Introduction (last contribution bullet), §8, §9.3 and Conclusion call the round-five adoption "a choice between two artifacts, not evidence that its design improves the task".

### R3. Restate the RQ2 lesson (EIC W2, R2 W1, DA M1). *Addressed.*
**Change.** §9.3 now opens: "In one case, two endpoints disagreed." It names both endpoints: macro-F1, QWK and directional precision favored round one's candidate, while directional recall favored the champion. It names the preference that settled the case (directional recall) and its cost structure (R9). It no longer says the candidate "improved every class-balanced measure", since five-class macro recall also fell; it lists the measures that improved: macro-F1, QWK and the precision of every directional class. It says the rejection was "decided after the read". It says the sweeps "do not show that pre-registered endpoints in general optimize the wrong quantity". The abstract and Conclusion carry the same statement. The former sentence "a pre-registered endpoint can pass every gate and still optimize the wrong thing" is gone.

### R4. AI-use statement; accurate description of the AI-generated review; explain the panel and re-review ledger entries (EIC W3, DA M4). *Addressed; wording needs the author's confirmation.*
**Change.**
- Declarations has a new AI-use paragraph. It names the Claude models and the Claude Code agent and their use for code, analyses and writing. It says most commits carry an AI co-author trailer, that every review under `paper/analysis/` was produced by AI agents, that none is human peer review, and that AI tools are not authors.
  - *Follow-up (2026-10-18).* The author later removed the whole Declarations section, this paragraph included, so the manuscript has no AI-use statement. App. D.2, Limitation 4, §11.1 and the read ledger still describe the AI-generated review.
- App. D.2 now describes the 2026-10-13 review as "an AI-generated review, not an independent one". A separate session of an AI agent wrote it on 13 October against commit `2415fa7`, before round four's results. It was committed after those results (`3755d55`) and before round five's protocol was frozen. D.2 states what the review read: Weinstock held-out and temporal labels, through round two's and round three's cached predictions and logits. It also states which proposals round five adopted (flatten head, learning-rate schedule, epoch selection) and which it did not take up: a position-aware MLP control, ordinal or auxiliary-regression heads, participant-grouped development folds with an untouched final set, and five-seed results beside the selected artifact. It reports the offset scan's result (R9).
- The read ledger (Table 12) renames that row and labels the two 2026-10-16 rows as answering simulated AI-agent reviews. It adds rows for the published counts and this revision's re-analyses (`d45f3a4`, `efedf2d`) and for the full-integer run (`b24a6fa`).
- Limitation 4 and §11.1 use the same description.
- No passage calls the review "external".

**Open point for the author.** The statement says the author "directed the work and takes responsibility for its content". Confirm that this, and the list of models, match your own account of how the tools were used.

### R5. Name the cause of the hybrid graph and scope every INT8 deployment statement (EIC W7, R3 W1, DA m2). *Addressed.*
**Change.** §7.9 names the cause: LiteRT Torch 0.9.1 lowers each one-dimensional convolution to a float convolution on dequantized int8 weights and activations, plus a separate float bias add, and lowers the adaptive average pool to a float sum and multiply. Only the final fully connected layer of the 23 operators runs in int8. §7.9 then states that the latency results "describe the hybrid lowering, not 8-bit inference of the model", and that an integer-only accelerator could not run the graph. The same scope appears in:
- the abstract ("the converter produces a hybrid float/INT8 graph");
- §5.1;
- the deployment-cost contribution bullet ("for the hybrid graph that LiteRT Torch 0.9.1 produces ... That hybrid INT8 graph was never faster than float");
- §7.9's first sentence ("no deployment cost in this model's hybrid graph, the one timed");
- Limitation 8;
- the Conclusion ("a product of LiteRT Torch 0.9.1's convolution lowering rather than of 8-bit inference");
- the historical device observations in App. A.3.

File-size savings are now attributed to artifacts, not to INT8 as such: the hybrid artifacts are 31% (study model) and 70% (bundled model) smaller than float, and the fully integer study artifacts 44% smaller. The sizes are recorded in `full_integer_r1.json` (`serialized_bytes`).

The stronger option was taken (R1). It found that the pooling also had to be re-expressed: with `mean()` or `AdaptiveAvgPool2d` the graph still contains a float DEQUANTIZE, SUM, MUL, QUANTIZE path, so the 9-operator graph described in the R3 report was not fully integer either. The integer artifacts were not timed, which Limitation 8 says.

### R6. Per-seed gates for rounds two and three; artifact-level versus recipe-level (R1 W1). *Addressed.*
**Change.** New Table 21 gives Δ [CI], stable change, INT8−float and the gates passed, for every seed and the five-seed mean, on both Weinstock components.
- Round two: only the chosen seed passes all three held-out gates. The seed-mean stable change (−2.76 points) fails gate 2, although the seed-mean gain is +0.55 [+0.05, +1.14].
- Round three: the chosen seed fails gate 2, while seed 4 and the five-seed mean pass all three gates.

§8, App. D.4–D.5, §9.3, Limitation 4, the read ledger and the Conclusion now describe round two's pass as an artifact-level result its recipe does not reproduce. Seed 4 is described as failing gate 1's lower bound (Δ +0.00 [−0.56, +0.65]), not as failing to beat the champion. The abstract limits the seed finding to rounds two and three. The removed claims are "the effect did not hinge on the chosen seed" (§8.2) and "the most consistent result of the program" (§9.3).

### R7. RT-CGM per-class comparison and the reallocation (R1 W2, R2 W1, DA M2). *Addressed.*
**Change.** New Table 22 gives RT-CGM prevalence and recall, precision and predicted share by class, candidate against champion. Its caption carries the directional-recall, macro-F1 (−1.65 [−1.75, −1.57]) and QWK (−0.90 [−1.01, −0.79]) intervals. Every passage that says round five "passed all gates" (§8, App. D.7, the read ledger, Conclusion; the abstract and Introduction cite the gain directly) also states the reallocation (`falling_fast` recall 0.576→0.673, `falling` 0.582→0.494, `falling_fast` precision 0.278→0.214) and the macro-F1 change. §8 adds that QWK and `falling_fast` precision are lower on all four components read.

### R8. Label the override rule post hoc (R1 W2, DA M2). *Addressed.*
**Change.** The last paragraph of §6.2 presents the rule as "a preference stated after both outcomes were known: directional recall on the primary component ranks first" and calls it "post hoc and fitted to the two cases". It notes that an outcome-neutral rule, overriding whenever another reported class-balanced metric falls on the primary component with an interval excluding zero, would have overridden both rounds:
- round one's directional recall fell by 7.91 points [−8.87, −6.97];
- round five's RT-CGM macro-F1 fell by 1.65 [−1.75, −1.57].

(Round one's protocol declared no secondary metric, so the rule is stated on reported metrics.) The roadmap's prospective advice is now one sentence: "A future round should freeze such a rule, naming the metric and its harm bound, before its read." 

### R9. Cost structure, QWK and precision beside every confirmatory read, and the offset scan (R2 W1, DA M1, M2). *Addressed.*
**Change.**
- Table 19 gains QWK (candidate/champion) and a paired ΔQWK interval for every confirmatory read.
- New Table 20 gives per-class precision for every read, with prevalence per cohort.
- §9.3 and App. D.1 state the cost structure. Directional recall weights the four movement classes equally and counts `stable` only through the floor. The inverse-frequency weights relative to `stable` are 23.5 for `falling_fast` on Weinstock and 26.3 on the DiaData mixture (full vectors in D.1). Neither distinguishes an adjacent band from a reversed direction. With QWK as the endpoint, round one's candidate would have been adopted and the candidates of rounds two, three and five rejected.
- The offset-scan result is in D.2. Near 44.7% held-out stable recall, the scan favored the round-two artifact (52.63% against 52.41% directional recall). The temporal component showed a trade-off. Offsets were chosen on the labels, so the result is descriptive.

The stronger option (matched operating points by logit adjustment, per-class AUROC/AUPRC) was not run.

### R10. Float and INT8 profiles of the bundled pair; correct §9.3; qualify §9.2 (R2 W2). *Addressed.*
**Change.** Table 25 now shows float/INT8 recall, precision and predicted share per class, plus directional recall, macro-F1, QWK, the adjacent-error share and reversal shares (anchor bands stay INT8-only, as the caption says). The caption gives INT8−float QWK intervals for three components. Table 26 shows the float and INT8 confusion matrices side by side. §9.3 now names conversion as a contributor to over-calling: the `falling_fast` call rate rises from 11.0% to 17.8% held-out (temporal 8.5→14.6%, RT-CGM 5.3→7.6%) and precision falls from 0.193 to 0.136, while directional recall moves by at most 0.5 points. §9.2 states that directional recall did not detect this shift (QWK −2.75 [−4.10, −1.63] held-out), and recommends comparing per-class prediction rates or window-level agreement with the float model.

### R11. RT-CGM label validity in every round-five summary (DA M3). *Addressed (rewording route).*
**Change.** Wherever the round-five gain is cited, the text says RT-CGM's rate labels may be distorted:
- abstract: "on a trial cohort whose rate labels may be distorted";
- Introduction, last bullet: "on a cohort whose rate labels may be distorted (Appendix B)";
- §8: the cadence re-audit could not fail, and collinear triples could come from resampling;
- Limitation 4, §9.3, App. D.7;
- Conclusion: "rests on a cohort whose rate labels may be distorted by resampling".

The bundling is presented as a two-artifact choice not backed by evidence of improvement (R2).

## Part 3. Suggested and editorial items

| Item | Outcome | Note |
|---|---|---|
| S1 | Partly addressed | The abstract and Introduction call the headline a secondary contrast. The abstract, Introduction and Conclusion call the primary contrast "untestable as posed". The §6.1 manipulation-check note was not added. |
| S2 | Addressed | The abstract leads with the pitfall, its size, the accuracy illusion and the mechanism. RQ2 has two sentences. |
| S3 | Partly addressed | RQ2 narratives moved to Appendix D, and the body went from 18 to 16 pages. Caveats were not consolidated in general. |
| S6 | Addressed | §6.2 cites Dwork et al. 2015 and Blum and Hardt 2015. |
| S11 | Partly addressed | §6.2 describes gate 1 as superiority in point and non-inferiority in interval. No threshold rationale was added. |
| S26 | Partly addressed | The +0.20-against-1.1-point comparison is removed from the abstract, Introduction, §8, §9.3, App. D.7 and Conclusion. Table 19's caption still gives the seed spread "for scale". |
| E1 | Addressed | Table 25: "edges adapted from". |
| E3 | Addressed | Prevalence beside precision in Tables 20 and 22. |
| S4, S5, S7–S10, S12–S25, E2, E4–E8 | Not addressed in this pass | Outside the required items. Seats' Questions for Authors are answered here only where they bear on R1–R11. |

An independent check of the revised text against each acceptance criterion found 17 wording defects (over-broad scopes, one wrong counterfactual, and the review's proposals and timing). All were corrected before this letter, and the numbers matched the records cell by cell.

## Part 4. New records and checks

- `experiments/panel_r3.py` → `paper/results/calibration-coverage/panel_r3_r1.json`. It reads only the published counts. Tests in `tests/test_panel_r3.py` check QWK against scikit-learn and the never-predicted-class precision rule.
- `experiments/full_integer.py` → `full_integer_r1.json`. `tests/test_full_integer.py` checks that the Conv2d version gives the same logits as the original. `participant_counts_glucobench_r1.json.gz` now includes the full-integer arms, and `experiments/participant_counts.py` verifies 27 published intervals (25 before).
- Test suite: 254 passed. Paper build: 32 pages (16-page body), 0 overfull boxes, 0 undefined references, ASCII-only source.
