# Calibration Coverage and GlucoEdge Paper Enhancement Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` when implementation is requested. Complete the tasks in dependency order and track progress with the checkboxes below.

**Date:** 2026-09-19

**Status:** Research and implementation plan; experiments have not been run.

**Goal:** Strengthen the paper by answering: **How does calibration-data coverage affect minority-class performance and deployment cost in compact glucose trend classifiers?**

**Architecture:** Keep the existing TrendCNN, forecast target, and INT8 quantizer fixed. Build a corrected evaluation around controlled calibration sampling, paired comparisons, and measured Android costs, then rewrite the manuscript from the resulting evidence.

**Tech stack:** Existing Python, NumPy, pandas, PyTorch, scikit-learn, LiteRT Torch/PT2E, Kotlin/LiteRT, pytest, and LaTeX.

**Spec:** The user-approved research question above and the protocol in this document. This is a new experimental phase after the [July literature revision](../../../paper/docs/superpowers/specs/2026-07-20-paper-context-literature-review-design.md); it supersedes that revision's restriction against new experiments for this work only.

## Global constraints

- **Not a medical device** — no proprietary data, schemas, or algorithms belong in this repo, and no code, comment, or output string should imply clinical validity or treatment guidance.
- Read GlucoBench as an external dependency; never modify or vendor it.
- Preserve grouping by `(id, id_segment)` and the five existing label thresholds.
- Keep 12 five-minute input readings and a three-step, 15-minute forecast horizon.
- Use only development data to construct calibration sets or choose settings. Never use held-out labels, ranges, or performance to design a sampler.
- Keep old artifacts and reports identifiable. New runs must not overwrite the bundled application models or historical result files.
- Produce one reproducible study, without adding an experiment-tracking service, a new app feature, or a model-architecture sweep.
- This document authorizes no execution or submission by itself. The current requested deliverable is the plan.

## 1. Research position and scope

The intended contribution is a controlled empirical study of calibration coverage in a small, imbalanced forecasting task, supported by Android measurements. It is not a claim to invent post-training quantization or to establish clinical usefulness.

Three approaches were considered: prose-only revision, a controlled calibration study using the existing CNN, and a broad comparison of architectures/datasets/quantizers. Use the controlled study: it directly answers the accepted question while retaining an interpretable number of experimental factors.

The minimum complete paper covers one model architecture, Weinstock, and one physical Android device. Its conclusions must name that scope. A second, previously unused public cohort is a valuable confirmation extension, but it is not a prerequisite for reporting an honest single-cohort study. Multiple architectures, QAT, FP16, energy measurements, and new sensor integrations are outside the core plan.

### Hypotheses, stated before new evaluation

- **H1 — coverage at a fixed budget:** At 200 calibration windows, broader input-range coverage improves mean recall across the four directional classes relative to random selection with the same participant allocation.
- **H2 — sample-count interaction:** Increasing the calibration budget from 200 to 1,000 has a different effect under range-aware sampling than under random sampling. Estimate the interaction rather than assuming larger sets always help.
- **H3 — mechanism:** Input clipping and rounding account for part of the INT8 change; internal quantization may explain additional changes. Wider ranges may also worsen resolution, so reduced clipping need not improve recall.
- **H4 — cost:** Additional coverage may improve prediction without materially changing the deployed graph, while increasing calibration preparation cost. Measure this; unchanged costs or absent benefits are valid findings.

Do not require any hypothesis to succeed to finish the paper. Report negative and inconclusive results, and do not search for favorable settings after inspecting the evaluation.

## 2. Evaluation protocol

### Correct the split before comparing calibration

The current dataset emits every eligible window from each formatter frame. GlucoBench exposes `train_idx`, `val_idx`, `test_idx`, and `test_idx_ood`, and adds historical prefix rows to some frames. Use those original row identities to distinguish context from scored regions; do not infer ownership from glucose values.

- Assign non-overlapping row roles: training rows first; validation target-region rows exclude training rows; temporal-test target-region rows exclude both training and validation rows. Held-out-participant rows remain separate.
- Keep full segment history available for constructing windows, but require the anchor and all three forecast steps to belong to the scored region. Earlier-split readings may serve as historical inputs only.
- Preserve `(participant, segment, anchor_time, target_time)` for every window. Assert disjoint scored `(participant, id_segment, target_time)` keys across splits, not just disjoint input-array objects. Shared clock timestamps across different participants are valid.
- Report **held-out-participant evaluation** as primary and **temporal evaluation on training participants** as secondary. Never concatenate them into the principal metric.
- Verify that no held-out participant contributes to training or calibration. Report actual eligible participant counts after filtering.

The historical Weinstock test informed the five-class decision. Describe this phase as a **locked, corrected reanalysis**, not a newly untouched test. Retrain from scratch with the five-class target fixed. A different random split of previously used data does not erase the prior exposure.

### Control interpolation

GlucoBench interpolates before splitting. Interpolated inputs can depend on later readings, and interpolated endpoints can change labels.

- Recover an observation mask by matching the formatter grid against the original nonmissing readings using `(participant, rounded_time)` and the formatter's timestamp-rounding rules, with segment consistency checked. Do not match globally on timestamps or mark a row as observed merely because it exists after interpolation.
- **Primary training, calibration, and evaluation:** use windows whose 12 inputs and future endpoint are all originally observed. Keep the existing segment and cadence checks. The intermediate forecast steps must remain in the assigned region even though they are not label inputs.
- **Sensitivity evaluation:** evaluate the same frozen artifacts on formatter windows containing imputation. Report disjoint groups: observed inputs/endpoints; imputed inputs with observed endpoints; any imputed endpoint. This is an offline sensitivity analysis, not evidence of causal live inference.
- Before model fitting, inspect training/calibration class support and unlabeled evaluation eligibility counts only. Reveal held-out class counts together with the final locked evaluation. If filtering removes a class from an evaluation component, its four-class metric is not estimable. If fewer than 1,000 calibration windows remain, report that budget as infeasible. Do not fill either gap by silently changing the protocol or sampling duplicate windows.

### Freeze training and data access

- Keep `TrendCNN` at 2,909 parameters, inverse-frequency weighted cross-entropy, Adam at `1e-3`, batch size 64, and 20 epochs.
- Use training seeds `0, 1, 2, 3, 4`. Record Python/NumPy/PyTorch RNG settings, determinism settings, device, and any remaining nondeterminism.
- Reuse a single fixed, audited participant/temporal split across training seeds. A training seed must not silently change the data split.
- Use the eligible validation windows as the calibration pool. No sampling rule uses their future labels. Validation loss may be logged; retain the final epoch without choosing checkpoints from evaluation scores.
- Develop the runner on synthetic data and development windows. Freeze the protocol, sampler definitions, environment, missing-class rules, and artifact manifest before inspecting new held-out labels, class counts, or metrics for the complete prespecified matrix.
- No adaptive early exit when results look favorable. Record every planned run, failure, and exclusion.

## 3. Calibration experiment matrix

Hold the model weights fixed within each seed while changing calibration inputs. Use the same calibration index sets across all five model seeds.

| Strategy | Definition | Budgets | Calibration draws |
|---|---|---|---|
| Sequential | First windows in deterministic `(id, id_segment, time)` order; historical-method reference | 200, 1,000 | One deterministic order |
| Uniform random | A seeded permutation of all eligible calibration windows | 200, 1,000 | 101, 102, 103 |
| Participant-balanced random | Round-robin participant allocation; random windows within each participant | 200, 1,000 | 101, 102, 103 |
| Participant-balanced range coverage | Exactly the same participant allocation as the preceding arm; spread selection across input-range cells within each participant | 200, 1,000 | 101, 102, 103 |

The participant-balanced random arm is a necessary control: it separates changing participant representation from changing within-participant input-range coverage.

### Sampler rules

1. For each window compute its input minimum and maximum; do not use the future endpoint or class.
2. Compute the 10th and 90th percentiles of each feature from the calibration pool only. Their Cartesian product defines up to nine low/middle/high range cells. Record the numerical edges and handle coincident edges by merging empty/duplicate bins.
3. For each draw, fix a shuffled participant order. Both balanced arms cycle through this order, skipping exhausted participants, so their selected per-participant counts match at each budget.
4. The balanced-random arm uses a fixed random permutation within each participant. The range arm cycles through that participant's nonempty range cells in a seeded order, selecting unused windows from a fixed permutation in each cell and skipping exhausted cells.
5. Construct an ordering through 1,000 windows once and take its first 200 for the smaller budget. Apply the same nested-set rule to uniform and sequential sampling.
6. Select without replacement. Save the selected window keys, seed, strategy, actual count, and calibration-pool hash locally.

This produces **100 INT8 artifacts**: five model seeds × [two sequential runs + three stochastic strategies × two budgets × three draws]. Also export five matched float artifacts. Cache predictions so analysis changes do not rerun inference.

### Measure coverage, not just the requested strategy

For each calibration set record participant and segment counts, windows per participant, unique input-reading count, input minimum/maximum/quantiles, and occupied range cells. Overlapping windows can make a nominal 1,000-sample set cover few distinct readings.

For each INT8 artifact record input/output dtypes, scales, zero points, representable input range, operator inventory, and SHA-256. Report low/high clipping separately on calibration and each evaluation component.

Use the actual quantization operation to count clipping:

```text
q_raw = round(x / scale) + zero_point
clipped = (q_raw < dtype_min) or (q_raw > dtype_max)
q = clip(q_raw, dtype_min, dtype_max)
x_qdq = scale * (q - zero_point)
```

Match the existing Python/Kotlin rounding behavior. Boundary-code occupancy alone is not evidence of clipping. Report clipped input-slot fraction and the fraction of windows with any clipped input, including breakdowns by true class for evaluation analysis only.

## 4. Outcomes, controls, and uncertainty

### Primary endpoint and contrast

The minority classes are fixed as `falling_fast`, `falling`, `rising`, and `rising_fast`; do not redefine them from each split's prevalence.

```text
minority_macro_recall = mean(recall[c] for c in the four directional classes)
primary_delta = minority_macro_recall(range_coverage_200)
              - minority_macro_recall(participant_balanced_random_200)
```

Compute the primary delta on the observed-only held-out-participant component, paired by model seed, calibration draw, and evaluated windows. Report absolute percentage-point changes and the individual class results.

Secondary outcomes are five-class macro recall, minority macro-F1, per-class precision/recall/F1/support, overall accuracy, predicted class proportions, and confusion matrices. Accuracy cannot determine the preferred strategy. Report the corresponding temporal results separately.

Secondary contrasts include range coverage versus uniform random, 1,000 versus 200 within each strategy, and the count-by-range interaction. Mark their intervals as descriptive; do not promote the most favorable secondary comparison to the primary result.

### Baselines and mechanism controls

- **Constant stable:** retain the existing prevalence baseline.
- **Recent slope:** compute `(g_t - g_(t-15)) / 15` using input readings only and map it through the same five thresholds. This tests whether the CNN adds information beyond slope persistence.
- **Logistic regression:** fit a multinomial classifier on the same 12 readings with training-only standardization, balanced class weights, fixed `C=1`, and no test tuning. Record convergence and keep the same split and label rules.
- **Float parity:** compare the PyTorch checkpoint with its float LiteRT artifact before using that pair in an INT8 comparison.
- **Float input-grid control:** feed `x_qdq` through the float artifact using each corresponding INT8 input grid. Float versus float-QDQ measures the input-grid intervention; float-QDQ versus INT8 describes the remaining internal/output quantization difference. Do not interpret the differences as an additive causal decomposition of every error.
- **Clip-only diagnostic:** also run float on inputs clipped to the INT8 representable range without rounding. This distinguishes range truncation from input-grid resolution. Record the distinction between real-valued range exceedance and actual rounded-code clipping.

Report whether broader sampling changed measured coverage. A sampler name alone cannot establish the proposed mechanism.

### Statistical reporting

- Use 2,000 paired participant-cluster bootstrap draws with a fixed analysis seed, preserving every selected participant's complete windows and paired model predictions.
- In each bootstrap draw, recompute metrics from the sampled confusion counts, then average paired differences across the prespecified model seeds and calibration draws.
- Report percentile 95% intervals conditional on those fitted models/calibration sets, plus separate model-seed and calibration-draw variation. Do not treat their Cartesian product as independent participants.
- Do not bootstrap individual overlapping windows. Do not interpret thousands of windows as thousands of independent people.
- If a class has zero support, report the metric as undefined. Report the fraction of bootstrap draws with undefined endpoints; if it exceeds 5%, omit that interval and state the support limitation instead of substituting zero recall.
- Preserve participant-level summaries locally and publish aggregate summaries. Participant-macro descriptive results must report per-class contributing participant counts rather than treating missing classes as failures.

## 5. Deployment-cost protocol

Separate **offline preparation** from **device deployment**.

| Cost | Measurement |
|---|---|
| Calibration preparation | Sampler time and observer-calibration time, separately from export time, with host/environment recorded |
| Serialized artifact | Exact bytes for every float and INT8 model |
| Model initialization | Time to create a usable classifier, reported separately from steady-state inference |
| Classification-call latency | An outer timer around `classify(window)`, including quantization, execution, dequantization, softmax, and class selection |
| Existing inference-path latency | The current internal `latencyNanos`, which includes buffer/conversion work but stops before softmax; never label it kernel-only latency |

The present manuscript says the recorded timings include softmax, but `TrendClassifier.classify` stops its timer before softmax. Correct that historical description and clearly distinguish the new measurement scope.

### Phone workload

- Use the available Galaxy S22 Ultra and real-device `CompiledModel` CPU path. Record actual model identifier, SoC, Android build, runtime versions, app build mode, native-library hash, and accessible thread/backend settings at execution time.
- Benchmark seed `0` and draw `101`: one float artifact plus all four strategies at both budgets, for **nine artifacts**. This selection is prespecified and independent of predictive performance.
- Use one frozen synthetic input workload, generated by the existing synthetic-trace code, for every artifact. Include changing trajectories and range extremes. Clearly separate synthetic timing data from research accuracy data.
- Run 30 paired rounds. Randomize artifact order within each round; after loading each artifact perform 100 warm-up calls followed by 200 timed calls on the same ordered inputs. That gives 6,000 timed calls per artifact.
- Collect raw durations, initialization durations, round/order, device thermal status, and relevant environmental conditions. Keep the explanation model unloaded, background work minimized, and charging/screen conditions fixed and documented.
- If a round encounters an execution error or thermal throttling, mark and retain its log, then repeat the entire paired round under the stated conditions. Do not remove individual slow calls based on their duration.
- Report mean, median, p95, and paired round-level differences with uncertainty. State the p95 estimator. Repeated calls on one phone do not provide evidence across devices.
- Verify artifact hashes and numerical parity for every benchmarked artifact before timing. Include synthetic quantization-boundary cases; do not assume the old golden-vector file covers new models.
- Measure initialization in fresh processes separately; call it process-cold initialization and disclose that filesystem/OS caches are uncontrolled.

Do not claim battery, energy, or total-app memory savings from model bytes or latency. Add those measurements only with a justified measurement method. A result of indistinguishable INT8 costs across calibration strategies is useful and must be reported without claiming equivalence from a nonsignificant difference.

## 6. Implementation tasks and acceptance checks

### Task 1 — Freeze the study and reproduce the environment

**Files:** This plan; `requirements.txt`; `conversion/requirements.txt`; new `paper/results/calibration-coverage/protocol.json` and `environment.txt` at execution time.

- [ ] Record the GlucoEdge and GlucoBench commits, configuration hash, data-file hashes, split identity, exclusions, seeds, sampling definitions, metrics, and measurement workloads.
- [ ] Reconcile the documented training pins with the conversion environment's recorded version drift. Verify a clean environment can train, convert, and predict a synthetic sample; record the working versions without assuming the old pins reproduce the shipped artifacts.
- [ ] Store new checkpoints, per-window outputs, and raw data-derived manifests under `results/calibration-coverage/<run-id>/`. Publish only reviewed aggregate reports and reproducibility metadata under `paper/results/calibration-coverage/`. Public split identity means hashes and aggregate counts; participant IDs, timestamps, selected window keys, per-window predictions, participant-level summaries, and absolute host paths remain local in gitignored `results/`. Sanitize environment exports before publication.
- [ ] Measure one synthetic/development conversion and prediction pass to estimate total runtime. Log failed runs and resume by run ID; never overwrite the old models.

**Acceptance:** A frozen protocol and working environment exist before new held-out scores are computed. No source data are committed.

### Task 2 — Build eligible windows with provenance

**Files:** Modify `training/dataset.py`, `training/train.py`; add a small `training/splits.py` adapter if needed; extend `tests/test_dataset.py` and `tests/test_train.py`.

- [ ] Preserve the current `(x, y)` training interface and add aligned window metadata for participant, segment, anchor, target, split role, and observation status.
- [ ] Implement the region and observation-mask rules in Section 2, without editing GlucoBench. Wire training, calibration, and evaluation to the same rules.
- [ ] Add a synthetic regression case where formatter prefix rows appear in adjacent frames: retain causal history, reject context-only targets, and assert no scored `(participant, id_segment, target_time)` key occurs in two splits. Verify that different participants sharing timestamps remain distinct.
- [ ] Add cases for a future endpoint crossing a split boundary, separate participants with identical glucose arrays, a segment gap, and an interpolated input/endpoint. Verify masks remain aligned with labels after filtering.
- [ ] Run the focused tests and generate the data-availability audit, with exclusion counts, training/calibration class counts, and unlabeled evaluation-component counts. Add held-out class support only during Task 4's locked evaluation.

**Check:** `pytest tests/test_dataset.py tests/test_train.py -v`

**Acceptance:** All provenance tests pass, calibration budgets are feasible or explicitly marked unavailable, and held-out labels remain uninspected. Preserve the existing label-boundary and segment-safety tests.

### Task 3 — Add deterministic calibration selection

**Files:** Extend `conversion/convert.py`; add `tests/test_calibration.py`.

- [ ] Separate index selection from model conversion so samplers can be tested without loading the LiteRT toolchain.
- [ ] Implement the four strategies, nested budgets, shared participant allocations, and recorded range-cell definitions from Section 3.
- [ ] Test repeatability, exact unique counts, 200-in-1,000 nesting, identical participant counts across the balanced arms, and selection invariance when future labels are shuffled.
- [ ] Test exhausted participants/cells and degenerate quantile edges. Require an explicit infeasibility error for a budget exceeding the eligible pool.
- [ ] Expose explicit checkpoint, calibration selection, and output paths so matrix runs cannot overwrite one another.

**Check:** `pytest tests/test_calibration.py -v`

**Acceptance:** Samplers satisfy their contracts on synthetic cases; conversion of one development artifact records its actual selected inputs and metadata.

### Task 4 — Execute the matrix and produce analysis

**Files:** Reuse `training/train.py`, `conversion/common.py`, and `conversion/benchmark.py`; create `experiments/calibration_coverage.py`, `experiments/analyze_calibration_coverage.py`, and `tests/test_calibration_analysis.py`.

- [ ] Train the five fixed-seed checkpoints and fit the two input-dependent baselines. Save exact commands and metadata per run.
- [ ] Export the five float and 100 INT8 artifacts. Run the complete locked evaluation with cached paired predictions and the input-grid/clip-only controls. Reveal and publish aggregate held-out class support at this point; apply the frozen missing-class rule without retuning the protocol.
- [ ] Keep sampler time, observer time, export time, and held-out inference time as distinct fields. Record graph/operator changes rather than attributing them to precision automatically.
- [ ] Implement aggregate metrics and paired participant bootstrap from Section 4. Test hand-computable confusion matrices, missing-class handling, identical-prediction zero deltas, and preservation of whole participant clusters during resampling.
- [ ] Test quantization at the two boundaries and just inside/outside them, plus rounding ties. Require the float-QDQ control to use the artifact's actual scale and zero point.
- [ ] Emit aggregate JSON and publication tables/figures from one analysis command. Record all planned cells, including infeasible or failed cells, instead of dropping them.

**Check:** `pytest tests/test_calibration.py tests/test_calibration_analysis.py -v`, then the complete `pytest -v` suite.

**Acceptance:** Every primary number can be regenerated from saved outputs; the primary contrast is reported even if it is unfavorable. Training/conversion runs remain separate from fast CI.

### Task 5 — Measure deployment costs and parity

**Files:** Extend `android/app/src/main/kotlin/com/glucoedge/app/inference/TrendClassifier.kt` only as needed to load explicit experiment files; add `android/app/src/androidTest/kotlin/com/glucoedge/app/CalibrationBenchmarkTest.kt`; reuse `GoldenParityTest.kt`, `QuantizationMath.kt`, and `conversion/export_golden_vectors.py`.

- [ ] Provide a narrow experimental file-loading path that reuses the current inference engine; do not add nine models to the production model picker or overwrite bundled assets.
- [ ] Export artifact-specific synthetic parity vectors and hashes. Retain the existing tests for the shipped artifacts.
- [ ] Implement outer-call timing and raw round logging in the instrumentation benchmark, preserving the definition of the internal timing field.
- [ ] Execute the fixed nine-artifact protocol from Section 5 on the real phone, including initialization measurements and artifact provenance checks.
- [ ] Generate cost summaries and latency plots using the recorded rounds. Report backend/runtime limitations directly.

**Checks, from `android/`:**

```bash
./gradlew :app:testDebugUnitTest :app:check
./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.glucoedge.app.GoldenParityTest
./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.glucoedge.app.CalibrationBenchmarkTest
```

**Acceptance:** Every timed artifact passes parity and has equal valid measurement counts. Emulator timing cannot substitute for the physical-device result.

### Task 6 — Rewrite the paper around the evidence

**Files:** `paper/main.tex`, `paper/references.bib`, `paper/README.md`; generated aggregate figures/tables under `paper/results/calibration-coverage/`.

- [ ] Use the neutral working title **“GlucoEdge: Calibration Coverage and Quantization Trade-offs in On-Device Glucose Trend Classification.”** Revise it only to reflect the actual measured scope, not to imply a positive result.
- [ ] Rewrite the abstract after analysis: question, controlled design, primary effect with uncertainty, mechanism evidence, device cost, and explicit scope limitation.
- [ ] Center the introduction on the gap between nominal calibration sample count and measured coverage under imbalance. Separate established PTQ knowledge from this study's contribution.
- [ ] Refocus related work on calibration/range estimation, class-wise quantization effects, CGM evaluation protocols, and device measurement. Search primary literature at execution time and avoid unverified “first” or novelty claims.
- [ ] Describe the corrected split, observation mask, historical test exposure, fixed training settings, samplers, and primary contrast before presenting results.
- [ ] Organize results as: coverage manipulation; minority-class effects; count interaction and mechanism controls; deployment costs; sensitivity analyses.
- [ ] Replace the headline historical accuracy/timing table with corrected results. Keep old measurements only in an explicitly historical comparison with their original provenance; never mix old and new denominators.
- [ ] Move detailed emulator troubleshooting and app implementation notes to an appendix. Keep parity, provenance, and the absence of medical claims in the main text.
- [ ] State whether the CNN outperforms recent slope/logistic regression. If it does not, retain that result and narrow the contribution to quantization behavior rather than superior forecasting.

**Planned tables and figures:**

| Artifact | Purpose |
|---|---|
| Table: eligible data and split audit | Show participant/class counts and interpolation exclusions |
| Table: experiment matrix and measured coverage | Distinguish nominal sample count from actual coverage |
| Figure: calibration input distributions and quantized ranges | Show whether each sampling intervention changed the expected range |
| Figure: minority recall versus calibration budget | Show matched strategy effects and uncertainty |
| Figure: per-class float / clip-only / float-QDQ / INT8 changes | Test the proposed error mechanism |
| Table: baselines and primary/secondary contrasts | Establish forecasting value and report the prespecified comparisons |
| Table or figure: device and preparation costs | Show predictive gains alongside artifact bytes, latency, and offline cost |

Use standard plotting tools for standalone publication figures. Export exact values alongside figures. Do not add probability-calibration terminology to describe quantizer range calibration.

**Acceptance:** Every central claim has a traceable result, every plotted comparison uses matched data, and negative results remain visible. No fixed page target overrides readability or evidence.

### Task 7 — Verify and prepare the arXiv package

**Files:** `paper/main.tex`, `paper/references.bib`, `paper/main.bbl`, `paper/main.pdf`, `paper/README.md`, and a regenerated `paper/glucoedge-arxiv-source.zip`.

- [ ] Build from `paper/` with `tectonic --keep-intermediates --keep-logs main.tex` and resolve missing citations/references and material layout warnings.
- [ ] Inspect every rendered page, including figure labels, table widths, uncertainty notation, and references. Check the new source against all table/figure exports.
- [ ] Update title, abstract, author metadata, date, page/figure/table counts, and reproducibility links in the README.
- [ ] Rebuild the source ZIP from an explicit list of current source, bibliography, and required figure files. The existing ZIP predates the current manuscript and must not be reused.
- [ ] Extract the package into a temporary directory, compile independently, and compare its sources/bibliography and resulting content with the approved manuscript. Include no raw CGM data, caches, logs, or experiment checkpoints.
- [ ] Prepare suggested arXiv metadata and document any remaining limitations. Account access, endorsement, license selection, and final public submission are later author actions, outside this plan's execution deliverable unless separately requested.

**Acceptance:** The paper and source package are reproducible and reviewable. “Ready to submit” means prepared for author submission, not guaranteed arXiv acceptance.

## 7. Optional confirmation after the core study

If an unused public cohort is available with adequate observed five-class support, freeze its preprocessing and apply the locked Weinstock float, balanced-random-200, and range-coverage-200 artifacts without retraining, target-data calibration, or threshold tuning. Audit prior exposure before calling it external confirmation. Do not assume the previously used iglu smoke-test dataset is untouched.

If transfer support is inadequate or accuracy is poor, report that limitation. A separately trained within-cohort replication is a different experiment and must be named as such. This extension must not hold up completion of an explicitly single-cohort paper.

## 8. Completion gates and execution order

1. **Protocol/environment gate:** fixed question, matrix, endpoint, environment, and data-use record.
2. **Validity gate:** eligible-window tests pass and class-support/provenance audit is complete.
3. **Evidence gate:** complete matrix, controls, baselines, uncertainty, and logged failures.
4. **Deployment gate:** physical-device parity and controlled cost measurements.
5. **Manuscript gate:** conclusions match the observed evidence and explicitly describe reanalysis/generalization limits.
6. **Packaging gate:** verified PDF, reproducible sources, current ZIP, and submission metadata.

Tasks 1–4 are sequential. Task 5 can proceed after the fixed benchmark artifacts exist while statistical analysis continues. Literature verification can run in parallel with experiments. Final results writing and packaging follow the completed evidence; no estimated experiment duration is a substitute for these gates.

## 9. Amendments — 2026-10-10

Grounded in the committed split audit (`experiments/split_audit.py`,
aggregate output `paper/results/split_audit.json`, run 2026-10-09). These
amendments were adopted before any held-out label, class count, or metric
was inspected, so the locked-evaluation rule is intact.

1. **Observation mask: no-gap-fill is primary; exact-observed moves to
   sensitivity.** The audit measured that only 75.8% of formatter grid rows
   coincide exactly with an original reading — minute rounding and
   re-gridding drift the grid relative to reading times — so the
   exact-observed window rule in Section 2 would discard roughly 40--48% of
   windows for a reason unrelated to gap filling. A no-gap-fill rule (every
   grid value lies on or between two original readings at most 7.5 minutes
   apart, the threshold the committed audit uses) keeps 95.6% of rows and
   about 77% of windows while still excluding every value that fills a
   missing reading. Primary training, calibration, and evaluation windows
   use the no-gap-fill mask. The sensitivity groups become: exact-observed
   subset; gap-filled inputs with no-gap endpoints; any gap-filled endpoint.

2. **Sequential versus uniform random is a prespecified secondary
   contrast.** The audit showed the historical sequential-200 selection is
   one participant and one segment, with 178 of 200 windows duplicating
   training windows. The cost of the toolchain-default protocol is itself a
   reportable quantity: report
   `minority_macro_recall(uniform_random_B) − minority_macro_recall(sequential_B)`
   at both budgets with the same paired participant bootstrap, rather than
   leaving sequential as an unanalyzed reference row.

3. **Observer arm as a mechanism control.** The shipped artifact's
   activation ranges come from a histogram observer (per-tensor affine), so
   range under-coverage is confounded with the observer's range-estimation
   rule. Add a min--max-observer variant for exactly two cells —
   sequential-200 and range-coverage-200, calibration draw 101 — across the
   five model seeds (10 additional INT8 artifacts, matrix total 110). Report
   it descriptively alongside H3's input-grid and clip-only controls; it is
   not a new primary comparison.

## Reference starting points

- [GlucoBench, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4c12e97f2e05304a451e18c9c945036f-Abstract-Conference.html): upstream benchmark and protocol context.
- [Hubara et al., Accurate Post Training Quantization With Small Calibration Sets](https://proceedings.mlr.press/v139/hubara21a.html): prior calibration-sensitive PTQ research; does not establish this study's proposed outcomes.
- [Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization](https://proceedings.mlr.press/v119/nagel20a.html): rounding as a distinct quantization consideration, already cited by the manuscript.
- [arXiv LaTeX submission guidance](https://info.arxiv.org/help/submit_tex.html): check again when packaging; the currently documented process accepts bibliography source or a compatible prebuilt `.bbl`.
