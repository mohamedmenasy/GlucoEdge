# Improving GlucoEdge's predictive results

## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Mode: experiment planning, informed by source inspection and local diagnostics
- Date: 2026-10-13
- Version: improvement_proposal_v1
- Verification: synthetic diagnostic executed; existing prediction outputs reanalyzed descriptively. Proposed training changes are **untested**.
- Baseline inspected: repository commit `2415fa7`, shipped sweep-r2 CNN seed 1.
- Scope: research engineering demonstration; not a medical device or treatment system.

**Recommendation: preserve temporal position in the small CNN, and compare candidates at a controlled stable-versus-directional tradeoff.** This is a more informative next experiment than another broad capacity or data sweep. A true ordinal head is the next independent hypothesis. No new predictive improvement has been established in this review.

## What the existing results establish

| Experiment | Observed result | Implication for the next experiment |
|---|---|---|
| Calibration coverage | Uniform-200 recovers approximately 13.4 points of directional recall lost by sequential-200 in the original absolute-input CNN. | A solved deployment defect; unlikely to provide another large predictive gain in the current delta model. |
| Difference encoding plus anchor | Five-seed float directional recall increases from approximately 0.487 to 0.509. | Preserve the successful difference representation as the initial control. |
| Sweep r1 | Held-out macro-F1 improves by 5.68 points, while the selected INT8 model's directional recall falls from approximately 0.517 to 0.438. | Optimizing a different metric can move the operating point without improving the intended behavior. |
| Sweep r2, currently shipped | Directional recall 0.52339, stable recall 0.45855; directional gain over the previous champion 0.596 points, CI approximately [-0.056, +1.331] points. | Useful candidate, but the interval includes zero. Its promotion gate was not a strict superiority test. |
| Sweep r3, rejected | Directional recall 0.53112, stable recall 0.41447. Relative to r2: +0.774 and -4.408 points respectively. | More directional recall came with a large stable-recall loss; compare tradeoff curves before attributing a gain to architecture. |

The encoding rows are five-seed means; the r2/r3 rows are their selected individual artifacts. These are different comparisons, not a single sequential learning curve.

Sources: [delta results](../../results/calibration-coverage/delta_inputs_r1.json), [r1](../../results/calibration-coverage/accuracy_sweep_r1.json), [r2](../../results/calibration-coverage/accuracy_sweep_r2.json), [r3](../../results/calibration-coverage/accuracy_sweep_r3.json).

## 1. A verified architectural limitation: some recency information disappears

The shipped CNN has two kernel-3, stride-one convolutions, then global average pooling. Each pre-pooling feature sees only five encoded positions. Pooling retains which local patterns occurred, but can erase where an interior pattern occurred. The last absolute glucose value divided by 16 is appended as a time position alongside the differences, rather than supplied separately.

I tested two synthetic histories sampled at minutes -55, -50, ..., 0:

- A: five readings at 150, then seven readings at 160; the rise is first observed 30 minutes before the anchor.
- B: seven readings at 150, then five readings at 160; the rise is first observed 20 minutes before the anchor.

Both end at 160 and have one +10 difference, but at different positions. The checkpoint and **both bundled LiteRT artifacts produce exactly identical output vectors** for A and B. The maximum absolute difference within each model is 0.0. The bundled artifact hashes match the frozen r3 champion and r2 seed-1 artifacts.

This proves a representational ambiguity, not an accuracy deficit on these particular examples: their future labels are unspecified. The CNN is not generally permutation invariant; local order, boundaries, and proximity to the anchor can affect it. Nor does this example prove the same invariance in the longer-receptive-field TCN.

**First ablation:** retain the current 12-value encoding, convolutions, class-weighted cross-entropy, training data, and five seeds. Replace `[144,12] -> average -> [144] -> Linear(144,5)` with `[144,12] -> flatten -> Linear(1728,5)`. This raises parameters from **42,725 to 50,645**, only **7,920 additional parameters / 18.54%**. It allows each local feature's contribution to depend on when it occurred, while preserving the five-logit client interface. Actual export size and latency still need measurement.

A useful compact comparator is a position-aware MLP on the same 12 encoded values: `12 -> 64 -> 64 -> 5`, with 5,317 parameters. It helps test whether convolution and pooling are helpful for this short fixed-length task. It is an optional control, not a claim that MLPs will win.

Only after this isolated test should a separate-anchor or multiscale-context model be considered. Appending an anchor to the difference sequence is a heterogeneous input representation; a separate level branch is plausible, but combining it with the first ablation would obscure which change helped. Simply taking the last feature of the existing two-layer CNN would discard most of the hour and is not an equivalent fix.

Evidence: [model](../../../training/model.py), [encoder](../../../training/deltas.py), [executed diagnostic](recency_collision.json), [reproduction script](reproduce_recency_collision.py).

## 2. Treat decision offsets as a control, not an automatic accuracy gain

The model is trained with balanced class weights and decoded by plain argmax. A scalar bias on the stable logit provides a cheap way to vary its operating point:

`prediction = argmax(logits + [0, 0, b, 0, 0])`.

Increasing `b` generally increases stable recall and decreases directional recall. This is decision-threshold adjustment, distinct from INT8 representative-data calibration and from probability calibration. A common positive temperature applied to all logits cannot change argmax decisions.

For a future experiment, fit `b` only on a designated validation calibration partition, then evaluate it on different validation participants. Compare both the champion and candidate with the same adjustment procedure. Do not improve only the candidate's decision rule and attribute the difference to its representation. Balanced class weights already alter the effective prior, so blindly applying a second inverse-frequency correction is not justified.

The cached-logit diagnostic in this folder uses previously evaluated labels to inspect the tradeoff. It is **post-hoc and exploratory**, and cannot turn the rejected r3 artifact into a confirmed improvement. Its numerical results and reproducibility details are recorded in [operating_points.json](operating_points.json) and [inspect_operating_points.py](inspect_operating_points.py). Labels, participant identifiers, and window keys were checked for exact alignment between the two caches; source hashes are recorded.

On the 46,376 held-out windows, scanning stable-logit offsets from -2 to +2 in fixed 0.01 increments gives:

| Artifact / descriptive decision rule | Stable recall | Directional recall |
|---|---:|---:|
| Shipped r2, unchanged | 45.85% | 52.34% |
| Rejected r3, unchanged | 41.45% | 53.11% |
| r2, first grid point satisfying champion stable minus 2 points (`b=-0.07`) | 44.72% | 52.63% |
| r3, same floor (`b=+0.09`) | 44.64% | 52.41% |
| r3, first grid point satisfying unchanged champion stable (`b=+0.13`) | 47.72% | 51.83% |

The two adjusted models near 44.7% stable recall favor r2, so a simple stable offset supplies no compelling reason to rescue r3. INT8 score ties make stable recall jump; the last row overshoots its 45.85% target and is **not** an exact matched-stable comparison. This is a bounded deterministic grid diagnostic, not proof of global curve dominance. The temporal results are also recorded: each model's independent oracle at champion-minus-2 gives r2 50.47% directional / 50.81% stable and r3 50.68% / 50.25%, another tradeoff rather than uniform dominance. No confidence interval or fresh-test claim is attached to these label-selected operating points.

There is a selection mismatch worth correcting prospectively: r3 allowed a **5-point** stable-recall decline on validation, but only **2 points** at confirmation. Under the same 2-point validation rule, the selected TCN seed 2 would have failed: 0.52560 < 0.54846 - 0.02. Seed 1 had virtually identical directional recall (0.53664 versus 0.53672, a 0.0077-point difference) and higher stable recall (0.53608). This does not authorize retrospectively changing the selected seed; it shows why the next selection rule should match its final objective and use a declared tie rule.

A stable constraint aligned across stages is helpful but does not guarantee transfer under cohort shift. Evaluate it on participants omitted from model fitting, and report failures as well as averages. Anchoring the floor to a fixed reference also avoids gradually spending another 2 points of stable recall at each model generation.

The general basis for separating class scores from cost-sensitive decisions is supported by [Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314); constrained objectives expressed through the confusion matrix are studied by [Narasimhan, AISTATS 2018](https://proceedings.mlr.press/v84/narasimhan18a.html). Neither source establishes that a bias correction improves this CGM task.

## 3. Test an ordered or continuous target separately

The five labels are ordered bins of the continuous future rate. Current training discards the within-bin rate and both intermediate future readings. The r1 ordinal smoothing and distance penalty still used an ordinary five-logit head; its design explicitly excluded CORN/CORAL. Their failure is not evidence against all ordinal models.

The cached r2 confusion matrix shows that 71.54% of held-out errors are between neighboring classes (76.82% temporally); 3.90% of all held-out windows are predicted in the opposite non-stable direction (2.81% temporally). These are descriptive error counts. Adjacent errors are dominated by stable-versus-directional confusion, so they motivate checking the ordinal structure but do not establish that an ordinal loss will resolve the tradeoff.

**Low-cost independent arm:** replace the softmax head with CORN's four conditional threshold probabilities. Cumulative products give monotonic exceedance probabilities; differences recover the five class probabilities. Keep the existing label cutoffs and choose the classification decision rule on validation. CORN's usual count-of-probabilities-above-0.5 decoder targets an ordinal median, which need not optimize directional recall. [Shi, Cao and Raschka](https://arxiv.org/html/2111.08851v5) establish the conditional construction and rank consistency, but do not evaluate this CGM task.

**Follow-up if needed:** preserve the five-class head and add an auxiliary robust regression loss on observed future changes at 5, 10, and 15 minutes. Those future rows already lie inside the existing 15-minute target span; they are training targets, never inputs. Discard the auxiliary head for deployment. Freeze the target scales and a small loss-weight grid using development data. This adds supervision about trajectory shape without changing the research question or final class thresholds.

A more ambitious alternative predicts a conditional distribution of the future rate and integrates it over the unchanged bands. Predicting only the conditional mean and binning it risks collapsing uncertain cases toward stable. [Gluformer](https://arxiv.org/abs/2209.04526) motivates non-Gaussian distributional CGM prediction, but its density-model advantage does not guarantee better point error or this classifier's recall. This remains a hypothesis, not a promised gain.

## 4. Make the next comparison capable of finding a real improvement

The existing trainer uses Adam at a constant learning rate of 0.001 and returns the final epoch. It prints validation loss but neither saves the best validation checkpoint nor decays the learning rate. Therefore the code does not establish that any architecture was evaluated near its best development performance. This is an optimization opportunity, not proof that the current models are undertrained.

Use a small declared learning-rate schedule and save epoch checkpoints. Select the checkpoint using the same directional objective and stable constraint as the model selection. Apply identical optimization budgets and checkpoint rules to the baseline and new head. In a staged experiment, first compare heads under the existing recipe to isolate architecture; then compare a changed training schedule against a baseline retrained under that schedule. Avoid selecting the best of many epochs on the final test set.

Suggested bounded first study:

1. Identify participants unused by any prior model fitting and development, using stored provenance. Establish development folds grouped by participant; preserve segment boundaries and the existing no-gap-fill eligibility rule. If no genuinely untouched final set is available, label the new study exploratory.
2. Compare global pooling versus a flatten head under the same current training recipe and five paired seeds. Optionally add the compact MLP control. Keep the current 12-reading horizon setup and uniform-200 conversion calibration.
3. Record per-epoch directional recall, stable recall, macro-F1, per-class precision/recall and confusion matrices. Compare operating curves in development; choose one decision rule and one final candidate before accessing the final set.
4. If position preservation helps across participant folds, evaluate softmax versus CORN using a small factorial comparison, or add the auxiliary continuous target as a separate arm. Do not stack untested changes without controls.
5. Evaluate the frozen selected INT8 artifact against the frozen champion on the final participants. Report paired participant-cluster uncertainty, the five-seed results, and the difference between the average training recipe and a selected individual model. Include stable recall, macro-F1, directional precision, opposing-direction errors, model bytes and measured latency.

The primary proposal is directional recall at a prespecified stable-recall constraint. A useful practical target is a reproducible 1–2 point gain with no stable-recall deterioration; this is a proposed success target, not an expected effect size. A claim of superiority requires evidence for improvement, rather than merely passing the old CI lower-bound > -0.5-point non-inferiority-style gate.

Weinstock's held-out and temporal results, and PEDAP's repeatedly read secondary results, are already exposed. Writing a new protocol does not restore their independence. They can serve as regression checks, while a genuinely untouched participant set or outer evaluation provides stronger evidence. Repeated overlapping windows must not be treated as millions of independent observations.

## What to deprioritize

- More INT8 calibration budgets, observers, or QAT before addressing representation: the current uniform-calibrated model loses only about 0.1 directional-recall points to quantization on the held-out component.
- Another generic increase in channels or indiscriminate addition of RT-CGM: the existing experiments do not support these as the leading opportunities.
- Claiming that longer context or ensembling is universally exhausted: r1 tested them under a different loss and macro-F1 objective. They remain secondary hypotheses, but should be revisited only after position and operating-point controls.
- A large pretrained model as the first step: the local diagnostic supplies a much cheaper specific hypothesis, and transfer-model evidence uses materially different contexts and objectives.

## Reproduce the diagnostics

From the repository root, using the existing environment and local artifacts:

```bash
venv/bin/python paper/analysis/accuracy-review-2026-10-13/reproduce_recency_collision.py
venv/bin/python paper/analysis/accuracy-review-2026-10-13/inspect_operating_points.py
```

The scripts write aggregate diagnostic reports beside themselves. The first uses only synthetic readings. The second reanalyzes cached prediction outputs and labels; it does not train, access new cohorts, or revise any frozen protocol. Both depend on gitignored local research artifacts. The manuscript, training pipeline, frozen results, bundled models, and external GlucoBench clone were not edited by this review.
