contract_role: domain
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: block
block_class: repairable
trigger: "A domain error or omission that changes what a reported result, comparison or contribution claim means but that revision can correct without invalidating the core result"
Basis: W1 and W2 (both Major) meet the committed block condition. Once the toolchain documentation the manuscript itself cites and the PTQ calibration-data literature are represented accurately, the headline framing (a "toolchain-default" selection that damages INT8) and the novelty of the "sharpest observation" change meaning. Both are repairable in revision and leave the sequential-versus-randomized result intact. W3 to W14 are Minor corrections (terminology, quotation accuracy, dataset provenance, quantization detail) that do not change the score. No fatal condition applies: nothing indicates that the evaluated artifacts differ from what is described, and the unit and label definitions are applied consistently.

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

Scope and method. I reviewed the manuscript as a researcher in on-device post-training quantization and CGM forecasting benchmarks, concentrating on PTQ and toolchain claims, how prior work in PTQ calibration and CGM prediction is represented, and the accuracy of quantization arithmetic and dataset descriptions. The clinical meaning of the target and the bootstrap mechanics are left to other seats. I checked toolchain statements against the LiteRT Torch source (litert_torch/quantize/pt2e_quantizer.py and docs/pytorch_converter/README.md in google-ai-edge/litert-torch) and the Google AI Edge pages the manuscript cites. I checked prior-work statements against the cited articles where accessible (Kistkins et al. via PMC10574913, GlucoBench's dataset table, DiaData's Zenodo v3 record and arXiv 2508.09160), and dataset and artifact statements against paper/results/. The technical core is accurate: the quantizer configuration, affine arithmetic, scales and ranges, artifact sizes and the trend-arrow framing all check out. The weaknesses concentrate in how the headline observation is positioned (W1, W2), plus a set of Minor accuracy and provenance corrections (W3 to W14). I found no reviewer-directed instructions in the manuscript or the supplementary files.

### S1: Quantizer configuration and range arithmetic are described exactly
**Evidence Anchor**: text: §5.1 (Shipped artifacts), "only the weights are quantized symmetrically, per channel"
In LiteRT Torch's get_symmetric_quantization_config, static activations use HistogramObserver with per_tensor_affine int8 in [−128, 127], and is_per_channel=True gives per_channel_symmetric weights in [−127, 127] through PerChannelMinMaxObserver. That is exactly what the manuscript describes, including its caveat about the helper's name. The other quantization details are arithmetically consistent: the zero-inclusive range, the shipped scale (0.9408 mg/dL per code with zero point −128, so 255 × 0.9408 = 239.9 mg/dL), the sequential-200 and randomized grids (250.9/255 ≈ 0.984 and 400.8/255 ≈ 1.572 mg/dL per code, hence "60% coarser") and the note that codes below the lowest reading go unused. split_audit.json reproduces the shipped input parameters.

### S2: Terms that collide across communities are disambiguated
**Evidence Anchor**: text: §1, "represented during quantizer calibration, not calibration of predicted"
"Calibration" is pinned to its PTQ sense, and probability calibration is listed as not evaluated. The target is kept separate from commercial trend arrows: it is a future endpoint label, not a device-specific look-back indicator, and the bands are described as a collapse of device taxonomies (Pettus and Edelman 2017; Rodacki et al. 2021). "One-hour history" is corrected to 55 elapsed minutes. These are the distinctions a mixed PTQ and diabetes-technology readership most often gets wrong.

### S3: The mechanism controls separate clipping error from rounding error
**Evidence Anchor**: table: tab:mechanism (Mechanism controls, §7.4)
The clip-only and float-QDQ controls are the right decomposition of input quantization error into range truncation and grid rounding, with the INT8 interior as the residual. The reported components close arithmetically (13.12 + 0.23 = 13.35 points), and the manuscript correctly handles readings at 401 that round into the top code.

### S4: Deployment cost is measured, not inferred from nominal ratios
**Evidence Anchor**: table: tab:devicebench (Nine-artifact device benchmark, §7.15)
Serialized sizes are reported exactly (11,976 versus 17,352 bytes, 31.0% smaller rather than a nominal 4×). INT8 is reported as slower than float on the tested phone. The benchmark records device, OS, runtime version, warm-up, shuffled paired rounds, hash and parity checks and thermal state. Emulator runs are kept out of the performance claims.

### S5: The Weinstock/GlucoBench description is audited and matches the source
**Evidence Anchor**: dataset: paper/results/split_audit.json (participants raw 200 and after_formatter 192; grid_rows)
The 200/192 participant counts match GlucoBench's own dataset table. The grid shares in §4.1 recompute from the audit (439,218 and 554,183 of 579,782 grid rows give 75.8%, 19.8% and 4.4%). Windows are generated within (id, id_segment) groups, so no window spans a sensor gap.

### W1: Sequential calibration is labelled a "toolchain default", contrary to the documentation the manuscript cites
**Severity**: Major
**Evidence Anchor**: text: Abstract, "The toolchain-default sequential selection covers two participants"
**Confidence**: 5 — wording checked on the cited Google AI Edge pages and in the LiteRT Torch repository
The abstract, Introduction, §5.2, §7.8, §8.1 and Conclusion call first-N, dataset-order selection the toolchain default (six occurrences). Neither toolchain chooses calibration data; the user's iterator does. The page cited as googleRepresentativeData (tf.lite.RepresentativeDataset) describes the calibration set as "a small subset of a few hundred samples randomly chosen, in no particular order, from the training or evaluation dataset". The page cited as googleLiteRTQuantization asks for "around ~100-500 samples" of training or validation data. LiteRT Torch's PT2E example fills the observers by running the prepared model on the conversion sample arguments and prescribes no selection order. The paraphrase in §8.1 ("Official guidance asks for a small representative calibration set") drops the random, unordered instruction, which is exactly what the sequential arm violates.
Impact: the 13.4-point contrast stands, but its meaning changes. The study measures the cost of departing from documented guidance, a shortcut that overlapping windows make easy. It does not show a hazard built into toolchain defaults, which is what the abstract and Conclusion state.
Suggested fix: replace "toolchain-default" with a neutral label such as "contiguous first-N (dataset-order) selection". Quote the RepresentativeDataset guidance accurately in §2.4 and §8.1 and say that the sequential arm contravenes it. Frame the contribution as measuring, with participant-clustered intervals and a mechanism control, how large the penalty for that shortcut is in overlapping-window CGM data.

### W2: The PTQ calibration-data literature is not engaged, so the "sharpest observation" is presented as new
**Severity**: Major
**Evidence Anchor**: text: §8.1, "prior PTQ work treats clipping, range estimation, and rounding as the levers"
**Confidence**: 4 — the cited works were verified; the positioning judgment is mine
Calibration-data choice is itself a studied lever. Williams and Aletras (2024) report large swings in downstream performance from calibration-data choice in LLM quantization and pruning, contrary to earlier assumptions of robustness. Yu, Yang and Shi (2021) find that PTQ accuracy tracks the statistical (Gram-matrix) similarity between calibration and source domains. ZeroQ (Cai et al. 2020) synthesizes calibration inputs whose batch-normalization statistics match the training data's, treating the statistics of calibration data, not real samples, as what range setting needs. Hubara et al. (2021), cited here only for "specialized methods for small calibration sets", show that accurate PTQ is achievable with small calibration sets. The two standard white papers on setting ranges from calibration data (Krishnamoorthi 2018; Nagel et al. 2021) are not cited. In general form (representativeness matters, and count beyond a few hundred samples does not) the observation is anticipated. What is new is its controlled measurement on overlapping, participant-clustered CGM windows, its localization to input clipping, and the cohort replication. Norm basis for the severity: the originality criterion of the NeurIPS 2023 Reviewer Guidelines asks whether it is clear how the work differs from previous contributions and whether related work is adequately cited (https://neurips.cc/Conferences/2023/ReviewerGuidelines).
Suggested fix: add a calibration-data paragraph to §2.4 and recast §8.1 as confirmation plus measurement, citing at least: M. Williams and N. Aletras, "On the Impact of Calibration Data in Post-training Quantization and Pruning," Proc. ACL 2024 (Long Papers), pp. 10100–10118, doi:10.18653/v1/2024.acl-long.544; H. Yu, L. Yang and H. Shi, "Is In-Domain Data Really Needed? A Pilot Study on Cross-Domain Calibration for Network Quantization," CVPR Workshops 2021, pp. 3043–3052; Y. Cai, Z. Yao, Z. Dong, A. Gholami, M. W. Mahoney and K. Keutzer, "ZeroQ: A Novel Zero Shot Quantization Framework," CVPR 2020, pp. 13169–13178; R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper," arXiv:1806.08342, 2018; M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen and T. Blankevoort, "A White Paper on Neural Network Quantization," arXiv:2106.08295, 2021.

### W3: The round-four "label-noise ceiling" measures label-definition agreement, not attainable recall
**Severity**: Minor
**Evidence Anchor**: text: §7.12 (round four), "the task is model- and data-limited, not label-noise-limited"
**Confidence**: 5 — the estimator is stated in exp0_sweep_r4.json; the objection is definitional
The "ceiling" in exp0_sweep_r4.json is the agreement between the endpoint label and a least-squares-slope label computed from the same horizon readings (the current sample plus three future samples): 0.954 directional recall. It is supplemented by agreement after synthetic i.i.d. noise of σ = 1 and 2 mg/dL (0.915 and 0.860). Both labels are functions of the future, so their agreement measures how stable the label definition is, not how much of the future is predictable from 55 minutes of past CGM. The 42-point "headroom" therefore bounds nothing a past-only model can reach, and "not label-noise-limited" does not follow. The perturbation also omits what dominates CGM-versus-blood-glucose error: blood-to-interstitial lag, calibration drift and autocorrelated noise. The claim recurs as an Introduction contribution bullet and in the Conclusion.
Suggested fix: rename the quantity "label-definition stability", remove "ceiling" and "headroom", and state that attainable recall from CGM-only inputs is unknown. If label noise is to be probed, perturb with a published sensor-error model such as A. Facchinetti, S. Del Favero, G. Sparacino, J. R. Castle, W. K. Ward and C. Cobelli, "Modeling the Glucose Sensor Error," IEEE Transactions on Biomedical Engineering 61(3), 2014.

### W4: Forecasting evaluation practice is described as RMSE-only
**Severity**: Minor
**Evidence Anchor**: text: §2.3, "Forecasting work evaluates RMSE, where direction surfaces only as a safety penalty on incoherent predicted variation"
**Confidence**: 4 — counter-examples verified; the sentence is a field-level generalization
Rate and direction accuracy are established evaluation targets for CGM prediction. Continuous glucose error-grid analysis pairs point accuracy with a rate error grid (Kovatchev et al. 2004) and has been applied to prediction algorithms to assess the point and dynamic accuracy of predicted series (Zanderigo et al. 2007). The J index of Facchinetti et al. (2011), which this manuscript cites, combines prediction regularity with time gain and was compared against RMSE and CG-EGA. "No published baseline exists for this exact task" remains literally true, but the surrounding text implies that direction goes unexamined in forecasting.
Suggested fix: acknowledge rate-error-grid and time-gain evaluation, and note that forecasts can be decoded into the five classes. GlucoBench's benchmark models are configured for the same Weinstock formatter, so decoding their 15-minute forecasts at ±1 and ±2 mg/dL/min would give a published-model comparator. Cite B. P. Kovatchev, L. A. Gonder-Frederick, D. J. Cox and W. L. Clarke, "Evaluating the accuracy of continuous glucose-monitoring sensors: continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data," Diabetes Care 27(8):1922–1928, 2004, doi:10.2337/diacare.27.8.1922; and F. Zanderigo, G. Sparacino, B. Kovatchev and C. Cobelli, "Glucose prediction algorithms from continuous monitoring data: assessment of accuracy via continuous glucose error-grid analysis," Journal of Diabetes Science and Technology 1(5):645–651, 2007.

### W5: The nearest classification result is quoted outside the cited paper's reported values
**Severity**: Minor
**Evidence Anchor**: text: §2.3, "recalls of 0.85--0.96 at 15-minute and 1-hour"
**Confidence**: 5 — checked against the abstract and Table 2 of the cited article
Kistkins et al. (2023) report best-model recalls of 0.96, 0.91 and 0.98 for hyper-, eu- and hypoglycemia (logistic regression, 15 min) and 0.85 and 0.87 for hyper- and hypoglycemia (LSTM, 1 h). Their Table 2 spans 0.07 to 0.98 across models, with 1-hour euglycemia recall of 0.59 to 0.70. The study has 11 participants and data brought to 15-minute frequency, so its 15-minute horizon is one step ahead. "0.85–0.96" matches neither the abstract (0.85 to 0.98) nor the table. No comparison depends on it, but the misquote flatters the prior result and omits the one-step detail that would strengthen the manuscript's own non-comparability argument.
Suggested fix: report "best-model recalls of 0.85–0.98, with euglycemia recall of 0.59–0.70 at 1 h; 11 participants; 15-minute data (one-step horizon); within-patient splits".

### W6: CGM out-of-range codes (39 and 401 mg/dL) are treated as measurements
**Severity**: Minor
**Evidence Anchor**: text: §7.4, "at the 401 sensor cap"
**Confidence**: 4 — the device listing was verified in GlucoBench's dataset table; the 40–400 mg/dL display range is from device knowledge, and the encoding was not checked in the Jaeb data dictionary
GlucoBench lists the Weinstock data as Dexcom G4. Dexcom systems of that generation display numeric sensor glucose only within 40–400 mg/dL and show LOW or HIGH outside it, so a pool spanning exactly 39–401 mg/dL almost certainly contains LOW/HIGH codes, that is, censored values. The manuscript calls 401 a "sensor cap" but uses it as a reading in inputs, labels and calibration. When a window's current or endpoint value is a code, its rate is only a bound, biased toward smaller magnitudes. Such windows concentrate in the fast classes the mechanism analysis is about (at least the 3.5% of held-out windows that reach 401). The top of the randomized arms' INT8 range (400.8 mg/dL) is set by this code.
Suggested fix: confirm the encoding in the T1D Exchange/Jaeb data dictionary and describe 39 and 401 as censoring codes in §4.1. Report by class the share of windows with a censored input or endpoint, and add a sensitivity tier that excludes censored endpoints or treats their rates as bounds.

### W7: What calibration changes inside the INT8 artifact is misstated
**Severity**: Minor
**Evidence Anchor**: text: §7.15, "changes nothing the device sees except the two quantization parameters"
**Confidence**: 4 — follows from the quantizer configuration and paper/results/calibration-coverage/artifacts_summary.json; the per-op dtype map was not inspected
With HistogramObserver on every annotated activation, calibration sets the scale and zero point of each quantized activation tensor, including the output the clients dequantize, not only the input pair. The manuscript's own interior component (QDQ−INT8) presupposes this. The artifact summary also lists MUL, QUANTIZE and DEQUANTIZE in the INT8 operator inventory but not in the float one, and a DEQUANTIZE inside a model with integer input and output implies at least one float-domain tensor or operation. Size and graph are unchanged, so "free at deployment" stands. But neither the interior error component nor the attribution of the INT8 latency penalty to "per-call overhead, not arithmetic" can be interpreted without knowing which ops run in integer arithmetic.
Suggested fix: reword to "changes only quantization parameters (input, intermediate and output scales and zero points); graph and size are unchanged". Publish the per-op dtype map of one INT8 artifact and say where the QUANTIZE and DEQUANTIZE ops sit. State the CPU backend and thread count used in the device benchmarks.

### W8: The round-five primary-endpoint cohort is under-described, and the audit that admitted it cannot fail
**Severity**: Minor
**Evidence Anchor**: text: §7.11 (round three), "A cadence audit admitted all 451 RT-CGM subjects under"
**Confidence**: 4 — based on the authors' audit files and DiaData's documentation; per-device native cadence not verified
The first audit (paper/results/diadata_audit_r1.json) records RT-CGM as the JDRF CGM RCT with mixed devices (DexCom SEVEN, MiniMed Paradigm, FreeStyle Navigator), "per-subject sensor not recoverable", and excludes it on interval grounds. The round-three audit readmits all 451 subjects because all 15,284,371 within-segment intervals fall in [270, 330] s. A 5-minute-gridded archive split at gaps yields that by construction, and DiaData's records describe timestamps rounded to 5 minutes and 10- or 15-minute devices resampled to 5 minutes. The rule cannot separate native from regridded data, so the reason for the round-one exclusion is unresolved. The manuscript gives no device description for the cohort that certifies the shipped model, and does not explain why 96 of 451 subjects contribute no scored windows. The round-one audit's collinear_fraction for RT-CGM (0.305) also exceeds that of every included cohort (0.125 to 0.229); if that statistic is an interpolation signature, it deserves a sentence. The paired contrast stays internally valid, since both models see identical windows.
Suggested fix: describe RT-CGM's devices and era in §7.11 and §10.2, and report subjects and windows by device or state that device is unrecoverable. Explain the 96 subjects without windows. Replace the cadence rule with one that can fail on regridded data (for example, runs of repeated or exactly collinear values), or report it as uninformative.

### W9: "Clinical" wording contradicts the research-only framing
**Severity**: Minor
**Evidence Anchor**: text: §8.2, "a worse clinical-shape classifier"
**Confidence**: 5 — wording, checked against the manuscript's own framing
§1 calls the target a research representation that "neither reproduces a manufacturer's algorithm nor provides a dosing rule", and §3 ties the bands to "rate bands commonly represented by CGM trend arrows". §7.12 nevertheless calls the same bands "the unchanged clinical thresholds" and decodes a regression head "at the clinical thresholds", and §8.2 calls the sequential artifact "a worse clinical-shape classifier". These phrases suggest a clinical validation that the paper disclaims.
Suggested fix: use "trend-arrow rate bands" (or "the ±1 and ±2 mg/dL/min bands") and "directional-recall profile".

### W10: Prior on-device glucose prediction and the OhioT1DM source are missing from Table 1
**Severity**: Minor
**Evidence Anchor**: table: tab:relatedwork (Table 1, Device evidence column)
**Confidence**: 4 — the missing study's existence, venue and DOI verified; its full results not reviewed
Table 1 lists Li et al. (2020) as the only prior work with device evidence. Zhu et al., however, built a wearable IoMT device that runs a recurrent glucose predictor on a low-power system-on-chip for real-time prediction and hypoglycemia detection, which bears directly on an on-device positioning. Two rows give OhioT1DM as their evaluation scope without citing the dataset.
Suggested fix: add T. Zhu, L. Kuang, J. Daniels, P. Herrero, K. Li and P. Georgiou, "IoMT-Enabled Real-Time Blood Glucose Prediction With Deep Learning and Edge Computing," IEEE Internet of Things Journal, 2022, doi:10.1109/JIOT.2022.3143375, as a device-evidence row. Cite C. Marling and R. Bunescu, "The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020," CEUR Workshop Proceedings 2675:71–74, 2020.

### W11: Methods named in the sweeps are uncited, and two outcomes are known results
**Severity**: Minor
**Evidence Anchor**: text: §7.12, "A CORN ordinal head reached 0.470"
**Confidence**: 5 — standard references
Focal loss, ordinal label smoothing, the distance-weighted cross-entropy penalty, the TCN, cosine learning-rate decay, the CORN head and weight soups appear without attribution. Two reported outcomes follow from prior work and should be framed that way. A uniform soup of independently seeded runs collapses because such runs are not linearly mode-connected; model soups average fine-tuned models that share a pretrained initialization. The rate-regression head's regression to the stable-dominated mean is a documented failure mode of imbalanced regression.
Suggested fix: cite T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for Dense Object Detection," ICCV 2017, pp. 2980–2988; S. Bai, J. Z. Kolter and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018; I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017; X. Shi, W. Cao and S. Raschka, "Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities," arXiv:2111.08851, 2021; M. Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time," ICML 2022, PMLR 162:23965–23998; J. Frankle, G. K. Dziugaite, D. M. Roy and M. Carbin, "Linear Mode Connectivity and the Lottery Ticket Hypothesis," ICML 2020, PMLR 119:3259–3269; Y. Yang, K. Zha, Y.-C. Chen, H. Wang and D. Katabi, "Delving into Deep Imbalanced Regression," ICML 2021, PMLR 139:11842–11851. Give sources for the ordinal smoothing and the distance penalty if they come from prior work.

### W12: Fixing the input range from the sensor's reportable range is not considered
**Severity**: Minor
**Evidence Anchor**: text: §8.1, "calibration design reduces to one rule"
**Confidence**: 4 — API availability verified in the LiteRT Torch source; the benefit for this model is untested
The failure mechanism is input clipping, and the input domain is known before any data are seen: the sensor reports within a fixed range (here 39–401 mg/dL, including the codes in W6). The PT2E quantizer API used here supports fixed quantization parameters (FixedQParamsQuantizationSpec in torchao.quantization.pt2e.quantizer, which LiteRT Torch's pt2e_quantizer already uses for ops that require fixed parameters). The input scale and zero point could therefore be set from the sensor range, making the input independent of calibration data. Delta encoding (§7.8) is one route; this one is more direct. The "one rule" in §8.1, not letting contiguous window order pick calibration data, is weaker than a rule that removes the dependence.
Suggested fix: add a fixed-input-range arm (input parameters spanning the sensor range, interior calibrated with sequential-200), or discuss it as the domain-informed alternative.

### W13: The absent resolution penalty is presented as a refuted fear, though 8-bit error analysis predicts it
**Severity**: Minor
**Evidence Anchor**: text: §8.1, "The feared resolution trade-off"
**Confidence**: 4 — standard uniform-quantization error analysis
At 1.572 mg/dL per code, rounding error is at most 0.79 mg/dL with an RMS of about 0.45 mg/dL (step/√12). That is close to the 1 mg/dL resolution at which CGMs report and far below the 15 mg/dL change that separates stable from falling or rising over 15 minutes, whereas sequential-200 clipping removes up to about 150 mg/dL. The clipping–rounding analysis the manuscript cites (Banner et al. 2019) and the range-setting discussion in Nagel et al. (2021, full citation in W2) imply that at 8 bits a bounded input loses little to full-range coverage. The manuscript's explanation ("a label that depends on the difference between readings") is incomplete.
Suggested fix: present the null resolution cost as the expected result of a step that is small relative to class-defining changes and sensor error, with the error arithmetic and these citations.

### W14: The delta-encoding resolution comparison omits the level channel's effective resolution
**Severity**: Minor
**Evidence Anchor**: text: §7.8, "model's INT8 input grid is $6.4\times$ finer (scale 0.247 versus 1.572"
**Confidence**: 5 — arithmetic from the reported scales and the per-tensor activation scheme
Activations, including the input, are quantized per tensor here, so the anchor channel (last value divided by 16) shares the 0.247 step. Absolute level is thus represented at 0.247 × 16 ≈ 3.95 mg/dL per code, about 2.5 times coarser than the absolute-input model's 1.572, while the differences are 6.4 times finer. "6.4× finer" holds for the differences only.
Suggested fix: report both effective resolutions and the anchor channel's headroom (401/16 ≈ 25.1 against an input range about 63 units wide at 0.247 per code).
