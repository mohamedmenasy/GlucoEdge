contract_role: perspective
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: warn
trigger: "Localized gaps that limit transfer without undermining the central claims."

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

### Reviewer Identity
Peer Reviewer 3 (perspective seat). I am a mobile ML systems engineer-researcher. I work on the compiler and runtime stack for on-device inference: PyTorch 2 export (PT2E) quantization flows, TFLite/LiteRT conversion and operator lowering, and XNNPACK CPU kernels. I also publish on on-device benchmarking methodology and cross-platform numerical reproducibility. I read the paper's deployment half as a toolchain practitioner. I am not a statistician, a CGM clinician or a quantization-algorithm researcher, and I defer to the other seats on inference, endpoints and clinical meaning.

### Overall Recommendation
Minor Revision, judged from the perspective seat only. The central RQ1 accuracy finding survives everything I checked. The changes I ask for are re-scoping of the deployment claims and one cheap toolchain check, not new data.

### Confidence Score
4

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

The final panel topology cannot be known until every seat has completed, so this seat report is not calibrated.

### Criterion-Bound Judgements
No review target is bound (criteria_binding_unavailable), so the rows below use only the contract's D4 description and make no venue-alignment claim.

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D4: framing and definitions accessible to adjacent-field readers | contract D4 description | MEETS | S3 (§1), S5 (§5.3) | "Calibration" is given its three meanings; the quantizer scheme and the client arithmetic are stated precisely enough to reproduce | none identified | no; it is a strength |
| D4: interdisciplinary (systems and deployment) claims substantiated | contract D4 description | PARTLY_MEETS | W1 (§7.9), W2 (Table 19), W4 (§8.2), W5 (§C.7) | Measurements are honest and correctly reported, but the INT8-versus-float conclusions describe one converter lowering path, and the Android parity coverage of the shipped artifacts is overstated | My lowering probe uses a random-weight toy model, not the study checkpoints, and I timed nothing on a device | yes; this drives the D4 warn |
| D4: implications for adjacent practitioners scoped | contract D4 description | PARTLY_MEETS | W3 (§9.1), W6 (§5.3) | The proposed remedy and the INT8 motivation are not mapped onto the encoding and targets actually deployed | none identified | yes; it contributes to the warn |
| D1, D2, D3, D5, D6 | contract (other owners) | NOT_ASSESSED | — | Outside the perspective seat's eligible roles | — | no |

### Summary Assessment
From a mobile-runtime angle, the calibration study is unusually well instrumented. It includes clip-only and QDQ controls, per-artifact hashes, a paired and order-randomized device benchmark, and an honest per-operator disclosure that the "INT8" artifact is mostly float. Its central systems lesson carries over to anyone running a static-PTQ pipeline over overlapping time-series windows: contiguous calibration selection silently caps the input range, and gating on accuracy hides the damage.

The weaker part is how the paper interprets its own artifact. I inspected the committed `.tflite` files and re-ran the repository's `convert_int8` on a random-weight model shaped like TrendCNN. The float convolutions come from how LiteRT Torch 0.9.1 lowers `aten.conv1d`: a 2D convolution with float bias added separately, which breaks the quantized-convolution fusion. An arithmetically identical `Conv2d` with (1,3) kernels gives an integer graph of 9 operators instead of 22.

So "INT8 is slower than float" and "no measured speed benefit" describe this lowering path, not 8-bit inference of this model. The accuracy findings very likely transfer: the toy graphs agree in argmax on all 5,000 random inputs I tried.

Three further issues follow:
- The microbenchmark measures per-call overhead at the timer's resolution, at a cadence unlike the app's.
- The proposed fixed-range remedy does not map onto the delta+anchor encoding that every deployed model uses.
- The text overstates Android parity coverage for the bundled artifacts that came after July.

### Strengths

### S1: Mechanism controls that localize the failure to one quantize operation
The clip-only and QDQ arms are the ablation a runtime engineer would ask for first. They put 13.14 of the 13.35 points at input clipping and almost nothing at grid resolution or interior quantization. My inspection of the committed sequential-200 and uniform-200 artifacts is consistent with this: only the input, activation, output and FC-bias quantization parameters differ, and the int8 weights are identical.
**Evidence Anchor**: table: Table 7 (tab:mechanism), Section 7.4

### S2: Per-operator disclosure of what the converter actually produced
Few applied papers report which operators of a "quantized" model really execute in int8, or which tensors calibration changes inside the graph. I verified both against the committed artifacts: the convolutions are float32 fed by DEQUANTIZE, only FULLY_CONNECTED is int8, and 12 graph tensors differ between calibration arms.
**Evidence Anchor**: text: §7.9 "the sequential-200 and uniform-200 artifacts of one checkpoint, 12 of 41"

### S3: Terminology disambiguated for three adjacent communities
The introduction separates quantization calibration from CGM sensor calibration and from probability calibration. That collision would otherwise mislead diabetes-technology readers and ML-calibration readers alike.
**Evidence Anchor**: text: §1 "Unqualified, calibration here means this quantization step, neither CGM"

### S4: Benchmark hygiene above the mobile-ML norm
The benchmark uses 30 paired rounds with reshuffled artifact order, warm-up, hash and parity verification before timing, recorded thermal state and a stated p95 estimator. The raw rounds are released. The released JSON matches the table.
**Evidence Anchor**: text: §C.6 "in 30 paired rounds with the artifact order reshuffled each"

### S5: Client-side quantization arithmetic specified for reproduction
Quantization parameters are read from model metadata, never hardcoded, and rounding is half-to-even in NumPy, Kotlin and Swift. I confirmed in both clients that the parameters come from the model at load time. Practitioners porting a model across platforms rarely get this level of specification.
**Evidence Anchor**: text: §5.3 "with quantization parameters read from the model"

### S6: A transferable lesson for conversion pipelines in practice
The "accuracy illusion" applies directly to model-conversion CI in production MLOps. A per-class or minority-recall gate against the float model is cheap, and the paper shows concretely that it catches what an accuracy gate approves.
**Evidence Anchor**: text: §9.2 "pipeline that gates a quantized model on raw accuracy would have promoted"

### Weaknesses

### W1: The hybrid float/int8 graph is a conv1d lowering artifact, and the INT8 deployment conclusions inherit it
**Severity**: Major
**Evidence Anchor**: text: §7.9 "The artifact is also less integer than its name"
**Confidence**: 4 — direct inspection of the committed artifacts plus a toolchain probe on a random-weight model; not repeated on the study checkpoints or timed on a device

The paper reports correctly that only the final FC runs in int8 (§5.1, §7.9), but it never says why. The quantizer is not the cause: LiteRT Torch's `PT2EQuantizer` annotates `aten.conv1d` (`_annotate_conv` and `_annotate_conv_relu` list it). The committed artifacts show what happens at lowering. Each Conv1d becomes RESHAPE, DEQUANTIZE (activations and weights), a float DEPTHWISE_CONV_2D or CONV_2D, RESHAPE, a separate float ADD for the bias, and QUANTIZE. Because the bias is split out, the dequantize-convolution-quantize pattern never fuses into an integer convolution.

I imported `conversion/convert.py:convert_int8` unchanged and ran it with LiteRT Torch 0.9.1 and TorchAO 0.17.0 on two random-weight models:
- A model shaped like TrendCNN, written with `Conv1d`, reproduced the study artifact's pattern (22 operators, both convolutions in float32).
- The same weights written as `Conv2d` with (1,3) kernels on a (B,1,1,12) view gave 9 operators, with DEPTHWISE_CONV_2D and CONV_2D in int8 and int32 bias.

Both models received identical input and output quantization parameters. Over 5,000 random inputs in the 40–400 range, their int8 outputs differed by at most one code (2.7% of inputs differed in any code), and the argmax agreed on all 5,000.

Why this matters across fields:
1. The conclusions "INT8 is slower than float in every arm", "no measured speed benefit" and "an accuracy risk against a size saving" (§7.9, Limitation 8, Conclusion) are properties of this lowering, with 23 operators against the float graph's 14. They are not properties of 8-bit inference of this model. The latency gap is also not attributed by operator-level profiling (see W2).
2. Integer-only targets cannot run the converted graph as a whole: NPUs, DSPs and microcontroller runtimes, the class of the wearable system-on-chip work the paper cites. Yet those are the targets where INT8 is the enabling format.
3. Reassuringly, the RQ1 accuracy findings very likely carry over. The hybrid graph already quantizes every activation that a full-integer graph would quantize (input, convolution outputs, pooled features, output). The concern that the stratification and observer nulls might fail in a full-integer graph is therefore mostly unfounded for this architecture. The paper can say so with evidence.

Suggested remedy:
- Minimum: state the cause and the converter version, and scope every INT8 latency and "integer" statement to "LiteRT Torch 0.9.1 conv1d lowering". That covers the abstract's "network in 8-bit integers (INT8)", contribution bullet 4, §7.9, Limitation 8 and the Conclusion.
- Stronger and cheap: copy each study checkpoint's weights into the Conv2d(1×k) form, reconvert with the same 20 selections, report agreement with the published INT8 predictions, and time the full-integer artifact in the existing paired benchmark.

### W2: The microbenchmark measures per-call overhead at timer resolution, not deployment latency
**Severity**: Minor
**Evidence Anchor**: table: Table 19 (tab:devicebench), Appendix C.6
**Confidence**: 4 — recomputed from `device_benchmark_ios_r1.json` and read `CalibrationBenchmarkTests.swift` and `TrendClassifier.swift`

`DispatchTime` on this hardware ticks every 41.67 ns, and the released medians are exact tick multiples: 2.958 µs is 71 ticks, 3.333 µs is 80 and 3.375 µs is 81. The "within 0.05 µs" spread between calibration arms is therefore about one tick.

This does not threaten the calibration-arm conclusion. The arms have byte-identical sizes and the same operator inventory, a structural argument that needs no timing. It does limit what the INT8-versus-float figure means:
- The inner timer includes client-side Swift array work: quantizing into a new `[Int8]`, writing and reading the buffers, and dequantizing. The +0.64 to +0.70 µs therefore mixes application overhead with the nine extra graph operators.
- The per-round float means range from 2.42 to 3.07 µs, so core placement or frequency state was not controlled. The paired design mitigates this but does not explain it.
- At the application's cadence, one inference every five minutes, the relevant numbers are cold single-call latency and energy per wake-up, not back-to-back means. The in-app observations of about 0.12 ms, roughly forty times the microbenchmark, are not reconciled with Table 19 anywhere in the text.

Suggested remedy:
- Minimum: state the timer resolution, say that a ~32k-MAC model is overhead-dominated at this scale, and label the 3 µs figure as hot-loop dispatch cost.
- Stronger: attribute time per operator with LiteRT's benchmark tool and op profiling, and report one cold invocation at app cadence. Alternatively, argue explicitly that latency is immaterial at this cadence and drop the INT8-versus-float latency comparison from the contribution list.

### W3: The proposed fixed-range remedy does not transfer to the delta+anchor encoding actually deployed
**Severity**: Minor
**Evidence Anchor**: text: §9.1 "fixing its range from the sensor's reporting limits"
**Confidence**: 4 — quantization parameters read from the bundled artifact and from `panel_revision_r1.json`; arithmetic is mine

The remedy in §9.1 is written for absolute-mg/dL input. Every sweep model, and the bundled pair, instead feeds 11 differences and a level/16 channel through one per-tensor input scale. For this mixed-unit tensor, the histogram observer's range is driven by rare large differences, and it varied widely across the three uniform-200 draws:
- draw 101 (the bundled artifact): [-27.06, +31.91];
- draw 102: [-129.8, +52.1];
- draw 103: [-220.4, +48.5].

The resolution for differences thus changes 4.6-fold with the draw (0.231 to 1.054 mg/dL per code). A range fixed from the sensor's 39–401 limits would have to span differences of ±362 mg/dL, about 2.84 mg/dL per code. That is 12 times coarser than the shipped 0.231, and more than half of the 5 mg/dL per five-minute step that separates stable from falling.

In the graph as converted, the int8 input interface buys nothing: the first operator after RESHAPE is DEQUANTIZE into a float convolution, and the clipping happens in the clients' `quantizeInt8` before the model runs.

Suggested remedy:
- Minimum: state that the remedy targets absolute inputs, and note that the deployed encoding still clips 2.8% of RT-CGM windows (§8.5).
- Better: discuss the two practical options. (a) For the current hybrid graph, a float input interface removes the failure mode at no compute cost. (b) For a full-integer graph, two graph inputs, differences and level, would each get their own quantization parameters. I could not get LiteRT Torch 0.9.1 to leave the first convolution unannotated (`set_module_name(..., None)` raises), so option (a) may need a converter-side change. Mark both as untested.

### W4: Android parity coverage of the post-July bundled artifacts is overstated
**Severity**: Minor
**Evidence Anchor**: text: §8.2 "after golden parity passed in the Android unit tests on the Java virtual"
**Confidence**: 4 — repository inspection (test sources, CI workflow, README, model manifest); unrecorded local runs cannot be excluded

The Android golden-parity test exists only as an instrumented test (`androidTest/.../GoldenParityTest.kt`). No JVM unit test loads a model or the golden vectors. CI runs `testDebugUnitTest` but no instrumented test, and the README calls Android parity a "local-only gate" that needs an emulator or device.

The only recorded Android CompiledModel parity run is the July Galaxy S22 Ultra run on the originally bundled artifacts, and that device is retired. An emulator run would exercise the classic Interpreter with XNNPACK disabled, not CompiledModel. Two statements in the manuscript therefore overstate what was verified:
- "golden parity passed in the Android unit tests on the Java virtual machine" (§8.2);
- "Each bundled pair passed both golden-parity suites bit-exactly on the continuous-integration simulators" (§A.1).

For the shipped pair, Python-versus-Swift parity is evidenced; Kotlin with LiteRT 2.1.0 CompiledModel is not.

Suggested remedy: reword both sentences to say what ran, namely the Android JVM unit tests (no model execution) and the iOS simulator and device parity. Then either run the instrumented suite on any arm64 device for the current pair or scope the "three clients agree" statement (§9.2) to the historical artifacts.

### W5: Golden-vector coverage is thin for a "bit-exact" claim on the shipped hybrid graph across three runtime versions
**Severity**: Minor
**Evidence Anchor**: text: §C.7 "selected from each float-predicted class and including values above the"
**Confidence**: 4 — golden vectors re-encoded and run through the committed artifacts with the Python LiteRT 2.1.5 interpreter; the synthetic windows are not CGM data

Three runtime versions are in play: Python LiteRT 2.1.5 produced the goldens, Android uses 2.1.0 and iOS uses 2.3.0. In the hybrid graph, float convolution outputs feed QUANTIZE operations, so bit-exact int8 agreement needs float kernels on different ISAs and versions never to straddle a rounding boundary. Twenty windows chosen only by float class do not test that.

The claim that the goldens include values above the INT8 input range is also stale. The exporter's selection rule (window maximum above 240 mg/dL) was written for the absolute model's 239.9 ceiling. After delta+anchor encoding, the current goldens span [-14.0, +16.3] inside the bundled range [-27.06, +31.91], so end-to-end parity no longer exercises the saturation path that RQ1 is about. Unit tests do cover saturation arithmetic.

On 20,000 random synthetic windows (277 trace windows in parentheses), the bundled pair behaves as follows:
- The top int8 output code is tied in 2.5% (1.1%) of windows. All three clients break the tie towards the first index, consistently but without documentation.
- The output range tops out at +1.99 logits (scale 0.1107, zero point 109). The top logit saturates in 8.4% (6.1%).
- The displayed INT8 top-class probability differs from float by up to 8 percentage points on the trace.

Suggested remedy: add boundary-stress vectors (inputs whose pre-quantize values fall near rounding boundaries, plus saturating differences), report the minimum distance to a boundary across the golden set, and state the tie rule.

### W6: The paper does not say which deployment target INT8 is for, and the measured phones favour float
**Severity**: Minor
**Evidence Anchor**: text: §5.3 "every complete window locally with the float artifact by default or INT8"
**Confidence**: 4 — app defaults confirmed in both `MainViewModel` sources; size figures are the paper's own

Both apps run float by default. On the measured phone, INT8 is slower, and it saves 5,376 or 146,624 bytes of model file. That saving is small next to the inference runtime the apps already bundle, and file size need not mean lower runtime memory when convolution weights pass through DEQUANTIZE. Peak memory was not measured.

The targets where INT8 is the enabling format are microcontrollers and integer-only accelerators, the class of the cited wearable system-on-chip, and the hybrid graph cannot run there (W1). As written, RQ1's audience is implicit, and a practitioner could conclude that INT8 is pointless for this task.

Suggested remedy: one paragraph naming the intended deployment target of the INT8 path, saying that on phones float is the sensible default, and positioning the calibration-coverage lesson for full-integer pipelines, where it applies with full force.

### Detailed Comments

#### Assumption Audit
- **Explicit assumptions**: Static INT8 PTQ through PT2E to LiteRT is the deployment path, and its cost is file size, preparation time and back-to-back latency. These are stated and measured honestly.
- **Implicit assumptions**: (1) The converter's "INT8" output stands for 8-bit inference of this model (W1). (2) A hot-loop microbenchmark represents latency for a classifier invoked every five minutes (W2). (3) INT8 is the artifact a phone deployment would ship, although both apps default to float (W6). (4) Parity on 20 synthetic windows generalizes across three runtime versions (W5).
- **Paradigmatic assumptions**: ML evaluation treats the model as the unit of analysis. Systems work treats the tuple (model, converter version, lowering, runtime, delegate, threads, device) as the unit. The paper half-adopts the systems view (Table 1's device-evidence column, Limitation 8) but phrases its deployment conclusions at the model level.

#### Cross-Disciplinary Connections
- **Parallel research**: The benchmark-reporting conventions of MLPerf Tiny and AI Benchmark, both already cited, ask for runtime, delegate, threads and warm/cold state. Operator-level latency modelling (nn-Meter) explains why the extra QUANTIZE, DEQUANTIZE and TRANSPOSE operators, not arithmetic, dominate a model this small.
- **Borrowing opportunities**: A CI assertion on operator dtypes, in the same spirit as the repository's native-library provenance guard, would have flagged the float convolutions automatically. Boundary-stress golden vectors come from numerical-reproducibility practice.
- **Methodological borrowing**: Per-operator profiling with LiteRT's benchmark tooling. One cold call at app cadence, and energy per inference via the platform power tools. Differential testing of converted against source models across runtime versions.

#### Practical Impact
- **Real-world application**: The calibration-coverage pitfall and the accuracy-gate lesson transfer to any static-PTQ pipeline over sliding windows (wearables, ECG, IMU). This is the paper's most portable systems contribution.
- **Implementation feasibility**: A full-integer graph appears to need only a model-definition rewrite (W1). The fixed-range remedy needs a custom annotation and, for the deployed encoding, an input split (W3).
- **Stakeholders**: Embedded and wearable engineers, who most need INT8 and are least served by the hybrid artifact. App developers combining LiteRT and LiteRT-LM.

#### Broader Implications
- **Ethical dimensions**: The privacy guard on iOS is a source-regex scan, and the paper calls it weaker. It does not cover linked binaries, and it misses, for example, `Data(contentsOf:)` on a remote URL. A check on the built app's linked frameworks and imported symbols would be stronger at similar cost.
- **Social impact**: Nothing beyond the research-use scoping the paper already states. The INT8 mode's displayed probabilities can deviate from float (W5), which is worth a sentence because the apps show confidence bars.
- **Future directions**: Full-integer conversion and a microcontroller run of the 2,909-parameter model. Energy per inference at CGM cadence. A two-input-tensor graph for the delta+anchor encoding. Cross-runtime differential testing across LiteRT versions.

### Cross-Disciplinary Reading Recommendations
- Zhang et al., "nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices", MobiSys 2021. It explains latency through kernel and fusion behaviour rather than parameter count, which is the lens W1 and W2 need.
- Pham et al., "Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance", ASE 2020. It covers nondeterminism-driven variance in training, relevant to the paper's non-bitwise retraining across CPU and MPS.
- The LiteRT post-training quantization guide the paper already cites (its integer-only quantization section). It distinguishes full-integer from mixed graphs and is the natural reference for W1.
- [UNVERIFIED] Search lead: empirical studies of faults introduced by deep-learning model conversion between frameworks, and differential testing of converted against source models (search "deep learning model conversion faults differential testing").
- [UNVERIFIED] Search lead: measurement methodology for the energy cost of on-device inference on smartphones.

### Questions for Authors
1. Did you know that the float convolutions come from the conv1d lowering? Does a Conv2d(1×k) re-expression of the study checkpoints reproduce the published INT8 predictions, and how does it time?
2. Which deployment target motivates the INT8 artifact, given that both apps default to float on phones?
3. What thread count and XNNPACK configuration does CompiledModel's CPU accelerator use by default here? What is the latency of one cold call at the app's cadence?
4. Has any bundled pair since July passed the Android instrumented golden-parity suite through CompiledModel on a physical arm64 device?
5. Would LiteRT Torch support two separately quantized graph inputs (differences, level) for the delta+anchor model?

### Minor Issues
- §5.1 and §7.9: please give the INT8 output quantization parameters, since the output range bounds the displayed probabilities.
- §5.3: the Android build resolves a `libLiteRt.so` collision between `litert` and `litertlm-android` with `pickFirsts`. The classifier's copy is guarded, but LiteRT-LM then runs against a library build it was not shipped with. One sentence would help readers reproduce the Explain path.
- §C.6: alongside "within 0.05 µs", state the timer tick (41.67 ns).
