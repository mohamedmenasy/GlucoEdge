# Bundled model provenance

Both models are the SWEEP-R2 winner of the 2026-10-12 accuracy sweep
round two (run `sweep-r2`, protocol frozen and committed before any
held-out label was read; spec
docs/superpowers/specs/2026-10-12-accuracy-sweep-r2-design.md): the
DiaData-only training mixture (five audited cohorts, 898 subjects, zero
Weinstock training windows), (96,144) channels (42,725 params), 1-hour
window, training seed 1, delta-anchor inputs (11 consecutive differences
+ last reading / 16 — the same `DeltaEncoding` transform both clients
already apply; golden vectors pin Python/Kotlin/Swift to the same
bytes). INT8 is calibrated on the study's `uniform_random_200_d101`
selection. Reproduce with:
- `venv/bin/python -m experiments.accuracy_sweep_r2 train --run-id sweep-r2 --stage-name M` (then S for seeds 3–4)
- `venv/bin/python -m experiments.accuracy_sweep_r2 convert --run-id sweep-r2`

Pre-registered confirmatory read
(paper/results/calibration-coverage/accuracy_sweep_r2.json): held-out
directional recall float 0.5244 / INT8 0.5234 vs the previous champion
INT8 0.5174 (+0.60 pp, participant-cluster bootstrap CI
[−0.06, +1.33 pp]); stable recall 0.4585 vs 0.4754 (−1.68 pp, inside the
−2 pp gate); temporal split agrees in sign (+0.50 pp). All three frozen
gates passed. Known cost, stated plainly: macro-F1 slips 0.364 vs 0.368
held-out, and stable-class recall is ~1.7 pp lower than the previous
pair. Zero-shot on the never-trained PEDAP cohort (7.0M windows,
descriptive): directional 0.541 vs 0.522.

| file | sha256 | bytes |
|---|---|---|
| `trend_float.tflite` | `96eef43c37779534dab86312a6c7c65954fade0f5b2af0cfc3084dffb03b9a01` | 177336 |
| `trend_int8.tflite` | `a887ea4e786e2b40ff6c235fef0cd70af1c4149421ac49358640da3715757c42` | 55216 |

Verification state: DEVICE-VERIFIED on the iPhone 18 Pro Max on
2026-10-12, the day these bytes shipped: the full test run — 51 tests in
13 suites, golden parity (INT8 bit-exact against these sha256s) and the
Swift delta transform included — passed on device. Also verified on the
Android JVM suite and the iOS simulator suite at ship time (CI 3/3 on
the ship PR).

The previous bundled pair — the 2026-10-10 delta-inputs `delta-r1`
anchor seed-0 artifacts, float `c30560f7…` (17,352 bytes), INT8
`8590a3d2…` (11,976 bytes), held-out directional recall float 0.5161 /
INT8 0.5174 — is now historical. It WAS device-verified on the iPhone
18 Pro Max on 2026-10-11 (golden parity + delta-transform suites 7/7 on
device, INT8 bit-exact; live in-app session: float 0.080 ms mean /
0.140 ms p95, INT8 0.086 ms / 0.160 ms). The pair before it
(absolute-input study seed-0, float `5a0f0744…`, INT8 `9f2f26f5…`) was
device-verified 2026-10-10; see git history and the paper.

Device-verification history (applies to PREVIOUS pairs, NOT the delta
bytes above): the original pre-study assets were device-verified on the
Galaxy S22 Ultra (2026-07-07; that device is broken as of 2026-10-10, so
the Android on-device record stays historical) and on the iPhone 18 Pro
Max (2026-10-10). The absolute-input study pair (`5a0f0744…`/`9f2f26f5…`)
was re-verified on the iPhone the same day — golden parity 5/5 on device
(INT8 bit-exact) — and both of those files were among the nine artifacts
hash-verified, parity-checked on 277 synthetic windows, and
latency-benchmarked on device (see
paper/results/calibration-coverage/device_benchmark_ios_r1.json).

Historical pre-study assets (float `eb96c7e6…`, INT8 `0dc35387…`, sequential
calibration): float acc 0.5184 / macro recall 0.5060, INT8 acc 0.5866 /
macro recall 0.4135 on the historical test split — see
docs/superpowers/specs/2026-07-01-litert-conversion-results.md.

Not a medical device. Trained only on the public GlucoBench benchmark.

## Runtime backend note (Task 8, updated by the CompiledModel-restore fix)

`TrendClassifier` is per-device-gated: **`CompiledModel` is the hot path on real hardware**,
per the project brief. Emulators get a separate `Interpreter` fallback.

Why the gate exists: on the dev/test AVD (`Medium_Phone_API_35`, Apple Silicon host),
`CompiledModel.create()` unconditionally SIGILL-crashes the process for both
`trend_float.tflite` and `trend_int8.tflite`, on both litert 2.1.0 and 2.1.6 - a native
CPU-feature probe (`rdsvl`, Arm SME) inside `libLiteRt.so` runs before any accelerator option
takes effect, and this guest's virtual CPU advertises SVE2/SME2 in `/proc/cpuinfo` without
actually supporting execution of those instructions. A SIGILL is not a catchable JVM
exception, so there is no runtime try/fallback available - the engine is chosen statically at
load time from build-time device signals (`Build.HARDWARE == "ranchu"`/`"cutf"`,
`Build.FINGERPRINT` containing `"generic"`, or `Build.MODEL` containing `"sdk_gphone"`), never
via try/catch around `CompiledModel.create()`. Full evidence (disassembly, tombstones) is in
`docs/superpowers/specs/2026-07-05-emulator-compiledmodel-sigill.md`.

On a device matching one of those emulator signals, `TrendClassifier` runs both models through
the classic `org.tensorflow.lite.Interpreter` (XNNPACK explicitly disabled, which avoids the
crashing probe), verified not to crash for either model via the full instrumented golden-vector
parity suite. This is not a defect in the int8 `TensorBuffer` buffer API - `writeInt8`/
`readInt8` are real methods (confirmed via `javap`) and the `CompiledModel`-based
implementation compiles cleanly.

**The `CompiledModel` branch is verified on real hardware** (2026-07-07): the golden-vector
parity suite passed 3/3 on a Samsung Galaxy S22 Ultra (SM-S908E, Android 16) with the device
gate routing to `CompiledModel` - no SIGILL, float logits within 1e-5 of the Python benchmark,
INT8 dequantized outputs bit-exact across all 20 vectors.

## Accuracy sweep sweep-r1 (2026-10-11): not shipped

A staged sweep (distance-weighted CE loss, 4x channels) beat the then-bundled INT8 pair by
+5.7 macro-F1 points on the pre-registered held-out read, but a post-hoc descriptive read
showed it trades directional (minority-class) recall for that gain (0.438 vs 0.517 held-out),
so that pair stayed. Full record: `paper/results/calibration-coverage/accuracy_sweep_r1.json`.
Round two (sweep-r2, above) fixed the endpoint to directional recall and its winner shipped
as the current bundled pair.
