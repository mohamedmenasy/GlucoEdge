# Bundled model provenance

Both models are artifacts of the 2026-10-10 calibration-coverage study
(run `r1`; protocol frozen at commit d63cc8f, study code merged via PR #11):
training seed 0 on the corrected eligible-window splits (no-gap-fill mask),
INT8 calibrated on the study's `uniform_random_200_d101` selection — 200
uniformly random eligible validation windows, draw seed 101. Reproduce with:
- `venv/bin/python -m experiments.calibration_coverage train --run-id r1`
- `venv/bin/python -m experiments.calibration_coverage select --run-id r1`
- `venv/bin/python -m experiments.calibration_coverage convert --run-id r1`

These are `seed0_float.tflite` and `seed0_int8_uniform_random_200_d101.tflite`
from that run, copied byte-identically. Held-out-participant metrics
(paper/results/calibration-coverage/aggregate_r1.json): float macro recall
0.4884 / directional recall 0.5087; INT8 0.4857 / 0.5069 — the INT8 input
range now reaches 400.8 mg/dL (the previous sequential calibration capped it
at 239.9 and cost 13.4 points of held-out directional recall; see the paper).

| file | sha256 | bytes |
|---|---|---|
| `trend_float.tflite` | `5a0f0744618b9126634af9d817e7dc171b86392ff22e2a1cd97b19b4cecfd24a` | 17352 |
| `trend_int8.tflite` | `9f2f26f5da5c2964abe9eaca71626c812d4710de4d698b7d83deee04fc5187a4` | 11976 |

Golden parity with these exact bytes is green on the iOS simulator suite
(48/48, INT8 bit-exact) and the goldens pin both sha256s. On-device
verification (Galaxy S22 Ultra / iPhone 18 Pro Max) was performed on the
PREVIOUS assets (2026-07-07 / 2026-10-10). The Android test device is no
longer available (broken as of 2026-10-10), so the Android on-device record
stays historical; re-verification of these assets is planned on the iPhone
only. The runtime path, tensor shapes, operator set, and file sizes are
unchanged from the verified assets.

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
