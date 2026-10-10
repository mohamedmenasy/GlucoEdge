# Bundled model provenance

Both models are the DELTA-ENCODED (anchor variant) artifacts of the
2026-10-10 delta-inputs experiment (run `delta-r1`, protocol frozen before
any metric was read; spec
docs/superpowers/specs/2026-10-10-second-cohort-and-delta-inputs-design.md):
training seed 0 on the study's eligible weinstock windows, inputs = 11
consecutive differences + last reading / 16 (both clients apply this
transform in `DeltaEncoding` before quantization; golden vectors pin
Python/Kotlin/Swift to the same bytes). INT8 is calibrated on the study's
`uniform_random_200_d101` selection. Reproduce with:
- `venv/bin/python -m experiments.delta_inputs train --run-id delta-r1 --variant anchor`
- `venv/bin/python -m experiments.delta_inputs convert --run-id delta-r1 --variant anchor`

Held-out-participant directional recall
(paper/results/calibration-coverage/delta_inputs_r1.json): float 0.5088,
INT8 (uniform-200) 0.5092 — above the absolute-input baseline (0.487) —
and the INT8 input grid is 6.4x finer (scale 0.247 vs 1.572 mg/dL/code),
which makes quantization nearly calibration-proof: even the
toolchain-default sequential selection costs only ~0.5 pp here (the frozen
ship gate passed at 0.4955 vs the 0.5 criterion — a thin margin, stated
plainly; the shipped INT8 uses the uniform selection, where no gap is
resolvable).

| file | sha256 | bytes |
|---|---|---|
| `trend_float.tflite` | `c30560f7283e4927a4b63ff0ffdd34f4108e035905b80c4ed4fd01b479dbb07a` | 17352 |
| `trend_int8.tflite` | `8590a3d25c409a46659cba3dab97f649c54b57061022e01cae0feec2f5366388` | 11976 |

Verification state: simulator suite 51/51 green against these bytes
(golden parity incl. the Swift delta transform, INT8 bit-exact); the
on-device parity re-run is PENDING — the iPhone was detached before it
could run. The previous bundled pair (absolute-input study seed-0,
shipped earlier on 2026-10-10) WAS device-verified on the iPhone that
day: float `5a0f0744…`, INT8 `9f2f26f5…` — now historical; see git
history and the paper.

Golden parity with these exact bytes is green on the iOS simulator suite
(48/48, INT8 bit-exact) and the goldens pin both sha256s. On-device
verification (Galaxy S22 Ultra / iPhone 18 Pro Max) was performed on the
PREVIOUS assets (2026-07-07 / 2026-10-10). The Android test device is no
longer available (broken as of 2026-10-10), so the Android on-device record
stays historical. These assets WERE re-verified on the iPhone 18 Pro Max on
2026-10-10: golden parity 5/5 on device (INT8 bit-exact), and both files
were among the nine artifacts hash-verified, parity-checked on 277 synthetic
windows, and latency-benchmarked on device (see
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
