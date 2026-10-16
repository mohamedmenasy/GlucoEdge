# Bundled model provenance

Both models are the SWEEP-R5 winner of the 2026-10-14 accuracy sweep
round five (run `sweep-r5`, protocol frozen and committed before the
confirmatory read; spec
docs/superpowers/specs/2026-10-14-accuracy-sweep-r5-design.md): the
round-two DiaData-only training mixture (five audited cohorts, 898
subjects, zero Weinstock training windows), (96,144) channels with a
position-preserving flatten head (50,645 params — the global average
pool is gone, the final features of all 12 positions feed the
classifier), 1-hour window, cosine learning-rate decay over 20 epochs
with the kept epoch picked by validation directional recall under the
stable floor (seed 4, kept epoch 11 of 0--19), delta-anchor inputs (11
consecutive differences + last reading / 16 — the same `DeltaEncoding`
transform both clients already apply; golden vectors pin
Python/Kotlin/Swift to the same bytes). INT8 is calibrated on the
study's `uniform_random_200_d101` selection. Reproduce with:
- `venv/bin/python -m experiments.accuracy_sweep_r5 train --run-id sweep-r5 --stage-name R` (then H)
- `venv/bin/python -m experiments.accuracy_sweep_r5 select --run-id sweep-r5 --stage-name R` (then H, S)
- `venv/bin/python -m experiments.accuracy_sweep_r5 convert --run-id sweep-r5`

Pre-registered confirmatory read
(paper/results/calibration-coverage/accuracy_sweep_r5.json): the primary
endpoint moved to a cohort with no prior reads — zero-shot directional
recall on DiaData's RT-CGM cohort (12.67M eligible windows, 355
participants; neither model under comparison ever trained on it). INT8
directional 0.5548 vs the previous champion INT8 0.5528 (+0.20 pp,
participant-cluster bootstrap CI [+0.11, +0.29 pp], fully positive);
stable recall 0.605 vs 0.607 (−0.16 pp, inside the −2 pp gate); INT8
within 0.25 pp of its own float. All three frozen gates passed. Known
cost, stated plainly: macro-F1 trails (0.410 vs 0.427 on RT-CGM, 0.342
vs 0.364 held-out), the reused held-out descriptive contrast is
−0.3 pp [−0.8, +0.1], and never-trained PEDAP is a near-tie (0.539 vs
0.541). On the non-gating secondaries the stable-class cost exceeds the
level that gates the primary: held-out stable recall 0.436 vs 0.459
(−2.2 pp) and temporal 0.499 vs 0.520 (−2.0 pp), and on PEDAP the
`falling` class drops 6.4 pp (0.396 vs 0.460) and `rising` 3.1 pp —
the certified gain is small and lives on the one cohort where no
selection pressure was ever applied.

| file | sha256 | bytes |
|---|---|---|
| `trend_float.tflite` | `bfb37468c5896be9f8c30f515c7499586a71da1c0ca0f77319080f8fcaa76153` | 208560 |
| `trend_int8.tflite` | `eacc0a741403b901330f0dfcc529195779b29f7bb0ed2de9ae90da8912dcf554` | 61936 |

Verification state: verified at ship time (2026-10-15) on the Android
JVM suite (39/39) and the iOS simulator suite (51 tests in 13 suites,
golden parity included — both bundled files hash-match the goldens and
the INT8 outputs are bit-exact against the Python record). Device-verified
the same day on the iPhone 18 Pro Max (iOS 27.0.1): the full suite — 51
tests in 13 suites — passed on device, including golden parity
(`bundledModelsMatchGoldenHashes`, float logits match Python, INT8
bit-exact), plus a live in-app session the same day (CompiledModel,
n=100 per model, synthetic trace): float 0.124 ms mean / 0.182 ms p95,
INT8 0.126 ms / 0.179 ms, skipped rows 0.

The SWEEP-R2 pair these bytes replace — float `96eef43c…` (177,336
bytes), INT8 `a887ea4e…` (55,216 bytes); held-out directional recall
float 0.5244 / INT8 0.5234 vs its own predecessor 0.5174, all r2 gates
passed — is now historical. It WAS device-verified on the iPhone 18 Pro
Max on 2026-10-12, the day its bytes shipped: 51 tests in 13 suites,
golden parity (INT8 bit-exact) and the Swift delta transform included,
plus a live in-app session (CompiledModel, n=100 per model, synthetic
trace) at float 0.102 ms mean / 0.148 ms p95 and INT8 0.149 ms /
0.212 ms. Full r2 record:
paper/results/calibration-coverage/accuracy_sweep_r2.json.

The pair before it — the 2026-10-10 delta-inputs `delta-r1`
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

## Accuracy sweep sweep-r3 (2026-10-12): not shipped

Sweep r3 (RT-CGM reopen, time-of-day channel, TCN arm) did not clear the ship gates:
the TCN candidate beat the then-bundled r2 pair on held-out directional recall (+0.77pp, CI fully
positive) but broke the stable-recall floor (-4.4pp against a -2pp gate), so that
pair stayed. Full record: `paper/results/calibration-coverage/accuracy_sweep_r3.json`.

## Accuracy sweep sweep-r4 (2026-10-13): identity outcome, no read spent

Sweep r4 (objective heads, robustness levers, weight averaging) saw every arm lose to the
shipped-recipe control on validation; the surviving candidate converted to a float/INT8 pair
sha256-identical to the then-bundled r2 champion, so the pre-registered contrast would have
compared the model to itself — no confirmatory read was performed and the protected labels
were left unspent. Full record: `paper/results/calibration-coverage/accuracy_sweep_r4.json`.
Round five (sweep-r5, above) changed the head and the recipe, won on a fresh zero-shot
cohort, and its winner is the current bundled pair.
