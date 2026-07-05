# Bundled model provenance

Both models were produced from this repository at commit 361148a, seed 0:
- `venv/bin/python -m training.train --dataset weinstock --classes 5 --epochs 20 --seed 0 --save-checkpoint`
- `venv/bin/python -m conversion.convert --dataset weinstock --classes 5`

Verified against docs/superpowers/specs/2026-07-01-litert-conversion-results.md
(float acc 0.5184 / macro recall 0.5060; int8 acc 0.5866 / macro recall 0.4135).

| file | sha256 | bytes |
|---|---|---|
| `trend_float.tflite` | `eb96c7e66ab13a41a6bacdcbeb00c0c089d63f67dd1a45c2947a48ff301b08e6` | 17352 |
| `trend_int8.tflite` | `0dc3538704c89f055fb7437e50a5bf42c81a64853a4b527d49c206569f8bea29` | 11976 |

Not a medical device. Trained only on the public GlucoBench benchmark.

## Runtime backend note (Task 8)

The intended hot inference path is LiteRT's `CompiledModel` API. On the dev/test AVD
(`Medium_Phone_API_35`, Apple Silicon host), `CompiledModel.create()` unconditionally
SIGILL-crashes the process for both `trend_float.tflite` and `trend_int8.tflite`, on both
litert 2.1.0 and 2.1.6 - a native CPU-feature probe (`rdsvl`, ARM SVE) inside `libLiteRt.so`
runs before any accelerator option takes effect, and this guest's virtual CPU advertises
SVE2/SME2 in `/proc/cpuinfo` without actually supporting execution of those instructions. A
SIGILL is not a catchable JVM exception, so there is no runtime try/fallback available - the
backend has to be chosen statically. `TrendClassifier` therefore runs both models through the
classic `org.tensorflow.lite.Interpreter` (XNNPACK explicitly disabled, which avoids the
crashing probe) instead of `CompiledModel`, verified not to crash for either model. Full
evidence (disassembly, tombstones) is in `.superpowers/sdd/task-8-report.md`. This is an
environment-forced substitution, not a defect in the int8 `TensorBuffer` buffer API - `writeInt8`/
`readInt8` are real methods (confirmed via `javap`) and the `CompiledModel`-based implementation
compiled cleanly on the first try.
