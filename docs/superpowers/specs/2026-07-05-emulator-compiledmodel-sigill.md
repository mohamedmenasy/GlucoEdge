# Decision record: CompiledModel SIGILLs on Apple-Silicon-hosted emulators

Date: 2026-07-05
Status: settled (real-hardware verification pending)

## What happened

`CompiledModel.create()` unconditionally SIGILL-crashes the process on the
project's arm64 AVD (`Medium_Phone_API_35`, Apple-Silicon host), for both
`trend_float.tflite` and `trend_int8.tflite`, on litert 2.1.0 and 2.1.6
alike. The faulting instruction is the sole `rdsvl` (an ARM SVE "read
streaming vector length" instruction — a CPU feature probe) inside
`libLiteRt.so`, disassembled at PC `0x4a2324`. The guest kernel's
`/proc/cpuinfo` advertises `sve2`/`sme2` support (inherited from the host
Apple Silicon chip via Hypervisor.framework passthrough), but the
virtualized vCPU cannot actually execute those instructions — a known class
of gap in ARM virtualization where advertised HWCAP doesn't match
executable ISA. The probe runs unconditionally during model compilation,
before any `Options`/`Accelerator` choice takes effect.

## Why there is no in-process fix

No `CompiledModel`-side switch exists to disable the XNNPACK ISA probe
(`CompiledModel.CpuOptions.xnnPackFlags` is a delegate-node-enablement
bitmask, not a kill switch; `Environment.Option` exposes only NPU
compiler-plugin paths). A SIGILL is uncatchable by Kotlin/JVM code, so
there is no runtime try/catch that can fall back after the fact — the
engine has to be chosen statically, before either native call is made.

## Resolution

`TrendClassifier` selects its backend once at `create()` time, from a
build-time device signal (`isEmulator()` — `Build.HARDWARE` is `ranchu` or
`cutf`, `Build.FINGERPRINT` contains `generic`, or `Build.MODEL` contains
`sdk_gphone`), never via try/catch around `CompiledModel.create()`.
Emulators matching that signal run the classic `org.tensorflow.lite.
Interpreter` with `setUseXNNPACK(false)`, which avoids the crashing probe.
`CompiledModel` remains the intended real-hardware path — it compiles and
its API surface is `javap`-verified against the actual litert 2.1.0 jar.

**Update 2026-07-07 — verified on real hardware.** The golden-vector
parity suite (`GoldenParityTest`, `connectedDebugAndroidTest`) was run on
a physical Samsung Galaxy S22 Ultra (SM-S908E, Android 16): 3/3 tests
passed with the device gate routing to `CompiledModel` — no SIGILL, float
logits matched Python within 1e-5 and INT8 dequantized outputs matched
bit-exactly across all 20 vectors. In-app latency on the same device
(full Kotlin `classify()` path, including buffer writes, quantization,
and softmax — a wider measurement than the conversion phase's
invoke-only CPU proxy): float mean 0.334 ms / p95 0.540 ms over a full
100-inference window; INT8 mean 0.484 ms / p95 0.665 ms over 21
inferences. INT8 showed no latency advantage on-device, consistent with
the CPU-proxy conclusion.
