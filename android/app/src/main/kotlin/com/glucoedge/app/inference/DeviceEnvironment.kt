package com.glucoedge.app.inference

import android.os.Build

/**
 * Detects the emulator/host pairing where `CompiledModel.create()` is known to SIGILL-crash
 * the process (see `docs/superpowers/specs/2026-07-05-emulator-compiledmodel-sigill.md`): a
 * native ARM SVE feature-probe instruction (`rdsvl`) inside `libLiteRt.so`, run unconditionally
 * at model-compile time, faults because this AVD's guest kernel *advertises* SVE2/SME2
 * (inherited from an Apple Silicon host via Hypervisor.framework passthrough) without actually
 * being able to execute it. A SIGILL kills the whole process and cannot be caught by
 * Kotlin/JVM code, so the engine must be chosen statically from build-time device signals,
 * never via a runtime try/catch around `CompiledModel.create()`.
 *
 * Single shared source of truth for both [com.glucoedge.app.inference.TrendClassifier]'s engine
 * selection and [com.glucoedge.app.MainActivity]'s UI device label - the two call sites must
 * agree on what counts as "emulator".
 */
fun isEmulator(): Boolean =
    Build.HARDWARE == "ranchu" ||
        Build.HARDWARE == "cutf" ||
        Build.FINGERPRINT.contains("generic") ||
        Build.MODEL.contains("sdk_gphone")
