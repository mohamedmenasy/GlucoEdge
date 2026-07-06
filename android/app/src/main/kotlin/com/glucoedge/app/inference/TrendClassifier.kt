package com.glucoedge.app.inference

import android.content.Context
import android.os.Build
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter

enum class ModelFile(val assetName: String) {
    FLOAT("trend_float.tflite"), INT8("trend_int8.tflite")
}

data class Prediction(
    val probabilities: FloatArray,
    val classIndex: Int,
    val latencyNanos: Long,
    // Raw model output (pre-softmax). Present so the golden-vector parity test can assert
    // logit-level agreement with the Python benchmark, not just softmax-then-argmax class -
    // rung 1 of the int8 buffer ladder held (real writeInt8/readInt8 on TensorBuffer), so this
    // extra fidelity was free to add.
    val logits: FloatArray,
)

interface Classifier {
    fun classify(window: FloatArray): Prediction
    fun close()

    /** Exactly "CompiledModel" or "Interpreter (emulator fallback)" - surfaced in the UI. */
    val engineLabel: String
}

private data class QuantParams(val scale: Float, val zeroPoint: Int)

/**
 * Detects the emulator/host pairing where `CompiledModel.create()` is known to SIGILL-crash
 * the process (see task-8-report.md): a native ARM SVE feature-probe instruction (`rdsvl`)
 * inside `libLiteRt.so`, run unconditionally at model-compile time, faults because this AVD's
 * guest kernel *advertises* SVE2/SME2 (inherited from an Apple Silicon host via
 * Hypervisor.framework passthrough) without actually being able to execute it. A SIGILL kills
 * the whole process and cannot be caught by Kotlin/JVM code, so the engine must be chosen
 * statically from build-time device signals, never via a runtime try/catch around
 * `CompiledModel.create()`.
 */
private fun isEmulator(): Boolean =
    Build.HARDWARE == "ranchu" ||
        Build.HARDWARE == "cutf" ||
        Build.FINGERPRINT.contains("generic") ||
        Build.MODEL.contains("sdk_gphone")

/**
 * Wraps a LiteRT inference backend (the hot path). For the INT8 model, input quantization
 * params are read ONCE at load time via the classic Interpreter API (same artifact) - never
 * hardcoded - and the Kotlin quantization replicates the corrected Python benchmark: round
 * then clip.
 *
 * ## Engine selection
 *
 * The project brief's intended hot path is LiteRT's `CompiledModel` API, and that is what
 * every real device runs. The one carve-out is emulators: `CompiledModel.create()`
 * unconditionally SIGILL-crashes the process on the required test AVD (`Medium_Phone_API_35`,
 * Apple-Silicon-hosted) - a native ARM SVE feature-probe (`rdsvl`) inside `libLiteRt.so`, run
 * before any `Options`/`Accelerator` choice takes effect, faults because the guest kernel
 * advertises SVE2/SME2 it can't actually execute (full evidence, disassembly, and tombstones:
 * `.superpowers/sdd/task-8-report.md`). Since a SIGILL can't be caught at runtime, the engine
 * is chosen statically at load time from [isEmulator], not via try/catch around
 * `CompiledModel.create()`.
 *
 * **The `CompiledModel` branch is UNVERIFIED on real hardware.** It compiles and its API
 * surface matches the brief exactly (confirmed via `javap` against litert 2.1.0), but it has
 * never been executed - this environment only has the SIGILL-affected emulator available. It
 * stays unverified until the golden-vector parity suite is run on a physical device.
 */
class TrendClassifier private constructor(private val engine: Engine) : Classifier {

    override val engineLabel: String get() = engine.label

    override fun classify(window: FloatArray): Prediction {
        require(window.size == 12) { "expected a 12-reading window" }
        val start = System.nanoTime()
        val logits = engine.runInference(window)
        val elapsed = System.nanoTime() - start
        val probs = QuantizationMath.softmax(logits)
        val cls = probs.indices.maxBy { probs[it] }
        return Prediction(probs, cls, elapsed, logits)
    }

    override fun close() = engine.close()

    /**
     * A backend capable of running the bundled `.tflite` file and returning raw logits for a
     * 12-reading window. Both implementations share identical quantization math
     * ([QuantizationMath]) so [classify] produces bit-for-bit comparable [Prediction]s
     * regardless of which engine loaded.
     */
    private interface Engine {
        val label: String
        fun runInference(window: FloatArray): FloatArray
        fun close()
    }

    /** Real-hardware hot path: LiteRT's `CompiledModel` API. Unverified - see class KDoc. */
    private class CompiledModelEngine(
        private val compiledModel: CompiledModel,
        private val inputQuant: QuantParams?,
        private val outputQuant: QuantParams?,
    ) : Engine {
        override val label = "CompiledModel"
        private val inputBuffers = compiledModel.createInputBuffers()
        private val outputBuffers = compiledModel.createOutputBuffers()

        override fun runInference(window: FloatArray): FloatArray {
            if (inputQuant == null) {
                inputBuffers[0].writeFloat(window)
            } else {
                inputBuffers[0].writeInt8(
                    QuantizationMath.quantizeInt8(window, inputQuant.scale, inputQuant.zeroPoint)
                )
            }
            compiledModel.run(inputBuffers, outputBuffers)
            return if (outputQuant == null) {
                outputBuffers[0].readFloat()
            } else {
                QuantizationMath.dequantizeInt8(
                    outputBuffers[0].readInt8(), outputQuant.scale, outputQuant.zeroPoint
                )
            }
        }

        override fun close() = compiledModel.close()
    }

    /**
     * Emulator fallback: the classic `org.tensorflow.lite.Interpreter`, XNNPACK explicitly
     * disabled to avoid the crashing SVE probe. Verified via the full instrumented parity
     * suite on `Medium_Phone_API_35`.
     */
    private class InterpreterEngine(
        private val interpreter: Interpreter,
        private val inputQuant: QuantParams?,
        private val outputQuant: QuantParams?,
    ) : Engine {
        override val label = "Interpreter (emulator fallback)"

        // 1 byte/element for int8, 4 bytes/element (float32) for the float model.
        private val inputBuffer: ByteBuffer =
            ByteBuffer.allocateDirect(12 * if (inputQuant == null) 4 else 1).order(ByteOrder.nativeOrder())
        private val outputBuffer: ByteBuffer =
            ByteBuffer.allocateDirect(5 * if (outputQuant == null) 4 else 1).order(ByteOrder.nativeOrder())

        override fun runInference(window: FloatArray): FloatArray {
            inputBuffer.rewind()
            if (inputQuant == null) {
                inputBuffer.asFloatBuffer().put(window)
            } else {
                inputBuffer.put(QuantizationMath.quantizeInt8(window, inputQuant.scale, inputQuant.zeroPoint))
            }
            inputBuffer.rewind()
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            return if (outputQuant == null) {
                val out = FloatArray(5)
                outputBuffer.asFloatBuffer().get(out)
                out
            } else {
                val raw = ByteArray(5)
                outputBuffer.get(raw)
                QuantizationMath.dequantizeInt8(raw, outputQuant.scale, outputQuant.zeroPoint)
            }
        }

        override fun close() = interpreter.close()
    }

    companion object {
        val CLASS_NAMES = listOf("falling_fast", "falling", "stable", "rising", "rising_fast")

        fun create(context: Context, model: ModelFile): TrendClassifier {
            val file = File(context.cacheDir, model.assetName)
            context.assets.open(model.assetName).use { input ->
                file.outputStream().use { input.copyTo(it) }
            }
            val quantized = model == ModelFile.INT8

            val engine = if (isEmulator()) {
                // useXNNPACK(false): the SVE feature-probe that SIGILL-crashes on emulators
                // lives in XNNPACK's native init path - see class KDoc and task-8-report.md.
                // Interpreter(File, Options) reads/mmaps the cacheDir copy directly - no
                // separate readBytes() + manual ByteBuffer wrap needed.
                val interpreter = Interpreter(file, Interpreter.Options().setUseXNNPACK(false))
                val (inQ, outQ) = readQuantParamsFromInterpreter(interpreter, quantized)
                InterpreterEngine(interpreter, inQ, outQ)
            } else {
                val (inQ, outQ) = readQuantParams(file, quantized)
                val compiled = CompiledModel.create(
                    file.absolutePath, CompiledModel.Options(Accelerator.CPU)
                )
                CompiledModelEngine(compiled, inQ, outQ)
            }
            return TrendClassifier(engine)
        }

        /** Reads quantization params off an already-open Interpreter (emulator path). */
        private fun readQuantParamsFromInterpreter(
            interpreter: Interpreter,
            quantized: Boolean,
        ): Pair<QuantParams?, QuantParams?> {
            if (!quantized) return null to null
            val inTq = interpreter.getInputTensor(0).quantizationParams()
            val outTq = interpreter.getOutputTensor(0).quantizationParams()
            return QuantParams(inTq.scale, inTq.zeroPoint) to QuantParams(outTq.scale, outTq.zeroPoint)
        }

        /**
         * One-shot metadata read via the classic Interpreter (same litert artifact), used only
         * on the `CompiledModel` (real-hardware) path since `CompiledModel` itself exposes no
         * quantization-params accessor. Reuses the cacheDir file `create()` already copied the
         * asset to (needed anyway for `CompiledModel.create(file.absolutePath, ...)` below) via
         * the `Interpreter(File, Options)` constructor - no separate manual byte read.
         */
        private fun readQuantParams(file: File, quantized: Boolean): Pair<QuantParams?, QuantParams?> {
            if (!quantized) return null to null
            val interpreter = Interpreter(file, Interpreter.Options().setUseXNNPACK(false))
            try {
                val inQ = interpreter.getInputTensor(0).quantizationParams()
                val outQ = interpreter.getOutputTensor(0).quantizationParams()
                return QuantParams(inQ.scale, inQ.zeroPoint) to QuantParams(outQ.scale, outQ.zeroPoint)
            } finally {
                interpreter.close()
            }
        }
    }
}
