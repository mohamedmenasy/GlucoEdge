package com.glucoedge.app.inference

import android.content.Context
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
}

private data class QuantParams(val scale: Float, val zeroPoint: Int)

/**
 * ENVIRONMENT-FORCED DEVIATION FROM THE INTENDED DESIGN (see task-8-report.md for the full
 * evidence trail; summarized here because it explains every line below):
 *
 * The intended hot path is LiteRT's `CompiledModel` API. On the required test environment (AVD
 * Medium_Phone_API_35, an Apple Silicon host), `CompiledModel.create()` unconditionally
 * SIGILL-crashes the whole process - for BOTH the float and int8 model, on both litert 2.1.0 and
 * 2.1.6 - at an `rdsvl` (ARM SVE "read streaming vector length") instruction inside a native CPU
 * feature probe in libLiteRt.so. That probe runs before any `Options`/`Accelerator` choice takes
 * effect, so it isn't something this code can configure around. The guest kernel reports SVE2/
 * SME2 support in `/proc/cpuinfo` (inherited from the host M-series chip via
 * Hypervisor.framework passthrough) that this virtualized environment can't actually execute.
 * A SIGILL kills the process outright - it is not a catchable JVM exception - so there is no way
 * to attempt `CompiledModel` and gracefully fall back at runtime; the backend has to be chosen
 * statically, at compile time.
 *
 * The classic `Interpreter`, with XNNPACK explicitly disabled, avoids the crashing probe (XNNPACK
 * is what runs it) and was verified not to crash for either model. So both models run through
 * `Interpreter` here, not just the int8 one. This is broader than the ladder's rung 3 anticipated
 * ("if the CompiledModel TensorBuffer genuinely has no int8 write path") - the TensorBuffer int8
 * write path is real and fine (confirmed via `javap`: `writeInt8(byte[])` / `readInt8()` exist
 * verbatim on `com.google.ai.edge.litert.TensorBuffer`, and the CompiledModel-based
 * implementation compiled cleanly on the first try). The blocker is a native crash in
 * `CompiledModel.create()` itself, unconditionally, unrelated to int8 vs float.
 */
class TrendClassifier private constructor(
    private val interpreter: Interpreter,
    private val inputQuant: QuantParams?,   // null => float model
    private val outputQuant: QuantParams?,
) : Classifier {
    // 1 byte/element for int8, 4 bytes/element (float32) for the float model.
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(12 * if (inputQuant == null) 4 else 1).order(ByteOrder.nativeOrder())
    private val outputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(5 * if (outputQuant == null) 4 else 1).order(ByteOrder.nativeOrder())

    override fun classify(window: FloatArray): Prediction {
        require(window.size == 12) { "expected a 12-reading window" }
        val start = System.nanoTime()
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
        val logits = if (outputQuant == null) {
            val out = FloatArray(5)
            outputBuffer.asFloatBuffer().get(out)
            out
        } else {
            val raw = ByteArray(5)
            outputBuffer.get(raw)
            QuantizationMath.dequantizeInt8(raw, outputQuant.scale, outputQuant.zeroPoint)
        }
        val elapsed = System.nanoTime() - start
        val probs = QuantizationMath.softmax(logits)
        val cls = probs.indices.maxBy { probs[it] }
        return Prediction(probs, cls, elapsed, logits)
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        val CLASS_NAMES = listOf("falling_fast", "falling", "stable", "rising", "rising_fast")

        fun create(context: Context, model: ModelFile): TrendClassifier {
            val file = File(context.cacheDir, model.assetName)
            context.assets.open(model.assetName).use { input ->
                file.outputStream().use { input.copyTo(it) }
            }
            val quantized = model == ModelFile.INT8
            val buffer: ByteBuffer = file.readBytes().let {
                ByteBuffer.allocateDirect(it.size).order(ByteOrder.nativeOrder()).put(it).apply { rewind() }
            }
            // useXNNPACK(false): see the class doc above - XNNPACK's native CPU feature probe
            // SIGILL-crashes on this test environment regardless of model or accelerator choice.
            val options = Interpreter.Options().setUseXNNPACK(false)
            val interpreter = Interpreter(buffer, options)
            val (inQ, outQ) = if (quantized) {
                val inTq = interpreter.getInputTensor(0).quantizationParams()
                val outTq = interpreter.getOutputTensor(0).quantizationParams()
                QuantParams(inTq.scale, inTq.zeroPoint) to QuantParams(outTq.scale, outTq.zeroPoint)
            } else {
                null to null
            }
            return TrendClassifier(interpreter, inQ, outQ)
        }
    }
}
