package com.glucoedge.app.inference

import kotlin.math.exp
import kotlin.math.roundToInt

object QuantizationMath {
    /**
     * Affine int8 quantization: round(x/scale) + zeroPoint, CLIPPED to
     * [-128, 127]. The clip is load-bearing: without it, glucose readings
     * above the representable ceiling wrap around int8 (260 mg/dL would
     * become ~19 mg/dL) - the exact bug the conversion-phase review caught
     * in the Python benchmark.
     */
    fun quantizeInt8(values: FloatArray, scale: Float, zeroPoint: Int): ByteArray =
        ByteArray(values.size) { i ->
            val q = (values[i] / scale).roundToInt() + zeroPoint
            q.coerceIn(-128, 127).toByte()
        }

    fun dequantizeInt8(values: ByteArray, scale: Float, zeroPoint: Int): FloatArray =
        FloatArray(values.size) { i -> (values[i].toInt() - zeroPoint) * scale }

    fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }
}
