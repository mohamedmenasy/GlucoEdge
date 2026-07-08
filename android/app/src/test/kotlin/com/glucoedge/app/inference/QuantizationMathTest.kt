package com.glucoedge.app.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class QuantizationMathTest {
    // The INT8 model's real params from the conversion record:
    private val scale = 0.9408126473426819f
    private val zeroPoint = -128

    @Test fun quantizesInRangeValueByRoundThenOffset() {
        // 110 / 0.9408... = 116.92 -> round 117; 117 + (-128) = -11
        val q = QuantizationMath.quantizeInt8(floatArrayOf(110f), scale, zeroPoint)
        assertEquals((-11).toByte(), q[0])
    }

    @Test fun clipsAboveCeilingInsteadOfWrapping() {
        // 260 mg/dL is above the ~239.9 ceiling: must clip to +127, never wrap negative
        val q = QuantizationMath.quantizeInt8(floatArrayOf(260f, 401f), scale, zeroPoint)
        assertArrayEquals(byteArrayOf(127, 127), q)
    }

    @Test fun clipsBelowFloor() {
        val q = QuantizationMath.quantizeInt8(floatArrayOf(-50f), scale, zeroPoint)
        assertEquals((-128).toByte(), q[0])
    }

    @Test fun dequantizeRoundTripsWithinScale() {
        val original = floatArrayOf(80f, 110f, 180f, 239f)
        val q = QuantizationMath.quantizeInt8(original, scale, zeroPoint)
        val back = QuantizationMath.dequantizeInt8(q, scale, zeroPoint)
        original.zip(back.toTypedArray()).forEach { (a, b) ->
            assertEquals(a, b, scale)  // error bounded by one quantization step
        }
    }

    @Test fun softmaxIsStableAndNormalized() {
        val p = QuantizationMath.softmax(floatArrayOf(1000f, 1001f, 999f, 0f, -1000f))
        // Probabilities sum to 1
        assertEquals(1.0f, p.sum(), 1e-5f)
        // Max probability is at index 1 (largest logit is 1001f at index 1)
        assertEquals(1, p.withIndex().maxByOrNull { it.value }?.index)
        // Extreme logits must not overflow to NaN
        assert(p.all { !it.isNaN() })
    }
}
