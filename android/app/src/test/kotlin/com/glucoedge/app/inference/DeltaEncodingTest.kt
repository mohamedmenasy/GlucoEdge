package com.glucoedge.app.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class DeltaEncodingTest {
    @Test
    fun deltaTransformMatchesPython() {
        // Hand vectors from tests/test_deltas.py: 11 consecutive differences
        // plus the last absolute value / 16, float32 order preserved.
        val window = FloatArray(12) { i -> 100f + i * i } // 100,101,104,109,...,221
        val out = DeltaEncoding.encode(window)
        assertEquals(12, out.size)
        for (i in 0 until 11) {
            assertEquals(window[i + 1] - window[i], out[i], 0f)
        }
        assertEquals(window[11] / 16f, out[11], 0f)
    }

    @Test
    fun anchorForCeilingGlucoseStaysInDeltaRange() {
        val window = FloatArray(12) { 401f }
        val out = DeltaEncoding.encode(window)
        assertEquals(401f / 16f, out[11], 0f) // 25.0625
        for (i in 0 until 11) assertEquals(0f, out[i], 0f)
    }
}
