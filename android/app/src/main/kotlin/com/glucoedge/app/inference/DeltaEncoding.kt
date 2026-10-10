package com.glucoedge.app.inference

/**
 * Delta input encoding (paper: delta-encoded inputs). The bundled models
 * consume 11 consecutive differences plus the last absolute reading / 16,
 * all float32, computed BEFORE input quantization. Must stay bit-identical
 * to Python's `training/deltas.py::delta_windows(..., anchor=True)` and the
 * Swift `DeltaEncoding.encode` - golden vectors pin all three.
 */
object DeltaEncoding {
    const val ANCHOR_SCALE = 16f

    fun encode(window: FloatArray): FloatArray {
        val out = FloatArray(window.size)
        for (i in 0 until window.size - 1) {
            out[i] = window[i + 1] - window[i]
        }
        out[window.size - 1] = window[window.size - 1] / ANCHOR_SCALE
        return out
    }
}
