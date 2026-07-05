package com.glucoedge.app.replay

/**
 * Sliding window of the last [windowSize] readings. Resets whenever the
 * time since the previous reading exceeds [cadenceMinutes] - the app-side
 * mirror of training's id_segment rule: a window must never span a gap in
 * the sensor record.
 */
class WindowBuffer(val windowSize: Int = 12, val cadenceMinutes: Long = 5) {
    data class AddResult(val window: FloatArray?, val wasReset: Boolean, val fill: Int)

    private val values = ArrayDeque<Float>()
    private var lastEpochMinutes: Long? = null

    fun add(reading: Reading): AddResult {
        val last = lastEpochMinutes
        val wasReset = last != null && (reading.epochMinutes - last) > cadenceMinutes
        if (wasReset) values.clear()
        lastEpochMinutes = reading.epochMinutes
        values.addLast(reading.mgdl)
        if (values.size > windowSize) values.removeFirst()
        val window = if (values.size == windowSize) values.toFloatArray() else null
        return AddResult(window, wasReset, values.size)
    }

    fun reset() {
        values.clear()
        lastEpochMinutes = null
    }
}
