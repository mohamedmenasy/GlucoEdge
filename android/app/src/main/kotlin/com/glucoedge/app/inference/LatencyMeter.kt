package com.glucoedge.app.inference

data class LatencyStats(val count: Int, val meanMs: Double, val medianMs: Double, val p95Ms: Double)

/** Rolling window of the last [capacity] inference wall times. */
class LatencyMeter(val capacity: Int = 100) {
    private val samplesNanos = ArrayDeque<Long>()

    @Synchronized
    fun record(nanos: Long) {
        samplesNanos.addLast(nanos)
        if (samplesNanos.size > capacity) samplesNanos.removeFirst()
    }

    @Synchronized
    fun reset() = samplesNanos.clear()

    val stats: LatencyStats
        @Synchronized get() {
            if (samplesNanos.isEmpty()) return LatencyStats(0, 0.0, 0.0, 0.0)
            val ms = samplesNanos.map { it / 1e6 }.sorted()
            val median = if (ms.size % 2 == 1) ms[ms.size / 2]
                         else (ms[ms.size / 2 - 1] + ms[ms.size / 2]) / 2.0
            val p95 = ms[((ms.size - 1) * 95) / 100]
            return LatencyStats(ms.size, ms.average(), median, p95)
        }
}
