package com.glucoedge.app.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class LatencyMeterTest {
    @Test fun statsOverKnownValues() {
        val m = LatencyMeter()
        // 1..100 ms recorded as nanos
        (1..100).forEach { m.record(it * 1_000_000L) }
        val s = m.stats
        assertEquals(100, s.count)
        assertEquals(50.5, s.meanMs, 1e-9)
        assertEquals(50.5, s.medianMs, 1e-9)
        assertEquals(95.0, s.p95Ms, 1.0)
    }

    @Test fun rollsOverCapacity() {
        val m = LatencyMeter(capacity = 10)
        (1..20).forEach { m.record(it * 1_000_000L) }
        assertEquals(10, m.stats.count)
        assertEquals(15.5, m.stats.meanMs, 1e-9)  // only 11..20 retained
    }

    @Test fun emptyMeterIsAllZero() {
        val s = LatencyMeter().stats
        assertEquals(0, s.count)
        assertEquals(0.0, s.meanMs, 0.0)
    }

    @Test fun resetClears() {
        val m = LatencyMeter()
        m.record(5_000_000L); m.reset()
        assertEquals(0, m.stats.count)
    }
}
