package com.glucoedge.app.replay

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WindowBufferTest {
    private fun reading(i: Long, v: Float = 100f + i) = Reading(epochMinutes = i * 5, mgdl = v)

    @Test fun emitsNothingUntilTwelveReadings() {
        val buf = WindowBuffer()
        for (i in 0L..10L) {
            val r = buf.add(reading(i))
            assertNull(r.window)
            assertEquals((i + 1).toInt(), r.fill)
        }
        val full = buf.add(reading(11))
        assertEquals(12, full.window!!.size)
        assertEquals(100f, full.window!![0], 1e-6f)
        assertEquals(111f, full.window!![11], 1e-6f)
    }

    @Test fun slidesByOneAfterFull() {
        val buf = WindowBuffer()
        for (i in 0L..11L) buf.add(reading(i))
        val next = buf.add(reading(12))
        assertEquals(101f, next.window!![0], 1e-6f)
        assertEquals(112f, next.window!![11], 1e-6f)
    }

    @Test fun resetsOnTimeGap() {
        val buf = WindowBuffer()
        for (i in 0L..11L) buf.add(reading(i))
        // 25-minute jump: window must never span the gap (id_segment rule)
        val afterGap = buf.add(Reading(epochMinutes = 11 * 5 + 25, mgdl = 150f))
        assertTrue(afterGap.wasReset)
        assertNull(afterGap.window)
        assertEquals(1, afterGap.fill)
    }

    @Test fun exactCadenceIsNotAGap() {
        val buf = WindowBuffer()
        buf.add(reading(0))
        val r = buf.add(reading(1))
        assertFalse(r.wasReset)
    }
}
