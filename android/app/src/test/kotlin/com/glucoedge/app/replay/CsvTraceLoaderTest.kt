package com.glucoedge.app.replay

import java.io.StringReader
import org.junit.Assert.assertEquals
import org.junit.Test

class CsvTraceLoaderTest {
    private fun load(csv: String) = CsvTraceLoader.load(StringReader(csv))

    @Test fun parsesHeaderAndRows() {
        val r = load("time,gl\n2026-01-01T00:00,110.0\n2026-01-01T00:05,112.5\n")
        assertEquals(2, r.readings.size)
        assertEquals(0, r.skippedRows)
        assertEquals(110.0f, r.readings[0].mgdl, 1e-6f)
        assertEquals(5L, r.readings[1].epochMinutes - r.readings[0].epochMinutes)
    }

    @Test fun skipsAndCountsMalformedRows() {
        val r = load("time,gl\n2026-01-01T00:00,110.0\nnot-a-time,99\n2026-01-01T00:05,abc\n2026-01-01T00:10,120.0\n")
        assertEquals(2, r.readings.size)
        assertEquals(2, r.skippedRows)
    }

    @Test fun emptyBodyYieldsNoReadings() {
        val r = load("time,gl\n")
        assertEquals(0, r.readings.size)
    }
}
