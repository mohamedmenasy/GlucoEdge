import Testing
@testable import GlucoEdge

struct CsvTraceLoaderTests {
    @Test func parsesHeaderAndRows() {
        let result = CsvTraceLoader.load("""
        time,gl
        2026-01-01T00:00,110.0
        2026-01-01T00:05,112.5
        """)
        #expect(result.skippedRows == 0)
        #expect(result.readings.count == 2)
        // 2026-01-01T00:00 UTC = 1767225600 s = 29453760 min
        #expect(result.readings[0] == Reading(epochMinutes: 29_453_760, mgdl: 110.0))
        #expect(result.readings[1].epochMinutes - result.readings[0].epochMinutes == 5)
    }

    @Test func countsMalformedRowsAsSkipped() {
        let result = CsvTraceLoader.load("""
        time,gl
        2026-01-01T00:00,110.0
        not-a-time,99.0
        2026-01-01T00:10,abc
        2026-01-01T00:15
        2026-01-01T00:20,120.0
        """)
        #expect(result.readings.count == 2)
        #expect(result.skippedRows == 3)
    }

    @Test func blankLinesAreIgnoredNotCounted() {
        let result = CsvTraceLoader.load("time,gl\n\n2026-01-01T00:00,110.0\n\n")
        #expect(result.readings.count == 1)
        #expect(result.skippedRows == 0)
    }

    @Test func parsesCrlfLineEndings() {
        // Review Focus 1: a CSV saved with \r\n must parse, not skip every row.
        let result = CsvTraceLoader.load("time,gl\r\n2026-01-01T00:00,110.0\r\n2026-01-01T00:05,112.0\r\n")
        #expect(result.readings.count == 2)
        #expect(result.skippedRows == 0)
    }
}
