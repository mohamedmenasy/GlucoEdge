import Testing
@testable import GlucoEdge

struct LatencyMeterTests {
    @Test func emptyMeterReportsZeros() {
        #expect(LatencyMeter().stats == LatencyStats.empty)
    }

    @Test func computesMeanMedianP95InMilliseconds() {
        let meter = LatencyMeter()
        for nanos: Int64 in [1_000_000, 2_000_000, 3_000_000, 4_000_000] {
            meter.record(nanos)
        }
        let stats = meter.stats
        #expect(stats.count == 4)
        #expect(abs(stats.meanMs - 2.5) < 1e-9)
        #expect(abs(stats.medianMs - 2.5) < 1e-9)   // even count: average of middle two
        #expect(abs(stats.p95Ms - 3.0) < 1e-9)      // sorted[(3*95)/100] = sorted[2]
    }

    @Test func oddCountMedianIsMiddleElement() {
        let meter = LatencyMeter()
        for nanos: Int64 in [5_000_000, 1_000_000, 3_000_000] { meter.record(nanos) }
        #expect(abs(meter.stats.medianMs - 3.0) < 1e-9)
    }

    @Test func rollingWindowEvictsOldest() {
        let meter = LatencyMeter(capacity: 2)
        meter.record(1_000_000)
        meter.record(2_000_000)
        meter.record(9_000_000)  // evicts the 1 ms sample
        let stats = meter.stats
        #expect(stats.count == 2)
        #expect(abs(stats.meanMs - 5.5) < 1e-9)
    }

    @Test func resetClearsSamples() {
        let meter = LatencyMeter()
        meter.record(1_000_000)
        meter.reset()
        #expect(meter.stats == LatencyStats.empty)
    }
}
