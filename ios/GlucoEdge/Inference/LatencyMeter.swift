struct LatencyStats: Equatable {
    let count: Int
    let meanMs: Double
    let medianMs: Double
    let p95Ms: Double

    static let empty = LatencyStats(count: 0, meanMs: 0, medianMs: 0, p95Ms: 0)
}

/// Rolling window of the last `capacity` inference wall times.
/// Used only from the main actor — no synchronization needed.
final class LatencyMeter {
    let capacity: Int
    private var samplesNanos: [Int64] = []

    init(capacity: Int = 100) {
        self.capacity = capacity
    }

    func record(_ nanos: Int64) {
        samplesNanos.append(nanos)
        if samplesNanos.count > capacity { samplesNanos.removeFirst() }
    }

    func reset() { samplesNanos.removeAll() }

    var stats: LatencyStats {
        if samplesNanos.isEmpty { return .empty }
        let ms = samplesNanos.map { Double($0) / 1e6 }.sorted()
        let n = ms.count
        let median = n % 2 == 1 ? ms[n / 2] : (ms[n / 2 - 1] + ms[n / 2]) / 2.0
        // Same index arithmetic as the Kotlin meter: sorted[((n-1)*95)/100].
        let p95 = ms[((n - 1) * 95) / 100]
        return LatencyStats(count: n, meanMs: ms.reduce(0, +) / Double(n), medianMs: median, p95Ms: p95)
    }
}
