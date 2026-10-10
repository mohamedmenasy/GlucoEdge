import Testing
@testable import GlucoEdge

@MainActor
private final class Collector {
    var values: [Float] = []
}

@MainActor
struct ReplayEngineTests {
    private let trace = [Reading(epochMinutes: 0, mgdl: 100),
                         Reading(epochMinutes: 5, mgdl: 110),
                         Reading(epochMinutes: 10, mgdl: 120)]

    /// First element of the stream, or nil if `timeout` elapses first.
    private func firstOrTimeout(_ stream: AsyncStream<Reading>, timeout: Duration) async -> Reading? {
        await withTaskGroup(of: Reading?.self) { group in
            group.addTask { for await r in stream { return r }; return nil }
            group.addTask { try? await Task.sleep(for: timeout); return nil }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    /// Polls `condition` every millisecond for up to ~2 s.
    private func waitUntil(_ condition: () -> Bool) async {
        for _ in 0..<2000 where !condition() {
            try? await Task.sleep(for: .milliseconds(1))
        }
    }

    @Test func emitsAllReadingsInOrderWhilePlaying() async {
        let engine = ReplayEngine(readings: trace, baseInterval: .milliseconds(1))
        engine.play()
        var seen: [Float] = []
        for await r in engine.events { seen.append(r.mgdl) }
        #expect(seen == [100, 110, 120])
    }

    @Test func emitsNothingWhilePaused() async {
        let engine = ReplayEngine(readings: trace, baseInterval: .milliseconds(1))
        // never played: 100 ms with a 1 ms interval would emit everything if broken
        let first = await firstOrTimeout(engine.events, timeout: .milliseconds(100))
        #expect(first == nil)
    }

    @Test func pauseStopsAndPlayResumes() async {
        // Long enough trace that the pause lands mid-stream (3 readings at
        // 1 ms would finish before pause() and pass vacuously).
        let long = (0..<40).map { Reading(epochMinutes: Int64($0) * 5, mgdl: Float($0)) }
        let engine = ReplayEngine(readings: long, baseInterval: .milliseconds(5))
        let collector = Collector()
        let consumer = Task { for await r in engine.events { collector.values.append(r.mgdl) } }
        engine.play()
        await waitUntil { collector.values.count >= 1 }
        engine.pause()
        // Let any reading already yielded before the pause drain, then require silence.
        try? await Task.sleep(for: .milliseconds(50))
        let countAtPause = collector.values.count
        #expect(countAtPause < 40, "pause must land mid-stream for this test to mean anything")
        try? await Task.sleep(for: .milliseconds(100))
        #expect(collector.values.count == countAtPause)
        engine.play()
        await waitUntil { collector.values.count == 40 }
        #expect(collector.values == long.map(\.mgdl))
        consumer.cancel()
    }

    @Test func speedSettingIsExposed() {
        let engine = ReplayEngine(readings: trace)
        #expect(engine.speed == .x1)
        engine.setSpeed(.x16)
        #expect(engine.speed == .x16)
    }
}
