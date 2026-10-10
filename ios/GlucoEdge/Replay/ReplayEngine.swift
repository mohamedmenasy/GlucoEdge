import Foundation

enum Speed: Int, CaseIterable, Sendable {
    case x1 = 1, x4 = 4, x16 = 16
}

/// Replays a fixed trace on a virtual clock: one reading per interval while
/// playing. The real 5-minute CGM cadence is compressed to `baseInterval`
/// per reading at 1x. Same pause semantics as the Kotlin Flow version:
/// nothing is emitted while paused, and the playing state is re-checked
/// after each delay so a pause during the sleep still blocks the emission.
@MainActor
final class ReplayEngine {
    private let readings: [Reading]
    private let baseInterval: Duration
    private(set) var isPlaying = false
    private(set) var speed: Speed = .x1

    init(readings: [Reading], baseInterval: Duration = .seconds(5)) {
        self.readings = readings
        self.baseInterval = baseInterval
    }

    /// Single-consumer stream of the replayed readings; finishes after the last one.
    var events: AsyncStream<Reading> {
        let (stream, continuation) = AsyncStream.makeStream(of: Reading.self)
        let task = Task {
            for reading in readings {
                await waitUntilPlaying()
                try? await Task.sleep(for: baseInterval / speed.rawValue)
                await waitUntilPlaying()
                if Task.isCancelled { break }
                continuation.yield(reading)
            }
            continuation.finish()
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    // ponytail: 10 ms poll — cancellation-safe and simple; switch to parked
    // continuations only if play() wakeup latency ever becomes visible.
    private func waitUntilPlaying() async {
        while !isPlaying && !Task.isCancelled {
            try? await Task.sleep(for: .milliseconds(10))
        }
    }

    func play() { isPlaying = true }

    func pause() { isPlaying = false }

    func setSpeed(_ speed: Speed) { self.speed = speed }
}
