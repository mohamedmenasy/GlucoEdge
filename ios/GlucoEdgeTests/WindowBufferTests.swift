import Testing
@testable import GlucoEdge

struct WindowBufferTests {
    private func reading(min: Int64, gl: Float = 100) -> Reading {
        Reading(epochMinutes: min, mgdl: gl)
    }

    @Test func emitsWindowOnlyWhenFull() {
        let buffer = WindowBuffer(windowSize: 3)
        #expect(buffer.add(reading(min: 0, gl: 1)).window == nil)
        #expect(buffer.add(reading(min: 5, gl: 2)).window == nil)
        let result = buffer.add(reading(min: 10, gl: 3))
        #expect(result.window == [1, 2, 3])
        #expect(result.fill == 3)
    }

    @Test func slidesOldestOut() {
        let buffer = WindowBuffer(windowSize: 3)
        _ = buffer.add(reading(min: 0, gl: 1))
        _ = buffer.add(reading(min: 5, gl: 2))
        _ = buffer.add(reading(min: 10, gl: 3))
        #expect(buffer.add(reading(min: 15, gl: 4)).window == [2, 3, 4])
    }

    @Test func gapBeyondCadenceResetsBuffer() {
        let buffer = WindowBuffer(windowSize: 3)
        _ = buffer.add(reading(min: 0))
        _ = buffer.add(reading(min: 5))
        let result = buffer.add(reading(min: 20))   // 15-minute gap: reset
        #expect(result.wasReset)
        #expect(result.fill == 1)
        #expect(result.window == nil)
    }

    @Test func gapOfExactlyCadenceDoesNotReset() {
        // Review Focus 2: the rule is strictly greater-than, same as training's
        // segment rule and the Kotlin buffer. A >= here would reset on every
        // normal 5-minute step.
        let buffer = WindowBuffer(windowSize: 3)
        _ = buffer.add(reading(min: 0))
        let result = buffer.add(reading(min: 5))
        #expect(!result.wasReset)
        #expect(result.fill == 2)
    }

    @Test func resetClearsValuesAndGapTracking() {
        let buffer = WindowBuffer(windowSize: 2)
        _ = buffer.add(reading(min: 0))
        buffer.reset()
        // After reset() there is no "previous reading": a large time jump is not a gap.
        let result = buffer.add(reading(min: 1000))
        #expect(!result.wasReset)
        #expect(result.fill == 1)
    }
}
