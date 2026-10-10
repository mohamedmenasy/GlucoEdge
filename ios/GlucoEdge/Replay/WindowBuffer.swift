/// Sliding window of the last `windowSize` readings. Resets whenever the
/// time since the previous reading exceeds `cadenceMinutes` — the app-side
/// mirror of training's id_segment rule: a window must never span a gap in
/// the sensor record.
final class WindowBuffer {
    struct AddResult: Equatable {
        let window: [Float]?
        let wasReset: Bool
        let fill: Int
    }

    let windowSize: Int
    let cadenceMinutes: Int64
    private var values: [Float] = []
    private var lastEpochMinutes: Int64?

    init(windowSize: Int = 12, cadenceMinutes: Int64 = 5) {
        self.windowSize = windowSize
        self.cadenceMinutes = cadenceMinutes
    }

    func add(_ reading: Reading) -> AddResult {
        let wasReset = lastEpochMinutes.map { reading.epochMinutes - $0 > cadenceMinutes } ?? false
        if wasReset { values.removeAll() }
        lastEpochMinutes = reading.epochMinutes
        values.append(reading.mgdl)
        if values.count > windowSize { values.removeFirst() }
        let window = values.count == windowSize ? values : nil
        return AddResult(window: window, wasReset: wasReset, fill: values.count)
    }

    func reset() {
        values.removeAll()
        lastEpochMinutes = nil
    }
}
