import Foundation

/// Delta input encoding (paper: delta-encoded inputs). The bundled models
/// consume 11 consecutive differences plus the last absolute reading / 16,
/// all Float32, computed BEFORE input quantization. Must stay bit-identical
/// to Python's `training/deltas.py::delta_windows(..., anchor: true)` and
/// the Kotlin `DeltaEncoding.encode` — golden vectors pin all three.
enum DeltaEncoding {
    static let anchorScale: Float = 16

    static func encode(_ window: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: window.count)
        for i in 0..<(window.count - 1) {
            out[i] = window[i + 1] - window[i]
        }
        out[window.count - 1] = window[window.count - 1] / anchorScale
        return out
    }
}
