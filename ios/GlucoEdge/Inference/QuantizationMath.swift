import Foundation

enum QuantizationMath {
    /// Affine int8 quantization: round(x/scale) + zeroPoint, CLIPPED to
    /// [-128, 127]. The clip is load-bearing: without it, glucose readings
    /// above the representable ceiling wrap around int8 (260 mg/dL would
    /// become ~19 mg/dL) — the exact bug the conversion-phase review caught
    /// in the Python benchmark. Ties round half to even, like np.round and
    /// the Kotlin client, so all three produce identical codes.
    static func quantizeInt8(_ values: [Float], scale: Float, zeroPoint: Int) -> [Int8] {
        values.map { v in
            let q = Int((v / scale).rounded(.toNearestOrEven)) + zeroPoint
            return Int8(clamping: q)
        }
    }

    static func dequantizeInt8(_ values: [Int8], scale: Float, zeroPoint: Int) -> [Float] {
        values.map { Float(Int($0) - zeroPoint) * scale }
    }

    static func softmax(_ logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else { return [] }
        let exps = logits.map { Float(exp(Double($0 - maxLogit))) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
}
