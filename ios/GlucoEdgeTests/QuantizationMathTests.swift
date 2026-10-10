import Testing
@testable import GlucoEdge

struct QuantizationMathTests {
    @Test func quantizesWithAffineTransform() {
        // round(100.4 / 0.94) = round(106.8) = 107; + (-128) = -21
        #expect(QuantizationMath.quantizeInt8([100.4], scale: 0.94, zeroPoint: -128) == [-21])
    }

    @Test func roundsExactTiesToEvenLikeNumpy() {
        // np.round / kotlin.math.round semantics: 0.5→0, 1.5→2, 2.5→2, -1.5→-2
        #expect(QuantizationMath.quantizeInt8([0.5, 1.5, 2.5, -1.5], scale: 1, zeroPoint: 0)
                == [0, 2, 2, -2])
    }

    @Test func saturatesInsteadOfWrapping() {
        // Review Focus 3: 260 mg/dL above the INT8 ceiling must clip to 127,
        // never wrap to a low glucose code (the conversion-phase Critical bug).
        #expect(QuantizationMath.quantizeInt8([260.0], scale: 0.9408126, zeroPoint: -128) == [127])
        #expect(QuantizationMath.quantizeInt8([-10_000.0], scale: 1, zeroPoint: 0) == [-128])
    }

    @Test func dequantizeInvertsTheAffineMap() {
        #expect(QuantizationMath.dequantizeInt8([-21, 127], scale: 0.5, zeroPoint: -128)
                == [53.5, 127.5])
    }

    @Test func softmaxSumsToOneAndPreservesArgmax() {
        let probs = QuantizationMath.softmax([1.0, 3.0, 0.5, 2.0, -1.0])
        #expect(abs(probs.reduce(0, +) - 1.0) < 1e-6)
        #expect(probs.firstIndex(of: probs.max()!) == 1)
    }

    @Test func softmaxIsStableForLargeLogits() {
        // Max-subtraction keeps exp() finite.
        let probs = QuantizationMath.softmax([1000.0, 1001.0])
        #expect(probs.allSatisfy { $0.isFinite })
        #expect(abs(probs.reduce(0, +) - 1.0) < 1e-6)
    }
}
