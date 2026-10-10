import Testing
@testable import GlucoEdge

struct DeltaEncodingTests {
    @Test func deltaTransformMatchesPython() {
        // Hand vectors from tests/test_deltas.py.
        let window: [Float] = (0..<12).map { 100 + Float($0 * $0) }
        let out = DeltaEncoding.encode(window)
        #expect(out.count == 12)
        for i in 0..<11 {
            #expect(out[i] == window[i + 1] - window[i])
        }
        #expect(out[11] == window[11] / 16)
    }

    @Test func anchorForCeilingGlucoseStaysInDeltaRange() {
        let out = DeltaEncoding.encode([Float](repeating: 401, count: 12))
        #expect(out[11] == Float(401.0 / 16.0))  // 25.0625
        for i in 0..<11 { #expect(out[i] == 0) }
    }
}
