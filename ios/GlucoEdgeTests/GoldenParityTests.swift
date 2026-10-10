import CryptoKit
import Foundation
import Testing
@testable import GlucoEdge

struct GoldenVectors: Decodable {
    struct OutputQuant: Decodable {
        let scale: Float
        let zeroPoint: Int
        enum CodingKeys: String, CodingKey { case scale; case zeroPoint = "zero_point" }
    }
    struct Vector: Decodable {
        let window: [Float]
        let floatLogits: [Float]
        let floatClass: Int
        let int8RawOutput: [Int]
        let int8Class: Int
        enum CodingKeys: String, CodingKey {
            case window
            case floatLogits = "float_logits"
            case floatClass = "float_class"
            case int8RawOutput = "int8_raw_output"
            case int8Class = "int8_class"
        }
    }
    let classNames: [String]
    let modelSha256: [String: String]
    let int8OutputQuant: OutputQuant
    let vectors: [Vector]
    enum CodingKeys: String, CodingKey {
        case classNames = "class_names"
        case modelSha256 = "model_sha256"
        case int8OutputQuant = "int8_output_quant"
        case vectors
    }
}

@MainActor
struct GoldenParityTests {
    private func goldens() throws -> GoldenVectors {
        let url = try #require(Bundle(for: TestBundleLocator.self)
            .url(forResource: "golden_vectors", withExtension: "json", subdirectory: "assets"))
        return try JSONDecoder().decode(GoldenVectors.self, from: Data(contentsOf: url))
    }

    private func modelURL(_ model: ModelFile) throws -> URL {
        try #require(Bundle.main.url(forResource: model.resourceName, withExtension: "tflite",
                                     subdirectory: "assets"))
    }

    @Test func bundledModelsMatchGoldenHashes() throws {
        let expected = try goldens().modelSha256
        for model in ModelFile.allCases {
            let data = try Data(contentsOf: try modelURL(model))
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            #expect(hash == expected["\(model.resourceName).tflite"],
                    "asset \(model.resourceName).tflite drifted from the goldens - regenerate both together")
        }
    }

    @Test func classNamesMatchGoldens() throws {
        #expect(TrendClassifier.classNames == (try goldens().classNames))
    }

    @Test func floatModelMatchesPythonLogitsAndClasses() throws {
        let goldens = try goldens()
        let classifier = try TrendClassifier(modelURL: try modelURL(.float), quantized: false)
        for (i, v) in goldens.vectors.enumerated() {
            let prediction = try classifier.classify(v.window)
            #expect(prediction.classIndex == v.floatClass, "vector \(i) class")
            #expect(prediction.logits.count == v.floatLogits.count, "vector \(i) logit count")
            for (j, expected) in v.floatLogits.enumerated() {
                #expect(abs(prediction.logits[j] - expected) <= 1e-5, "vector \(i) logit \(j)")
            }
        }
    }

    @Test func int8ModelMatchesPythonBitExactly() throws {
        let goldens = try goldens()
        let quant = goldens.int8OutputQuant
        let classifier = try TrendClassifier(modelURL: try modelURL(.int8), quantized: true)
        for (i, v) in goldens.vectors.enumerated() {
            let prediction = try classifier.classify(v.window)
            #expect(prediction.classIndex == v.int8Class, "vector \(i) int8 class")
            // Exact parity: dequantize the golden raw int8 output through the
            // same QuantizationMath the app uses; both sides compute
            // (q - zeroPoint) * scale in fp32 from the same integers, so
            // equality is exact — zero tolerance, as on Android.
            let rawBytes = v.int8RawOutput.map { Int8(truncatingIfNeeded: $0) }
            let expectedLogits = QuantizationMath.dequantizeInt8(rawBytes, scale: quant.scale,
                                                                 zeroPoint: quant.zeroPoint)
            #expect(prediction.logits == expectedLogits, "vector \(i) int8 logits")
        }
    }

    // Manual latency benchmark: not run in CI. On a device:
    //   TEST_RUNNER_LATENCY_BENCH=1 xcodebuild test ... -destination 'platform=iOS,name=<iPhone>' \
    //     -only-testing:GlucoEdgeTests/GoldenParityTests/latencyBenchmark()
    // (xcodebuild forwards TEST_RUNNER_-prefixed variables to the test process
    // with the prefix stripped.) Equal n for both models.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["LATENCY_BENCH"] == "1"))
    func latencyBenchmark() throws {
        let goldens = try goldens()
        let window = goldens.vectors[0].window
        for model in ModelFile.allCases {
            let classifier = try TrendClassifier(modelURL: try modelURL(model),
                                                 quantized: model == .int8)
            let meter = LatencyMeter(capacity: 100)
            _ = try classifier.classify(window)  // warmup, not recorded
            for _ in 0..<100 {
                meter.record(try classifier.classify(window).latencyNanos)
            }
            let s = meter.stats
            print("LATENCY_BENCH \(model.rawValue): n=\(s.count) mean=\(s.meanMs)ms " +
                  "median=\(s.medianMs)ms p95=\(s.p95Ms)ms")
        }
    }
}
