import CryptoKit
import Foundation
import Testing
@testable import GlucoEdge

/// Task 5 (iOS re-scope): the prespecified nine-artifact device benchmark.
/// Not run in CI — on a device:
///   TEST_RUNNER_CALIB_BENCH=1 xcodebuild test ... -destination 'platform=iOS,id=<udid>' \
///     -only-testing:GlucoEdgeTests/CalibrationBenchmarkTests
/// Raw rounds are recorded as an xcresult attachment (JSON).
struct BenchmarkManifest: Decodable {
    struct Artifact: Decodable {
        struct OutputQuant: Decodable {
            let scale: Float
            let zeroPoint: Int
            enum CodingKeys: String, CodingKey { case scale; case zeroPoint = "zero_point" }
        }
        let name: String
        let sha256: String
        let bytes: Int
        let kind: String
        let outputQuant: OutputQuant?
        let rawOutputs: [[Int]]?
        let logits: [[Float]]?
        enum CodingKeys: String, CodingKey {
            case name, sha256, bytes, kind
            case outputQuant = "output_quant"
            case rawOutputs = "raw_outputs"
            case logits
        }
    }
    let windows: [[Float]]
    let warmupIndices: [Int]
    let timedIndices: [Int]
    let rounds: Int
    let orderSeed: UInt64
    let artifacts: [Artifact]
    enum CodingKeys: String, CodingKey {
        case windows
        case warmupIndices = "warmup_indices"
        case timedIndices = "timed_indices"
        case rounds
        case orderSeed = "order_seed"
        case artifacts
    }
}

/// SplitMix64: deterministic artifact order per round, independent of Foundation.
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

@MainActor
struct CalibrationBenchmarkTests {
    private func manifest() throws -> BenchmarkManifest {
        let url = try #require(Bundle(for: TestBundleLocator.self)
            .url(forResource: "benchmark_manifest", withExtension: "json",
                 subdirectory: "BenchmarkArtifacts"))
        return try JSONDecoder().decode(BenchmarkManifest.self, from: Data(contentsOf: url))
    }

    private func artifactURL(_ name: String) throws -> URL {
        try #require(Bundle(for: TestBundleLocator.self)
            .url(forResource: name, withExtension: "tflite",
                 subdirectory: "BenchmarkArtifacts"))
    }

    /// Hash + numerical parity for one artifact on every workload window.
    private func verify(_ a: BenchmarkManifest.Artifact,
                        windows: [[Float]]) throws -> TrendClassifier {
        let url = try artifactURL(a.name)
        let data = try Data(contentsOf: url)
        #expect(data.count == a.bytes, "\(a.name) size")
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(hash == a.sha256, "\(a.name) sha256 drifted from the manifest")
        let classifier = try TrendClassifier(modelURL: url, quantized: a.kind == "int8")
        for (i, w) in windows.enumerated() {
            let p = try classifier.classify(w)
            if a.kind == "int8" {
                let quant = try #require(a.outputQuant)
                let raw = try #require(a.rawOutputs)[i].map { Int8(truncatingIfNeeded: $0) }
                let expected = QuantizationMath.dequantizeInt8(
                    raw, scale: quant.scale, zeroPoint: quant.zeroPoint)
                #expect(p.logits == expected, "\(a.name) window \(i) int8 logits")
            } else {
                let expected = try #require(a.logits)[i]
                for (j, e) in expected.enumerated() {
                    #expect(abs(p.logits[j] - e) <= 1e-5, "\(a.name) window \(i) logit \(j)")
                }
            }
        }
        return classifier
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["CALIB_BENCH"] == "1"),
          .timeLimit(.minutes(30)))
    func nineArtifactPairedRounds() throws {
        let m = try manifest()
        #expect(m.artifacts.count == 9)
        let warmup = Array(m.windows[m.warmupIndices[0]..<m.warmupIndices[1]])
        let timed = Array(m.windows[m.timedIndices[0]..<m.timedIndices[1]])

        // Parity + provenance for every artifact BEFORE any timing.
        for a in m.artifacts { _ = try verify(a, windows: m.windows) }

        var results: [String: Any] = [
            "device_model": utsnameMachine(),
            "system": ProcessInfo.processInfo.operatingSystemVersionString,
            "processor_count": ProcessInfo.processInfo.processorCount,
            "is_low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "timer": "DispatchTime outer; Prediction.latencyNanos inner (quantize+run+dequantize, no softmax)",
            "rounds": m.rounds,
            "order_seed": m.orderSeed,
            "warmup_calls": warmup.count,
            "timed_calls": timed.count,
        ]
        var perArtifact: [String: [[String: Any]]] = [:]
        m.artifacts.forEach { perArtifact[$0.name] = [] }

        for round in 0..<m.rounds {
            var rng = SplitMix64(state: m.orderSeed &+ UInt64(round))
            let order = m.artifacts.shuffled(using: &rng)
            let thermal = ProcessInfo.processInfo.thermalState.rawValue
            for (position, a) in order.enumerated() {
                let initStart = DispatchTime.now()
                let classifier = try TrendClassifier(modelURL: try artifactURL(a.name),
                                                     quantized: a.kind == "int8")
                let initNanos = DispatchTime.now().uptimeNanoseconds
                    - initStart.uptimeNanoseconds
                for w in warmup { _ = try classifier.classify(w) }
                var inner = [Int64](); inner.reserveCapacity(timed.count)
                var outer = [Int64](); outer.reserveCapacity(timed.count)
                for w in timed {
                    let t0 = DispatchTime.now()
                    let p = try classifier.classify(w)
                    outer.append(Int64(DispatchTime.now().uptimeNanoseconds
                                       - t0.uptimeNanoseconds))
                    inner.append(p.latencyNanos)
                }
                perArtifact[a.name]!.append([
                    "round": round, "order_position": position,
                    "thermal_state": thermal, "init_ns": initNanos,
                    "inner_ns": inner, "outer_ns": outer,
                ])
            }
        }
        results["artifacts"] = perArtifact

        let json = try JSONSerialization.data(withJSONObject: results,
                                              options: [.sortedKeys])
        Attachment.record(json, named: "calibration_benchmark_ios.json")
        // Compact per-artifact medians in the log for a quick read.
        for a in m.artifacts {
            let all = perArtifact[a.name]!.flatMap { $0["inner_ns"] as! [Int64] }.sorted()
            let median = Double(all[all.count / 2]) / 1e6
            print("CALIB_BENCH \(a.name): n=\(all.count) median_inner=\(median)ms")
        }
    }

    private func utsnameMachine() -> String {
        var sys = utsname()
        uname(&sys)
        return withUnsafeBytes(of: &sys.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
