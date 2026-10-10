import Foundation
import LiteRT

enum ModelFile: String, CaseIterable {
    case float = "trend_float"
    case int8 = "trend_int8"

    var resourceName: String { rawValue }
}

struct Prediction {
    let probabilities: [Float]
    let classIndex: Int
    /// Wall time of the engine run only — softmax and class selection excluded,
    /// same scope as the Android client and the paper.
    let latencyNanos: Int64
    /// Raw model output (pre-softmax), for logit-level parity assertions.
    let logits: [Float]
}

/// Main-actor isolated: the only caller is the main-actor view model, and a
/// ~0.3 ms inference on main matches the Android client's Main dispatcher.
@MainActor
protocol Classifier {
    func classify(_ window: [Float]) throws -> Prediction
    /// Exactly "CompiledModel" — surfaced in the UI stats line.
    var engineLabel: String { get }
}

struct ClassifierError: Error, CustomStringConvertible {
    let description: String
}

private struct QuantParams {
    let scale: Float
    let zeroPoint: Int
}

/// Runs a bundled `.tflite` through LiteRT's CompiledModel API (CPU).
/// For the INT8 model, input/output quantization params are read ONCE at
/// load time from the model itself — never hardcoded — and the Swift
/// quantization replicates the corrected Python benchmark: half-even round,
/// then clip. No explicit close(): ARC releases the CompiledModel in deinit
/// (divergence from the Kotlin client, which closes explicitly).
@MainActor
final class TrendClassifier: Classifier {
    static let classNames = ["falling_fast", "falling", "stable", "rising", "rising_fast"]

    let engineLabel = "CompiledModel"

    private let model: CompiledModel
    private let inputBuffers: [TensorBuffer]
    private let outputBuffers: [TensorBuffer]
    private let inputQuant: QuantParams?
    private let outputQuant: QuantParams?

    init(modelURL: URL, quantized: Bool) throws {
        let environment = try Environment()
        model = try CompiledModel(filePath: modelURL.path, environment: environment)
        inputBuffers = try model.createInputBuffers()
        outputBuffers = try model.createOutputBuffers()
        if quantized {
            guard let inQ = try model.inputTensorQuantization(inputIndex: 0).perTensor,
                  let outQ = try model.outputTensorQuantization(outputIndex: 0).perTensor
            else {
                throw ClassifierError(description:
                    "INT8 model does not expose per-tensor quantization params")
            }
            inputQuant = QuantParams(scale: inQ.scale, zeroPoint: Int(inQ.zeroPoint))
            outputQuant = QuantParams(scale: outQ.scale, zeroPoint: Int(outQ.zeroPoint))
        } else {
            inputQuant = nil
            outputQuant = nil
        }
    }

    func classify(_ window: [Float]) throws -> Prediction {
        guard window.count == 12 else {
            throw ClassifierError(description: "expected a 12-reading window, got \(window.count)")
        }
        if let inputQuant {
            try inputBuffers[0].write(QuantizationMath.quantizeInt8(
                window, scale: inputQuant.scale, zeroPoint: inputQuant.zeroPoint))
        } else {
            try inputBuffers[0].write(window)
        }
        let start = DispatchTime.now()
        try model.run(inputs: inputBuffers, outputs: outputBuffers)
        let elapsed = Int64(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds)
        let logits: [Float]
        if let outputQuant {
            let raw: [Int8] = try outputBuffers[0].read()
            logits = QuantizationMath.dequantizeInt8(raw, scale: outputQuant.scale,
                                                     zeroPoint: outputQuant.zeroPoint)
        } else {
            logits = try outputBuffers[0].read()
        }
        let probs = QuantizationMath.softmax(logits)
        let classIndex = probs.indices.max(by: { probs[$0] < probs[$1] })!
        return Prediction(probabilities: probs, classIndex: classIndex,
                          latencyNanos: elapsed, logits: logits)
    }
}
