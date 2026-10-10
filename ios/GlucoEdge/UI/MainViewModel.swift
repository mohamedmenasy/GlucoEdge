import Foundation
import Observation

struct TraceSource {
    let readings: [Reading]
    let skippedRows: Int
    let label: String
}

/// Same state shape and behavior as the Android MainViewModel: replayed
/// readings feed the gap-aware window, full windows are classified, and the
/// Explain note is generated on demand from replayed facts only.
@MainActor
@Observable
final class MainViewModel {
    private(set) var recentReadings: [Float] = []   // up to last 12 mg/dL values
    private(set) var currentMgdl: Float?
    private(set) var prediction: Prediction?
    private(set) var model: ModelFile = .float
    private(set) var playing = false
    private(set) var speed: Speed = .x1
    private(set) var stats = LatencyStats.empty
    private(set) var rebuffering = false
    private(set) var errorMessage: String?
    private(set) var engineLabel = ""
    private(set) var explainerState: ExplainerState
    let skippedRows: Int
    let traceLabel: String

    @ObservationIgnored private let engine: ReplayEngine
    @ObservationIgnored private let buffer = WindowBuffer()
    @ObservationIgnored private let meter = LatencyMeter()
    @ObservationIgnored private var classifier: Classifier?
    @ObservationIgnored private let classifierFactory: (ModelFile) throws -> Classifier
    @ObservationIgnored private let noteGenerator: NoteGenerator
    @ObservationIgnored private var replayTask: Task<Void, Never>?

    init(trace: TraceSource,
         classifierFactory: @escaping (ModelFile) throws -> Classifier,
         explainAvailability: ExplainAvailability,
         noteGenerator: NoteGenerator,
         baseInterval: Duration = .seconds(5)) {
        skippedRows = trace.skippedRows
        traceLabel = trace.label
        self.classifierFactory = classifierFactory
        self.noteGenerator = noteGenerator
        engine = ReplayEngine(readings: trace.readings, baseInterval: baseInterval)
        switch explainAvailability {
        case .available: explainerState = .ready
        case .unavailable(let reason): explainerState = .unavailable(reason)
        }

        guard !trace.readings.isEmpty else {
            errorMessage = "No valid readings in trace"
            return
        }
        loadClassifier(.float)
        let stream = engine.events
        replayTask = Task { [weak self] in
            for await reading in stream {
                self?.onReading(reading)
            }
        }
    }

    deinit { replayTask?.cancel() }

    private func loadClassifier(_ model: ModelFile) {
        do {
            let loaded = try classifierFactory(model)
            classifier = loaded
            engineLabel = loaded.engineLabel
        } catch {
            classifier = nil
            errorMessage = "Model load failed: \(error)"
        }
    }

    private func onReading(_ reading: Reading) {
        let result = buffer.add(reading)
        var newPrediction: Prediction?
        if let window = result.window, let classifier {
            do {
                newPrediction = try classifier.classify(window)
            } catch {
                errorMessage = "Inference failed: \(error)"
            }
        }
        if let p = newPrediction { meter.record(p.latencyNanos) }
        recentReadings = Array((recentReadings + [reading.mgdl]).suffix(12))
        currentMgdl = reading.mgdl
        prediction = newPrediction ?? (result.wasReset ? nil : prediction)
        rebuffering = result.window == nil
        stats = meter.stats
    }

    func onPlayPause() {
        if engine.isPlaying { engine.pause() } else { engine.play() }
        playing = engine.isPlaying
    }

    func onSpeed(_ s: Speed) {
        engine.setSpeed(s)
        speed = s
    }

    func onToggleModel() {
        model = model == .float ? .int8 : .float
        loadClassifier(model)
        meter.reset()
        stats = meter.stats
        prediction = nil
    }

    func onExplain() {
        if case .unavailable = explainerState { return }
        guard explainerState != .generating,
              let prediction, let currentMgdl,
              let low = recentReadings.min(), let high = recentReadings.max(),
              let first = recentReadings.first, let last = recentReadings.last
        else { return }
        let ctx = PredictionContext(
            currentMgdl: currentMgdl,
            windowMin: low,
            windowMax: high,
            netChangeMgdl: last - first,
            className: TrendClassifier.classNames[prediction.classIndex],
            confidencePct: Int(prediction.probabilities[prediction.classIndex] * 100),
            traceLabel: traceLabel)
        explainerState = .generating
        Task {
            do {
                explainerState = .note(try await noteGenerator.generate(PromptBuilder.build(ctx)))
            } catch let error as NoteError {
                explainerState = .error(error.message)
            } catch {
                explainerState = .error("Note generation failed: \(error.localizedDescription)")
            }
        }
    }
}
