import Testing
@testable import GlucoEdge

@MainActor
private final class FakeClassifier: Classifier {
    let engineLabel: String
    var calls = 0
    var shouldThrow = false

    init(label: String = "Fake") { engineLabel = label }

    func classify(_ window: [Float]) throws -> Prediction {
        calls += 1
        if shouldThrow { throw ClassifierError(description: "boom") }
        return Prediction(probabilities: [0.1, 0.1, 0.6, 0.1, 0.1], classIndex: 2,
                          latencyNanos: 1_000_000, logits: [0, 0, 1, 0, 0])
    }
}

@MainActor
private final class FakeNoteGenerator: NoteGenerator {
    var started = 0
    var finish: CheckedContinuation<String, any Error>?

    func generate(_ prompt: String) async throws -> String {
        started += 1
        return try await withCheckedThrowingContinuation { finish = $0 }
    }
}

@MainActor
struct MainViewModelTests {
    private func readings(_ n: Int) -> [Reading] {
        (0..<n).map { Reading(epochMinutes: Int64($0) * 5, mgdl: 100 + Float($0)) }
    }

    private func makeVM(classifier: FakeClassifier = FakeClassifier(),
                        note: FakeNoteGenerator = FakeNoteGenerator(),
                        readingCount: Int = 14) -> MainViewModel {
        MainViewModel(
            trace: TraceSource(readings: readings(readingCount), skippedRows: 1, label: "synthetic"),
            classifierFactory: { _ in classifier },
            explainAvailability: .available,
            noteGenerator: note,
            baseInterval: .milliseconds(1))
    }

    /// Polls `condition` every millisecond for up to ~2 s, then asserts it.
    private func waitUntil(_ condition: () -> Bool) async {
        for _ in 0..<2000 where !condition() {
            try? await Task.sleep(for: .milliseconds(1))
        }
        #expect(condition())
    }

    @Test func classifiesOnceWindowFills() async {
        let classifier = FakeClassifier()
        let vm = makeVM(classifier: classifier)
        vm.onPlayPause()
        // 14 readings, window 12 → readings 12, 13, 14 produce 3 predictions
        await waitUntil { vm.currentMgdl == 113 }
        #expect(classifier.calls == 3)
        #expect(vm.stats.count == 3)
        #expect(vm.prediction?.classIndex == 2)
        #expect(vm.rebuffering == false)
        #expect(vm.engineLabel == "Fake")
        #expect(vm.recentReadings.count == 12)
        #expect(vm.skippedRows == 1)
    }

    @Test func rebufferingShownUntilWindowFull() async {
        let vm = makeVM(readingCount: 5)   // never fills a 12-window
        vm.onPlayPause()
        await waitUntil { vm.currentMgdl == 104 }
        #expect(vm.prediction == nil)
        #expect(vm.rebuffering == true)
    }

    @Test func toggleSwapsClassifierResetsStatsAndKeepsReplaying() async {
        // Review Focus 5: after a mid-replay toggle, later readings must be
        // classified by the NEW model, stats/prediction reset, replay continues.
        let floatFake = FakeClassifier(label: "FloatFake")
        let int8Fake = FakeClassifier(label: "Int8Fake")
        let vm = MainViewModel(
            trace: TraceSource(readings: readings(30), skippedRows: 0, label: "synthetic"),
            classifierFactory: { $0 == .float ? floatFake : int8Fake },
            explainAvailability: .available,
            noteGenerator: FakeNoteGenerator(),
            baseInterval: .milliseconds(5))
        vm.onPlayPause()
        await waitUntil { vm.prediction != nil }
        vm.onToggleModel()
        #expect(vm.model == .int8)
        #expect(vm.engineLabel == "Int8Fake")
        #expect(vm.prediction == nil)
        #expect(vm.stats.count == 0)
        #expect(vm.playing == true)
        let floatCallsAtToggle = floatFake.calls
        await waitUntil { vm.currentMgdl == 129 }
        #expect(int8Fake.calls >= 1)
        #expect(floatFake.calls == floatCallsAtToggle)
        #expect(vm.stats.count == int8Fake.calls)
    }

    @Test func inferenceFailureSurfacesErrorAndReplayContinues() async {
        let classifier = FakeClassifier()
        classifier.shouldThrow = true
        let vm = makeVM(classifier: classifier)
        vm.onPlayPause()
        await waitUntil { vm.errorMessage != nil }
        #expect(vm.errorMessage?.contains("Inference failed") == true)
        await waitUntil { vm.currentMgdl == 113 }   // replay kept going
        #expect(vm.prediction == nil)
    }

    @Test func emptyTraceIsAnError() {
        let vm = MainViewModel(
            trace: TraceSource(readings: [], skippedRows: 0, label: "synthetic"),
            classifierFactory: { _ in FakeClassifier() },
            explainAvailability: .available,
            noteGenerator: FakeNoteGenerator())
        #expect(vm.errorMessage == "No valid readings in trace")
    }

    @Test func explainRunsOnceAndSecondTapIsIgnoredWhileGenerating() async {
        // Review Focus 4: double-tap must not start a second generation.
        let note = FakeNoteGenerator()
        let vm = makeVM(note: note)
        vm.onPlayPause()
        await waitUntil { vm.prediction != nil }
        vm.onExplain()
        #expect(vm.explainerState == .generating)
        vm.onExplain()     // ignored while generating
        await waitUntil { note.started == 1 }
        try? await Task.sleep(for: .milliseconds(20))
        #expect(note.started == 1)
        note.finish?.resume(returning: "A short note.")
        await waitUntil { vm.explainerState == .note("A short note.") }
    }

    @Test func explainErrorBecomesReadableState() async {
        let note = FakeNoteGenerator()
        let vm = makeVM(note: note)
        vm.onPlayPause()
        await waitUntil { vm.prediction != nil }
        vm.onExplain()
        await waitUntil { note.started == 1 }
        note.finish?.resume(throwing: NoteError(message: "Apple's safety filter declined this note."))
        await waitUntil { vm.explainerState == .error("Apple's safety filter declined this note.") }
    }

    @Test func explainWithoutPredictionDoesNothing() {
        let note = FakeNoteGenerator()
        let vm = makeVM(note: note)   // never played: no prediction yet
        vm.onExplain()
        #expect(vm.explainerState == .ready)
        #expect(note.started == 0)
    }

    @Test func availabilityRefreshEnablesExplainWithoutClobberingANote() async {
        // Mirrors Android's onResumeCheck: enabling Apple Intelligence in
        // Settings and returning to the app must surface Explain; a shown
        // note must survive a refresh; losing availability shows the reason.
        let note = FakeNoteGenerator()
        let vm = MainViewModel(
            trace: TraceSource(readings: readings(14), skippedRows: 0, label: "synthetic"),
            classifierFactory: { _ in FakeClassifier() },
            explainAvailability: .unavailable("off"),
            noteGenerator: note,
            baseInterval: .milliseconds(1))
        vm.refreshExplainAvailability(.available)
        #expect(vm.explainerState == .ready)

        vm.onPlayPause()
        await waitUntil { vm.prediction != nil }
        vm.onExplain()
        await waitUntil { note.started == 1 }
        vm.refreshExplainAvailability(.available)
        #expect(vm.explainerState == .generating)   // in-flight generation untouched
        note.finish?.resume(returning: "Kept.")
        await waitUntil { vm.explainerState == .note("Kept.") }
        vm.refreshExplainAvailability(.available)
        #expect(vm.explainerState == .note("Kept."))

        vm.refreshExplainAvailability(.unavailable("Explain needs Apple Intelligence — enable it in Settings."))
        #expect(vm.explainerState == .unavailable("Explain needs Apple Intelligence — enable it in Settings."))
    }

    @Test func unavailableExplainStateCarriesReason() {
        let vm = MainViewModel(
            trace: TraceSource(readings: readings(14), skippedRows: 0, label: "synthetic"),
            classifierFactory: { _ in FakeClassifier() },
            explainAvailability: .unavailable("Explain needs Apple Intelligence — enable it in Settings."),
            noteGenerator: FakeNoteGenerator())
        #expect(vm.explainerState == .unavailable("Explain needs Apple Intelligence — enable it in Settings."))
        vm.onExplain()   // must stay unavailable
        #expect(vm.explainerState == .unavailable("Explain needs Apple Intelligence — enable it in Settings."))
    }
}
