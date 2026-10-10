import FoundationModels
import Foundation
import Testing
@testable import GlucoEdge

@MainActor
struct FoundationModelsLiveTests {
    // Live generation runs on physical devices with Apple Intelligence. A
    // simulator reports the host Mac's availability and its first generation
    // can fail while guardrail assets load — nondeterministic, so CI skips it;
    // opt in locally with TEST_RUNNER_FM_LIVE=1.
    private nonisolated static var liveEnabled: Bool {
        guard SystemLanguageModel.default.isAvailable else { return false }
        #if targetEnvironment(simulator)
        return ProcessInfo.processInfo.environment["FM_LIVE"] == "1"
        #else
        return true
        #endif
    }

    // (GlucoEdge.PromptBuilder: FoundationModels exports its own PromptBuilder.)
    private let prompt = GlucoEdge.PromptBuilder.build(PredictionContext(
        currentMgdl: 142, windowMin: 110, windowMax: 145, netChangeMgdl: 32,
        className: "rising", confidencePct: 78, traceLabel: "synthetic"))

    // Checks the real prompt survives Apple's guardrails — glucose-flavored
    // text being refused would make Explain dead on arrival.
    @Test(.enabled(if: liveEnabled))
    func realPromptSurvivesGuardrails() async throws {
        let note = try await FoundationModelsNoteGenerator().generate(prompt)
        print("FM_NOTE: \(note)")
        #expect(!note.isEmpty)
    }

    // Each note must be independent, like Android's per-note conversation:
    // a reused session carries every earlier prompt and note into the next
    // one (breaking "nothing else may enter the prompt") and eventually
    // exhausts the on-device model's context window.
    @Test(.enabled(if: liveEnabled))
    func manyNotesInARowStayIndependent() async throws {
        let generator = FoundationModelsNoteGenerator()
        for i in 0..<25 {
            do {
                _ = try await generator.generate(prompt)
            } catch {
                Issue.record("note \(i) failed: \(error)")
                return
            }
        }
    }
}
