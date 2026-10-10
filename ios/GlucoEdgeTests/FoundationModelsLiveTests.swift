import FoundationModels
import Testing
@testable import GlucoEdge

@MainActor
struct FoundationModelsLiveTests {
    // Runs only where Apple Intelligence is available (the user's iPhone, not
    // CI simulators). Checks the real prompt survives Apple's guardrails —
    // glucose-flavored text being refused would make Explain dead on arrival.
    // (GlucoEdge.PromptBuilder: FoundationModels exports its own PromptBuilder.)
    // Device only: a simulator reports availability from the host Mac, and its
    // first generation can fail while guardrail assets load (seen locally) —
    // nondeterministic in CI. The physical iPhone is the real target anyway.
    private nonisolated static var onDeviceWithAppleIntelligence: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return SystemLanguageModel.default.isAvailable
        #endif
    }

    @Test(.enabled(if: onDeviceWithAppleIntelligence))
    func realPromptSurvivesGuardrails() async throws {
        let prompt = GlucoEdge.PromptBuilder.build(PredictionContext(
            currentMgdl: 142, windowMin: 110, windowMax: 145, netChangeMgdl: 32,
            className: "rising", confidencePct: 78, traceLabel: "synthetic"))
        let note = try await FoundationModelsNoteGenerator().generate(prompt)
        print("FM_NOTE: \(note)")
        #expect(!note.isEmpty)
    }
}
