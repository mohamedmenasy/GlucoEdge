import Testing
@testable import GlucoEdge

struct PromptBuilderTests {
    @Test func buildsThePromptVerbatim() {
        let prompt = PromptBuilder.build(PredictionContext(
            currentMgdl: 142.4, windowMin: 110.0, windowMax: 145.0,
            netChangeMgdl: 32.4, className: "rising", confidencePct: 78,
            traceLabel: "synthetic"))
        #expect(prompt == "You are writing a short caption for a demo app screen. The app " +
            "replays recorded glucose data and a small on-device classifier " +
            "predicted the trend. Data: the current value is " +
            "142 mg/dL; over the last hour the " +
            "readings ranged from 110 to " +
            "145 mg/dL with a net change of +32 " +
            "mg/dL; the predicted trend for the next 15 minutes is " +
            "\"rising\" with 78% confidence; this " +
            "is a synthetic recording being replayed. Write 2-3 plain " +
            "sentences describing this pattern. Do not give advice, " +
            "recommendations, or dosing. Do not address the reader as a " +
            "patient. Do not mention insulin or treatment.")
    }

    @Test func negativeNetChangeHasNoPlusSign() {
        let prompt = PromptBuilder.build(PredictionContext(
            currentMgdl: 90, windowMin: 90, windowMax: 130,
            netChangeMgdl: -38.6, className: "falling", confidencePct: 61,
            traceLabel: "real"))
        #expect(prompt.contains("a net change of -39 mg/dL"))
    }
}
