import Foundation

/// Same prompt text as the Android client's PromptBuilder.kt. (Inside this
/// module the name shadows FoundationModels' own PromptBuilder result builder.)
enum PromptBuilder {
    static func build(_ ctx: PredictionContext) -> String {
        let netInt = Int(ctx.netChangeMgdl.rounded())
        let net = netInt >= 0 ? "+\(netInt)" : "\(netInt)"
        return "You are writing a short caption for a demo app screen. The app " +
            "replays recorded glucose data and a small on-device classifier " +
            "predicted the trend. Data: the current value is " +
            "\(Int(ctx.currentMgdl.rounded())) mg/dL; over the last hour the " +
            "readings ranged from \(Int(ctx.windowMin.rounded())) to " +
            "\(Int(ctx.windowMax.rounded())) mg/dL with a net change of \(net) " +
            "mg/dL; the predicted trend for the next 15 minutes is " +
            "\"\(ctx.className)\" with \(ctx.confidencePct)% confidence; this " +
            "is a \(ctx.traceLabel) recording being replayed. Write 2-3 plain " +
            "sentences describing this pattern. Do not give advice, " +
            "recommendations, or dosing. Do not address the reader as a " +
            "patient. Do not mention insulin or treatment."
    }
}
