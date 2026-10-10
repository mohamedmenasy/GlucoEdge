/// Only replayed-data facts — nothing else may enter the prompt.
struct PredictionContext {
    let currentMgdl: Float
    let windowMin: Float
    let windowMax: Float
    let netChangeMgdl: Float
    let className: String
    let confidencePct: Int
    let traceLabel: String
}
