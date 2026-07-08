package com.glucoedge.app.explain

/** Only replayed-data facts - nothing else may enter the prompt. */
data class PredictionContext(
    val currentMgdl: Float,
    val windowMin: Float,
    val windowMax: Float,
    val netChangeMgdl: Float,
    val className: String,
    val confidencePct: Int,
    val traceLabel: String,
)
