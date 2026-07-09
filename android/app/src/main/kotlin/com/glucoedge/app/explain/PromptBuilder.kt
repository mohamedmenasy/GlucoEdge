package com.glucoedge.app.explain

import kotlin.math.roundToInt

object PromptBuilder {
    fun build(ctx: PredictionContext): String {
        val net = ctx.netChangeMgdl.roundToInt().let { if (it >= 0) "+$it" else "$it" }
        return "You are writing a short caption for a demo app screen. The app " +
            "replays recorded glucose data and a small on-device classifier " +
            "predicted the trend. Data: the current value is " +
            "${ctx.currentMgdl.roundToInt()} mg/dL; over the last hour the " +
            "readings ranged from ${ctx.windowMin.roundToInt()} to " +
            "${ctx.windowMax.roundToInt()} mg/dL with a net change of $net " +
            "mg/dL; the predicted trend for the next 15 minutes is " +
            "\"${ctx.className}\" with ${ctx.confidencePct}% confidence; this " +
            "is a ${ctx.traceLabel} recording being replayed. Write 2-3 plain " +
            "sentences describing this pattern. Do not give advice, " +
            "recommendations, or dosing. Do not address the reader as a " +
            "patient. Do not mention insulin or treatment."
    }
}
