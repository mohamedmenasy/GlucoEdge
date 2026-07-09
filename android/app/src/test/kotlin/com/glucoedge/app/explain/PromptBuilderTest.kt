package com.glucoedge.app.explain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class PromptBuilderTest {
    private val ctx = PredictionContext(
        currentMgdl = 148f, windowMin = 112f, windowMax = 150f,
        netChangeMgdl = 34f, className = "rising", confidencePct = 62,
        traceLabel = "synthetic",
    )

    @Test fun rendersExactTemplate() {
        assertEquals(
            "You are writing a short caption for a demo app screen. The app " +
            "replays recorded glucose data and a small on-device classifier " +
            "predicted the trend. Data: the current value is 148 mg/dL; over " +
            "the last hour the readings ranged from 112 to 150 mg/dL with a " +
            "net change of +34 mg/dL; the predicted trend for the next 15 " +
            "minutes is \"rising\" with 62% confidence; this is a synthetic " +
            "recording being replayed. Write 2-3 plain sentences describing " +
            "this pattern. Do not give advice, recommendations, or dosing. " +
            "Do not address the reader as a patient. Do not mention insulin " +
            "or treatment.",
            PromptBuilder.build(ctx),
        )
    }

    @Test fun negativeNetChangeGetsMinusSign() {
        val prompt = PromptBuilder.build(ctx.copy(netChangeMgdl = -21f))
        assert(prompt.contains("net change of -21 mg/dL"))
        assertFalse(prompt.contains("+-"))
    }
}
