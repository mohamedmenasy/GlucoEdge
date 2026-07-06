package com.glucoedge.app.ui

import com.glucoedge.app.inference.Classifier
import com.glucoedge.app.inference.Prediction
import com.glucoedge.app.replay.Reading
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

private class FakeClassifier : Classifier {
    var calls = 0
    override val engineLabel: String = "FakeEngine"
    override fun classify(window: FloatArray): Prediction {
        calls++
        return Prediction(
            floatArrayOf(0.1f, 0.1f, 0.6f, 0.1f, 0.1f),
            2,
            1_000_000L,
            floatArrayOf(0.1f, 0.1f, 0.6f, 0.1f, 0.1f),
        )
    }
    override fun close() {}
}

class MainViewModelTest {
    private val dispatcher = StandardTestDispatcher()
    private val trace = (0L..19L).map { Reading(it * 5, 100f + it) }
    private lateinit var fake: FakeClassifier

    @Before fun setUp() {
        Dispatchers.setMain(dispatcher)
        fake = FakeClassifier()
    }
    @After fun tearDown() = Dispatchers.resetMain()

    private fun viewModel() = MainViewModel(
        traceSource = TraceSource(trace, skippedRows = 1, label = "synthetic"),
        classifierFactory = { fake },
    )

    @Test fun noPredictionUntilWindowFills() = runTest(dispatcher) {
        val vm = viewModel()
        vm.onPlayPause()                    // start playing
        advanceTimeBy(11 * 5_000L); runCurrent()
        assertNull(vm.uiState.value.prediction)
        assertEquals(0, fake.calls)
        advanceTimeBy(5_000L); runCurrent() // 12th reading
        assertNotNull(vm.uiState.value.prediction)
        assertEquals(2, vm.uiState.value.prediction!!.classIndex)
        assertEquals(1, fake.calls)
    }

    @Test fun exposesTraceMetadata() = runTest(dispatcher) {
        val vm = viewModel()
        assertEquals("synthetic", vm.uiState.value.traceLabel)
        assertEquals(1, vm.uiState.value.skippedRows)
    }

    @Test fun modelToggleSwapsClassifierAndResetsStats() = runTest(dispatcher) {
        val vm = viewModel()
        vm.onPlayPause()
        advanceTimeBy(12 * 5_000L); runCurrent()
        assertEquals(1, fake.calls)
        vm.onToggleModel()
        runCurrent()
        assertEquals(com.glucoedge.app.inference.ModelFile.INT8, vm.uiState.value.model)
        assertEquals(0, vm.uiState.value.stats.count)
    }
}
