package com.glucoedge.app.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.glucoedge.app.inference.Classifier
import com.glucoedge.app.inference.LatencyMeter
import com.glucoedge.app.inference.LatencyStats
import com.glucoedge.app.inference.ModelFile
import com.glucoedge.app.inference.Prediction
import com.glucoedge.app.replay.Reading
import com.glucoedge.app.replay.ReplayEngine
import com.glucoedge.app.replay.Speed
import com.glucoedge.app.replay.WindowBuffer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class TraceSource(val readings: List<Reading>, val skippedRows: Int, val label: String)

data class UiState(
    val recentReadings: List<Float> = emptyList(),   // up to last 12 mgdl values
    val currentMgdl: Float? = null,
    val prediction: Prediction? = null,
    val model: ModelFile = ModelFile.FLOAT,
    val playing: Boolean = false,
    val speed: Speed = Speed.X1,
    val stats: LatencyStats = LatencyStats(0, 0.0, 0.0, 0.0),
    val skippedRows: Int = 0,
    val rebuffering: Boolean = false,
    val traceLabel: String = "",
    val error: String? = null,
    val engineLabel: String = "",
)

class MainViewModel(
    private val traceSource: TraceSource,
    private val classifierFactory: (ModelFile) -> Classifier,
) : ViewModel() {
    private val _uiState = MutableStateFlow(
        UiState(skippedRows = traceSource.skippedRows, traceLabel = traceSource.label)
    )
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val engine = ReplayEngine(traceSource.readings)
    private val buffer = WindowBuffer()
    private val meter = LatencyMeter()
    private var classifier: Classifier? = null

    init {
        if (traceSource.readings.isEmpty()) {
            _uiState.update { it.copy(error = "No valid readings in trace") }
        } else {
            classifier = classifierFactory(ModelFile.FLOAT)
            _uiState.update { it.copy(engineLabel = classifier?.engineLabel ?: "") }
            viewModelScope.launch {
                engine.events.collect { reading -> onReading(reading) }
            }
        }
    }

    private fun onReading(reading: Reading) {
        val result = buffer.add(reading)
        val prediction = result.window?.let { window ->
            runCatching { classifier?.classify(window) }
                .onFailure { e -> _uiState.update { s -> s.copy(error = "Inference failed: ${e.message}") } }
                .getOrNull()
        }
        prediction?.let { meter.record(it.latencyNanos) }
        _uiState.update { s ->
            s.copy(
                recentReadings = (s.recentReadings + reading.mgdl).takeLast(12),
                currentMgdl = reading.mgdl,
                prediction = prediction ?: if (result.wasReset) null else s.prediction,
                rebuffering = result.window == null,
                stats = meter.stats,
            )
        }
    }

    fun onPlayPause() {
        if (engine.isPlaying.value) engine.pause() else engine.play()
        _uiState.update { it.copy(playing = engine.isPlaying.value) }
    }

    fun onSpeed(speed: Speed) {
        engine.setSpeed(speed)
        _uiState.update { it.copy(speed = speed) }
    }

    fun onToggleModel() {
        val next = if (_uiState.value.model == ModelFile.FLOAT) ModelFile.INT8 else ModelFile.FLOAT
        classifier?.close()
        classifier = classifierFactory(next)
        meter.reset()
        _uiState.update {
            it.copy(model = next, stats = meter.stats, prediction = null, engineLabel = classifier?.engineLabel ?: "")
        }
    }

    override fun onCleared() {
        classifier?.close()
    }
}
