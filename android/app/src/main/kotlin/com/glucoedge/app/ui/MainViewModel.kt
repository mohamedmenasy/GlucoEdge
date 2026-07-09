package com.glucoedge.app.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.glucoedge.app.explain.ExplainerState
import com.glucoedge.app.explain.LitertLmExplainer
import com.glucoedge.app.explain.NoteGenerator
import com.glucoedge.app.explain.PredictionContext
import com.glucoedge.app.explain.PromptBuilder
import com.glucoedge.app.inference.Classifier
import com.glucoedge.app.inference.LatencyMeter
import com.glucoedge.app.inference.LatencyStats
import com.glucoedge.app.inference.ModelFile
import com.glucoedge.app.inference.Prediction
import com.glucoedge.app.inference.TrendClassifier
import com.glucoedge.app.replay.Reading
import com.glucoedge.app.replay.ReplayEngine
import com.glucoedge.app.replay.Speed
import com.glucoedge.app.replay.WindowBuffer
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

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
    private val modelFileProvider: () -> java.io.File? = { null },
    private val noteGeneratorFactory: (java.io.File) -> NoteGenerator = { file ->
        LitertLmExplainer(file, cacheDirPath = "")
    },
    // Real generation runs off the main thread since it can take seconds; tests inject
    // the StandardTestDispatcher here so advanceUntilIdle() can observe completion -
    // a bare Dispatchers.Default hop is invisible to the test's virtual-time scheduler
    // and produces a race (confirmed: explainSurfacesErrorsWithoutCrashing flaked without this).
    private val explainDispatcher: CoroutineDispatcher = Dispatchers.Default,
) : ViewModel() {
    private val _uiState = MutableStateFlow(
        UiState(skippedRows = traceSource.skippedRows, traceLabel = traceSource.label)
    )
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val engine = ReplayEngine(traceSource.readings)
    private val buffer = WindowBuffer()
    private val meter = LatencyMeter()
    private var classifier: Classifier? = null

    private val _explainerState = MutableStateFlow<ExplainerState>(
        if (modelFileProvider() != null) ExplainerState.Ready else ExplainerState.Hidden
    )
    val explainerState: StateFlow<ExplainerState> = _explainerState.asStateFlow()
    private var noteGenerator: NoteGenerator? = null

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

    fun onExplain() {
        val modelFile = modelFileProvider() ?: return
        val prediction = _uiState.value.prediction ?: return
        if (_explainerState.value is ExplainerState.Generating ||
            _explainerState.value is ExplainerState.LoadingModel) return
        val readings = _uiState.value.recentReadings
        val ctx = PredictionContext(
            currentMgdl = _uiState.value.currentMgdl ?: return,
            windowMin = readings.min(),
            windowMax = readings.max(),
            netChangeMgdl = readings.last() - readings.first(),
            className = TrendClassifier.CLASS_NAMES[prediction.classIndex],
            confidencePct = (prediction.probabilities[prediction.classIndex] * 100).toInt(),
            traceLabel = _uiState.value.traceLabel,
        )
        viewModelScope.launch {
            val generator = noteGenerator ?: noteGeneratorFactory(modelFile).also { noteGenerator = it }
            _explainerState.value =
                if (generator.isInitialized) ExplainerState.Generating else ExplainerState.LoadingModel
            runCatching {
                withContext(explainDispatcher) { generator.generate(PromptBuilder.build(ctx)) }
            }.onSuccess { note ->
                _explainerState.value = ExplainerState.Note(note)
            }.onFailure { e ->
                _explainerState.value = ExplainerState.Error("Note generation failed: ${e.message}")
            }
        }
    }

    /** Re-detect the model file (spec: checked at startup AND on resume). */
    fun onResumeCheck() {
        val present = modelFileProvider() != null
        _explainerState.update { state ->
            when {
                !present -> ExplainerState.Hidden
                state is ExplainerState.Hidden -> ExplainerState.Ready
                else -> state // never clobber an in-flight generation or a shown note
            }
        }
    }

    override fun onCleared() {
        classifier?.close()
        noteGenerator?.close()
    }
}
