package com.glucoedge.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.glucoedge.app.explain.ExplainerState
import com.glucoedge.app.inference.TrendClassifier
import com.glucoedge.app.replay.Speed

private val TREND_ARROWS = listOf("↓↓", "↓", "→", "↑", "↑↑")

@Composable
fun MainScreen(viewModel: MainViewModel, deviceLabel: String) {
    val state by viewModel.uiState.collectAsState()

    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error); return@Column }

        GlucoseChart(state.recentReadings)

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("${state.currentMgdl?.let { "%.0f".format(it) } ?: "--"} mg/dL",
                 style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.width(8.dp))
            Text("(${state.traceLabel} trace)", style = MaterialTheme.typography.bodySmall)
        }

        if (state.rebuffering) Text("window rebuffering…", style = MaterialTheme.typography.bodySmall)

        state.prediction?.let { p ->
            Text("${TREND_ARROWS[p.classIndex]}  ${TrendClassifier.CLASS_NAMES[p.classIndex]}",
                 style = MaterialTheme.typography.headlineLarge)
            Text("predicted trend, next 15 min", style = MaterialTheme.typography.bodySmall)
            TrendClassifier.CLASS_NAMES.forEachIndexed { i, name ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(name, Modifier.width(96.dp), style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(
                        progress = { p.probabilities[i] },
                        modifier = Modifier.fillMaxWidth().height(8.dp),
                    )
                }
            }
        }

        Text(
            "model: ${state.model.name.lowercase()}  ·  engine: ${state.engineLabel}  ·  " +
            "inferences: ${state.stats.count}  ·  mean %.3f ms  ·  p95 %.3f ms  ·  skipped rows: %d  ·  %s"
                .format(state.stats.meanMs, state.stats.p95Ms, state.skippedRows, deviceLabel),
            style = MaterialTheme.typography.bodySmall,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::onPlayPause) { Text(if (state.playing) "Pause" else "Play") }
            Speed.entries.forEach { s ->
                OutlinedButton(onClick = { viewModel.onSpeed(s) }) {
                    Text(if (s == state.speed) "[${s.multiplier}×]" else "${s.multiplier}×")
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = viewModel::onToggleModel) {
                Text(
                    if (state.model.name == "FLOAT") "Switch to INT8 model" else "Switch to float model",
                    maxLines = 1
                )
            }
        }

        val explainer by viewModel.explainerState.collectAsState()
        when (val ex = explainer) {
            ExplainerState.Hidden -> {}
            else -> {
                OutlinedButton(
                    onClick = viewModel::onExplain,
                    enabled = ex !is ExplainerState.Generating && ex !is ExplainerState.LoadingModel,
                ) { Text("Explain", maxLines = 1) }
                when (ex) {
                    ExplainerState.LoadingModel -> Text("loading model…", style = MaterialTheme.typography.bodySmall)
                    ExplainerState.Generating -> Text("generating…", style = MaterialTheme.typography.bodySmall)
                    is ExplainerState.Note -> Column {
                        Text("On-device demo note — not medical guidance.",
                             style = MaterialTheme.typography.labelSmall)
                        Text(ex.text, style = MaterialTheme.typography.bodyMedium)
                    }
                    is ExplainerState.Error -> Text(ex.message,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall)
                    else -> {}
                }
            }
        }

        Spacer(Modifier.weight(1f))
        Text(
            "Portfolio demo on public research data — not a medical device, not treatment guidance.",
            style = MaterialTheme.typography.labelSmall,
        )
    }
}

@Composable
private fun GlucoseChart(readings: List<Float>) {
    Canvas(Modifier.fillMaxWidth().height(140.dp)) {
        if (readings.size < 2) return@Canvas
        val min = (readings.min() - 10f).coerceAtLeast(40f)
        val max = (readings.max() + 10f).coerceAtMost(410f)
        val span = (max - min).coerceAtLeast(1f)
        val stepX = size.width / 11f
        readings.zipWithNext().forEachIndexed { i, (a, b) ->
            drawLine(
                color = Color(0xFF1565C0),
                start = Offset(i * stepX, size.height * (1 - (a - min) / span)),
                end = Offset((i + 1) * stepX, size.height * (1 - (b - min) / span)),
                strokeWidth = 4f,
            )
        }
    }
}
