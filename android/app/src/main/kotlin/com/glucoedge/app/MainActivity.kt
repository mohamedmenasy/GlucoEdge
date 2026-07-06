package com.glucoedge.app

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.glucoedge.app.inference.TrendClassifier
import com.glucoedge.app.replay.CsvTraceLoader
import com.glucoedge.app.ui.GlucoEdgeTheme
import com.glucoedge.app.ui.MainScreen
import com.glucoedge.app.ui.MainViewModel
import com.glucoedge.app.ui.TraceSource
import java.io.InputStreamReader

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                val (stream, label) = runCatching {
                    assets.open("real_trace.csv") to "real"
                }.getOrElse { assets.open("synthetic_trace.csv") to "synthetic" }
                val loaded = stream.use { CsvTraceLoader.load(InputStreamReader(it)) }
                return MainViewModel(
                    traceSource = TraceSource(loaded.readings, loaded.skippedRows, label),
                    classifierFactory = { model -> TrendClassifier.create(applicationContext, model) },
                ) as T
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val deviceLabel = if (Build.FINGERPRINT.contains("generic") ||
            Build.MODEL.contains("sdk_gphone") || Build.HARDWARE == "ranchu"
        ) "emulator" else "device"
        setContent { GlucoEdgeTheme { MainScreen(viewModel, deviceLabel) } }
    }
}
