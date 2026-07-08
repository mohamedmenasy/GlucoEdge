package com.glucoedge.app.explain

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File

interface NoteGenerator {
    val isInitialized: Boolean
    suspend fun generate(prompt: String, onToken: (String) -> Unit = {}): String
    fun close()
}

/**
 * Wraps a LiteRT-LM Engine over a user-provided .litertlm file (never bundled,
 * never downloaded - the app has no INTERNET permission). Lazy: the engine
 * initializes on first use, which takes seconds for a ~550 MB model.
 * CPU backend: no manifest additions, deterministic setup.
 */
class LitertLmExplainer(
    private val modelFile: File,
    private val cacheDirPath: String,
) : NoteGenerator {
    private var engine: Engine? = null
    override val isInitialized get() = engine != null

    private companion object {
        const val MAX_NOTE_CHARS = 700
        val SAMPLER = SamplerConfig(topK = 40, topP = 0.95, temperature = 0.2)
    }

    override suspend fun generate(prompt: String, onToken: (String) -> Unit): String {
        val eng = engine ?: Engine(
            EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(),
                cacheDir = cacheDirPath,
            )
        ).also { it.initialize(); engine = it }

        val sb = StringBuilder()
        eng.createConversation(ConversationConfig(samplerConfig = SAMPLER)).use { conversation ->
            conversation.sendMessageAsync(prompt).collect { token ->
                val t = token.toString()
                sb.append(t)
                onToken(t)
                if (sb.length >= MAX_NOTE_CHARS) return@collect // cap; flow completes naturally
            }
        }
        return sb.toString().take(MAX_NOTE_CHARS).trim()
    }

    override fun close() {
        engine?.close()
        engine = null
    }
}
