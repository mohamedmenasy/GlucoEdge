package com.glucoedge.app.explain

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.isActive

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
                // Bounds prompt + output tokens together (it's the kv-cache size;
                // generation auto-stops once reached) - NOT an output-only cap.
                // The prompt built by PromptBuilder runs ~150-200 tokens, so 512
                // leaves ~300-350 tokens of headroom for the note itself, which
                // MAX_NOTE_CHARS then trims further by characters. Deliberately
                // higher than the design doc's "near 128": that figure assumed
                // maxNumTokens was output-only, which would leave zero (or
                // negative) budget once the prompt is counted against it and
                // would break generation outright.
                maxNumTokens = 512,
            )
        ).also { it.initialize(); engine = it }

        val sb = StringBuilder()
        // takeWhile checks the predicate before each emission is delivered to the
        // collector, so collection - and therefore this JNI-backed generation
        // stream - stops within one token of crossing MAX_NOTE_CHARS instead of
        // running until the model decides it's done. Standard kotlinx.coroutines
        // takeWhile swallows its internal short-circuit signal so this normally
        // returns without throwing, but litertlm's Flow is backed by a native
        // callback bridge we don't control; if cancelling it ever surfaces as a
        // real CancellationException instead, that's still a successful
        // stop-at-cap (not an error) as long as *this* coroutine wasn't the one
        // being cancelled from outside - so only that case is swallowed below,
        // and genuine outer cancellation is rethrown untouched.
        try {
            eng.createConversation(ConversationConfig(samplerConfig = SAMPLER)).use { conversation ->
                conversation.sendMessageAsync(prompt)
                    .takeWhile { sb.length < MAX_NOTE_CHARS }
                    .collect { token ->
                        val t = token.toString()
                        sb.append(t)
                        onToken(t)
                    }
            }
        } catch (e: CancellationException) {
            if (currentCoroutineContext().isActive) {
                // Cap-triggered stop leaking as an exception rather than being
                // absorbed internally - treat it as the successful completion it
                // is and return what was captured so far.
            } else {
                throw e
            }
        }
        return sb.toString().take(MAX_NOTE_CHARS).trim()
    }

    override fun close() {
        engine?.close()
        engine = null
    }
}
