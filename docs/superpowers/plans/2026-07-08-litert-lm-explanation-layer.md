# LiteRT-LM Explanation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An on-demand, fully local Gemma-generated note describing the current trend prediction, per `docs/superpowers/specs/2026-07-08-litert-lm-explanation-layer-design.md`.

**Architecture:** New `explain/` package in the existing app: pure, unit-tested core (model locator, prompt builder, state machine) + a thin `LitertLmExplainer` wrapping the LiteRT-LM Kotlin `Engine`, wired into `MainViewModel` behind a factory so JVM tests use fakes. Feature is invisible unless a `*.litertlm` file exists in the app's external-files dir. Task 3 is a human-in-the-loop device verification (the model download requires accepting the Gemma license, so the human provides the file).

**Tech Stack:** `com.google.ai.edge.litertlm:litertlm-android` (Kotlin API: `Engine`, `EngineConfig`, `Backend.CPU()`, `createConversation`, `sendMessageAsync(...).collect`), Gemma 3 1B int4 `.litertlm`, existing Compose/ViewModel stack.

## Global Constraints

- **No INTERNET permission** — unchanged; the existing `checkNoInternetPermission` gate must stay green.
- **No model weights, no GlucoBench data committed.**
- The demo caption is **static UI**, verbatim: `On-device demo note — not medical guidance.` The model's output is never trusted to carry it.
- No string may imply clinical validity or treatment guidance; the prompt template explicitly forbids advice/dosing and must not be weakened.
- The core app must remain fully functional with no model present (feature `Hidden`).
- No LLM in CI, ever: CI runs only JVM unit tests + the existing gates.
- Environment: JAVA_HOME=/opt/homebrew/opt/openjdk@17; gradle from `android/`; AVD `Medium_Phone_API_35`; physical device (Galaxy S22 Ultra) required for Task 3 only.
- Version discipline: pin the newest stable `litertlm-android` resolvable from Google's Maven at implementation time in `libs.versions.toml`; record the chosen version in the task report (established substitution pattern).

---

### Task 1: Pure core — ModelLocator, PredictionContext, PromptBuilder, ExplainerState (TDD)

**Files:**
- Create: `android/app/src/main/kotlin/com/glucoedge/app/explain/ExplainerState.kt`, `android/app/src/main/kotlin/com/glucoedge/app/explain/PredictionContext.kt`, `android/app/src/main/kotlin/com/glucoedge/app/explain/PromptBuilder.kt`, `android/app/src/main/kotlin/com/glucoedge/app/explain/ModelLocator.kt`
- Modify: `android/gradle/libs.versions.toml`, `android/app/build.gradle.kts` (add the litertlm dependency)
- Test: `android/app/src/test/kotlin/com/glucoedge/app/explain/PromptBuilderTest.kt`, `android/app/src/test/kotlin/com/glucoedge/app/explain/ModelLocatorTest.kt`

**Interfaces:**
- Consumes: nothing app-internal.
- Produces (Tasks 2–3 rely on these exact shapes):
```kotlin
sealed interface ExplainerState {
    data object Hidden : ExplainerState
    data object Ready : ExplainerState
    data object LoadingModel : ExplainerState
    data object Generating : ExplainerState
    data class Note(val text: String) : ExplainerState
    data class Error(val message: String) : ExplainerState
}
data class PredictionContext(
    val currentMgdl: Float, val windowMin: Float, val windowMax: Float,
    val netChangeMgdl: Float, val className: String,
    val confidencePct: Int, val traceLabel: String,
)
object PromptBuilder { fun build(ctx: PredictionContext): String }
object ModelLocator { fun findModel(dir: java.io.File?): java.io.File? }
```

- [ ] **Step 1: Resolve and pin the litertlm-android version**

```bash
curl -s https://dl.google.com/android/maven2/com/google/ai/edge/litertlm/litertlm-android/maven-metadata.xml | grep -o "<latest>[^<]*</latest>"
```
Pin that version in `android/gradle/libs.versions.toml`:
```toml
# under [versions]
litertlm = "<resolved version>"
# under [libraries]
litertlm-android = { module = "com.google.ai.edge.litertlm:litertlm-android", version.ref = "litertlm" }
```
and in `android/app/build.gradle.kts` dependencies: `implementation(libs.litertlm.android)`. If the metadata URL 404s, resolve via `https://maven.google.com/web/index.html#com.google.ai.edge.litertlm` semantics: add the dependency with a plausible version and let Gradle's error list available versions; pin the newest stable. Record the version in the report.

- [ ] **Step 2: Write the failing tests**

`android/app/src/test/kotlin/com/glucoedge/app/explain/PromptBuilderTest.kt`:
```kotlin
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
```

`android/app/src/test/kotlin/com/glucoedge/app/explain/ModelLocatorTest.kt`:
```kotlin
package com.glucoedge.app.explain

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelLocatorTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun nullDirYieldsNull() = assertNull(ModelLocator.findModel(null))

    @Test fun emptyDirYieldsNull() {
        assertNull(ModelLocator.findModel(tmp.root))
    }

    @Test fun ignoresOtherExtensions() {
        tmp.newFile("model.tflite"); tmp.newFile("notes.txt")
        assertNull(ModelLocator.findModel(tmp.root))
    }

    @Test fun findsLitertlmFile() {
        val f = tmp.newFile("gemma3-1b-it-int4.litertlm")
        assertEquals(f, ModelLocator.findModel(tmp.root))
    }

    @Test fun picksFirstAlphabeticallyWhenSeveral() {
        tmp.newFile("b.litertlm")
        val a = tmp.newFile("a.litertlm")
        assertEquals(a, ModelLocator.findModel(tmp.root))
    }
}
```

- [ ] **Step 3: Run to verify they fail**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge/android
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest --tests "com.glucoedge.app.explain.*"
```
Expected: compilation FAILS (unresolved `PredictionContext`/`PromptBuilder`/`ModelLocator`).

- [ ] **Step 4: Implement**

`ExplainerState.kt`:
```kotlin
package com.glucoedge.app.explain

/** UI state of the explanation feature. Hidden means: no model file on device. */
sealed interface ExplainerState {
    data object Hidden : ExplainerState
    data object Ready : ExplainerState
    data object LoadingModel : ExplainerState
    data object Generating : ExplainerState
    data class Note(val text: String) : ExplainerState
    data class Error(val message: String) : ExplainerState
}
```

`PredictionContext.kt`:
```kotlin
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
```

`PromptBuilder.kt`:
```kotlin
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
```

`ModelLocator.kt`:
```kotlin
package com.glucoedge.app.explain

import java.io.File

/** Any *.litertlm in the app's external-files dir enables the feature. */
object ModelLocator {
    fun findModel(dir: File?): File? =
        dir?.listFiles { f -> f.isFile && f.name.endsWith(".litertlm") }
            ?.minByOrNull { it.name }
}
```

- [ ] **Step 5: Tests pass, dependency resolves, guard still green**

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest :app:checkNoInternetPermission :app:assembleDebug
```
Expected: all 23 pre-existing + 7 new tests pass; the guard passes (litertlm-android must not merge INTERNET — if it does, STOP and report: that finding changes the design). Record the debug APK size (`ls -la app/build/outputs/apk/debug/app-debug.apk`) in the report — this is the "after" for the size-delta measurement; the "before" is the current main build (~30 MB per the Task-10 review of the android-app branch; re-measure from a clean pre-change build if convenient).

- [ ] **Step 6: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android/gradle/libs.versions.toml android/app/build.gradle.kts android/app/src/main/kotlin/com/glucoedge/app/explain android/app/src/test/kotlin/com/glucoedge/app/explain
git commit -m "Add explanation-layer core: model gating, prompt template, state machine"
```

---

### Task 2: LitertLmExplainer + ViewModel wiring + UI card

**Files:**
- Create: `android/app/src/main/kotlin/com/glucoedge/app/explain/LitertLmExplainer.kt`
- Modify: `android/app/src/main/kotlin/com/glucoedge/app/ui/MainViewModel.kt`, `android/app/src/main/kotlin/com/glucoedge/app/ui/MainScreen.kt`, `android/app/src/main/kotlin/com/glucoedge/app/MainActivity.kt`
- Test: `android/app/src/test/kotlin/com/glucoedge/app/ui/MainViewModelTest.kt` (add cases)

**Interfaces:**
- Consumes: Task 1's `ExplainerState`, `PredictionContext`, `PromptBuilder`, `ModelLocator`; existing `UiState`, `Prediction`, `TrendClassifier.CLASS_NAMES`.
- Produces:
```kotlin
interface NoteGenerator {
    val isInitialized: Boolean
    /** Streams tokens; returns the full text. Throws on engine failure. */
    suspend fun generate(prompt: String, onToken: (String) -> Unit = {}): String
    fun close()
}
class LitertLmExplainer(modelFile: java.io.File, cacheDirPath: String) : NoteGenerator
// MainViewModel additions:
//   constructor params: modelFileProvider: () -> java.io.File? = { null },
//                       noteGeneratorFactory: (java.io.File) -> NoteGenerator = { LitertLmExplainer(it, "") }
//   val explainerState: StateFlow<ExplainerState>
//   fun onExplain()
```

- [ ] **Step 1: Write the failing ViewModel tests (append to MainViewModelTest.kt)**

```kotlin
private class FakeGenerator(
    private val result: String = "A calm, factual note.",
    private val fail: Boolean = false,
) : NoteGenerator {
    override var isInitialized = false
    var prompts = mutableListOf<String>()
    override suspend fun generate(prompt: String, onToken: (String) -> Unit): String {
        isInitialized = true
        prompts.add(prompt)
        if (fail) throw RuntimeException("engine broke")
        onToken(result)
        return result
    }
    override fun close() {}
}

@Test fun explainerHiddenWithoutModelFile() = runTest(dispatcher) {
    val vm = viewModel() // default modelFileProvider returns null
    assertEquals(ExplainerState.Hidden, vm.explainerState.value)
}

@Test fun explainerReadyWithModelAndGeneratesNote() = runTest(dispatcher) {
    val gen = FakeGenerator()
    val vm = MainViewModel(
        traceSource = TraceSource(trace, 0, "synthetic"),
        classifierFactory = { fake },
        modelFileProvider = { java.io.File("/fake/model.litertlm") },
        noteGeneratorFactory = { gen },
    )
    assertEquals(ExplainerState.Ready, vm.explainerState.value)
    vm.onPlayPause()
    advanceTimeBy(12 * 5_000L); runCurrent()   // prediction exists now
    vm.onExplain()
    advanceUntilIdle()
    assertEquals(ExplainerState.Note("A calm, factual note."), vm.explainerState.value)
    // Prompt was built from replayed-data facts via PromptBuilder:
    assert(gen.prompts.single().contains("predicted trend for the next 15 minutes is \"stable\""))
    assert(gen.prompts.single().contains("Do not give advice"))
}

@Test fun explainSurfacesErrorsWithoutCrashing() = runTest(dispatcher) {
    val vm = MainViewModel(
        traceSource = TraceSource(trace, 0, "synthetic"),
        classifierFactory = { fake },
        modelFileProvider = { java.io.File("/fake/model.litertlm") },
        noteGeneratorFactory = { FakeGenerator(fail = true) },
    )
    vm.onPlayPause()
    advanceTimeBy(12 * 5_000L); runCurrent()
    vm.onExplain()
    advanceUntilIdle()
    assert(vm.explainerState.value is ExplainerState.Error)
}

@Test fun explainDoesNothingWithoutPrediction() = runTest(dispatcher) {
    val vm = MainViewModel(
        traceSource = TraceSource(trace, 0, "synthetic"),
        classifierFactory = { fake },
        modelFileProvider = { java.io.File("/fake/model.litertlm") },
        noteGeneratorFactory = { FakeGenerator() },
    )
    vm.onExplain()
    advanceUntilIdle()
    assertEquals(ExplainerState.Ready, vm.explainerState.value)
}
```
(Needed imports: `com.glucoedge.app.explain.ExplainerState`, `com.glucoedge.app.explain.NoteGenerator`, `kotlinx.coroutines.test.advanceUntilIdle`.)

- [ ] **Step 2: Run to verify they fail** — same gradle test command; expected: compilation FAILS (no `NoteGenerator`, no `explainerState`).

- [ ] **Step 3: Implement LitertLmExplainer**

`android/app/src/main/kotlin/com/glucoedge/app/explain/LitertLmExplainer.kt`:
```kotlin
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
```
**API decision ladder** (resolve while compiling; document the rung in the report): (1) the code above as written; (2) if `sendMessageAsync(String)` needs a `Message`/`Contents` wrapper in the pinned version, adapt per its docs; (3) if `EngineConfig`/`ConversationConfig` expose a max-tokens parameter, ALSO set it near 128 — the char cap stays either way; (4) if `cacheDir` isn't a config field, drop it. Do not change: CPU backend, sampler values, the char cap, lazy init.

- [ ] **Step 4: Wire the ViewModel**

In `MainViewModel.kt` — add constructor params (after `classifierFactory`):
```kotlin
    private val modelFileProvider: () -> java.io.File? = { null },
    private val noteGeneratorFactory: (java.io.File) -> NoteGenerator = { file ->
        LitertLmExplainer(file, cacheDirPath = "")
    },
```
add state + logic:
```kotlin
    private val _explainerState = MutableStateFlow<ExplainerState>(
        if (modelFileProvider() != null) ExplainerState.Ready else ExplainerState.Hidden
    )
    val explainerState: StateFlow<ExplainerState> = _explainerState.asStateFlow()
    private var noteGenerator: NoteGenerator? = null

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
                withContext(Dispatchers.Default) { generator.generate(PromptBuilder.build(ctx)) }
            }.onSuccess { note ->
                _explainerState.value = ExplainerState.Note(note)
            }.onFailure { e ->
                _explainerState.value = ExplainerState.Error("Note generation failed: ${e.message}")
            }
        }
    }
```
(imports: `com.glucoedge.app.explain.*`, `kotlinx.coroutines.Dispatchers`, `kotlinx.coroutines.withContext`; extend `onCleared` with `noteGenerator?.close()`). Note for the test dispatcher: `Dispatchers.Default` inside `withContext` — in tests, `runTest` handles this; if the test hangs, inject a `CoroutineDispatcher = Dispatchers.Default` constructor param instead and pass the test dispatcher (document which was needed).

- [ ] **Step 5: UI card in MainScreen.kt**

After the model-toggle Row, add:
```kotlin
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
```
(collectAsState + getValue already imported; the caption string must be byte-exact.) In `MainActivity.kt`'s factory, pass the real providers:
```kotlin
                    modelFileProvider = { com.glucoedge.app.explain.ModelLocator.findModel(getExternalFilesDir(null)) },
                    noteGeneratorFactory = { file ->
                        com.glucoedge.app.explain.LitertLmExplainer(file, cacheDirPath = cacheDir.path)
                    },
```

- [ ] **Step 6: Full JVM suite + guard + emulator gating proof**

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest :app:checkNoInternetPermission :app:assembleDebug
```
Expected: 23 + 7 + 4 = 34 tests pass, guard green. Then install on the running emulator (no model file there) and screenshot: the UI must look **identical to before** — no Explain button (gating proof). Save screenshot to `.superpowers/sdd/` or the scratchpad and reference it in the report.

- [ ] **Step 7: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android/app/src/main/kotlin/com/glucoedge/app android/app/src/test/kotlin/com/glucoedge/app/ui
git commit -m "Wire on-demand LiteRT-LM explanation into ViewModel and screen"
```

---

### Task 3: Device verification (human-in-the-loop) + README

**Files:**
- Modify: `README.md` (stretch-goal roadmap item + setup subsection in the Android app section)

**Interfaces:**
- Consumes: everything above; requires the human to supply the model file (Gemma license acceptance) and the S22 Ultra.

- [ ] **Step 1: Ask the human for the model file**

The Gemma 3 1B int4 `.litertlm` must be downloaded by the human (HuggingFace `litert-community/Gemma3-1B-IT`, the int4 `.litertlm` artifact; requires accepting the Gemma license). STOP and ask; when they provide a local path, push it:
```bash
~/Library/Android/sdk/platform-tools/adb push <local-path>.litertlm /sdcard/Android/data/com.glucoedge.app/files/
```

- [ ] **Step 2: On-device checks (S22 Ultra)**

Install the Task-2 APK, launch, play to a prediction, then verify in order, with screenshots: (a) the Explain button appears; (b) first tap shows "loading model…" then a note renders under the byte-exact caption; (c) generation time observed and noted; (d) **airplane mode on** → another Explain tap still generates (the money shot — screenshot it); (e) the note text contains no advice/dosing language (read it; if it does, tighten the prompt template's prohibition sentences and re-run Task 1's template test — do not weaken the caption).

- [ ] **Step 3: Measure and record the APK size delta**

Compare the Task-2 `app-debug.apk` size against pre-feature main (`git stash`-free approach: `git checkout main -- <nothing needed>` — simply note the size recorded in Task 1's report vs current). Record both numbers in the report and README.

- [ ] **Step 4: README update**

Replace the stretch-goal paragraph at the end of the Roadmap section with a struck-through done item pointing to a new short subsection under the Android app section (place after the parity bullet list):
```markdown
### Optional: on-device explanation notes (LiteRT-LM)

With a Gemma 3 1B model pushed to the device, the app can generate a short
plain-language note about the current prediction — fully offline (the app
has no INTERNET permission), on-demand, and labeled on-screen as
"On-device demo note — not medical guidance." Without the model file the
feature is invisible and nothing else changes (CI and emulator builds are
unaffected; no weights live in this repo).

Setup (once): download the int4 `.litertlm` of
[Gemma 3 1B](https://huggingface.co/litert-community/Gemma3-1B-IT) on your
computer (Gemma license acceptance required), then:

    adb push <file>.litertlm /sdcard/Android/data/com.glucoedge.app/files/

Adding the LiteRT-LM runtime grew the debug APK from <before> to <after>
MB; the model file itself (~550 MB) never enters the repository.
```
Fill `<before>`/`<after>` with the Step-3 numbers. Keep the roadmap strike-through consistent with the other done items.

- [ ] **Step 5: Final sweep + commit**

```bash
cd android && JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest :app:checkNoInternetPermission
cd .. && git add README.md && git commit -m "Verify explanation layer on device; document setup and APK cost"
```

---

## Self-Review Notes

Spec coverage: gating/Hidden (T1 ModelLocator + T2 ViewModel + T2 Step 6 emulator proof); prompt template + exact-string test (T1); lazy engine, CPU, one-at-a-time, length cap (T2); static caption byte-exact (T2 Step 5, verified on device T3); error handling non-crashing (T2 test + implementation); no-LLM-in-CI (all CI steps are JVM-only); airplane-mode demo, APK delta, README setup (T3); litertlm version pinning + INTERNET-guard tripwire (T1). Type consistency: `NoteGenerator`/`ExplainerState`/`PredictionContext` signatures match across tasks. Known judgment calls: LiteRT-LM Kotlin API details behind a documented decision ladder (T2 Step 3), Dispatchers note for the test dispatcher (T2 Step 4).
