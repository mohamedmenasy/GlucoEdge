# LiteRT-LM on-device explanation layer

Date: 2026-07-08
Status: approved design

## Purpose

The original brief's optional stretch goal, and the project's last thread:
a **fully local** natural-language note that describes the current trend
prediction in plain language, generated on-device by a small open-weight
model via LiteRT-LM. Explicitly a **demo feature, not medical guidance** —
and that label is enforced by the app's UI, never delegated to the model.

The no-network property is unchanged: the app still requests no INTERNET
permission, so generation is provably local. (This is the feature's money
shot: an LLM note appearing in airplane mode.)

## Decisions (settled with the user)

1. **Model: Gemma 3 1B int4** as a `.litertlm` file (~550 MB). Exact
   artifact filename/source pinned at plan time from the official LiteRT-LM
   model listing.
2. **UX: on-demand Explain button** — visible only when a prediction
   exists AND the model file is present on the device. Absent model ⇒ the
   feature is invisible and the app behaves exactly as today (CI,
   emulators, fresh clones unaffected).
3. **Structure: Approach A** — thin layer inside the existing app (new
   `explain/` package + one card in `MainScreen`). No product flavors; the
   `litertlm-android` dependency is always compiled in, and the resulting
   APK size delta is **measured and documented**, not hidden. A `withLm`
   flavor split is the recorded fallback if that delta proves egregious.

## Model distribution (constraint-driven)

The model can never be committed (hundreds of MB; external weights) and
the app cannot download it (no INTERNET permission, by design). The user
places it once:

```bash
adb push gemma3-1b-it-int4.litertlm \
  /sdcard/Android/data/com.glucoedge.app/files/
```

The app reads it from `context.getExternalFilesDir(null)` — its own
scoped external dir, readable with zero extra permissions. The app checks
for the file at startup and on resume. Downloading the model file happens
on the user's computer from the official source (documented in README);
the app itself never fetches anything.

## Components

### `Explainer` (new package `com.glucoedge.app.explain`)

- Wraps the LiteRT-LM Kotlin API: `Engine(EngineConfig(modelPath,
  backend = Backend.CPU()))` from `com.google.ai.edge.litertlm:litertlm-android`
  (exact version pinned at plan time; CPU backend by default — no manifest
  additions; GPU is a config change left out until latency demands it).
- **Lazy**: the Engine initializes on first tap, on a background
  dispatcher, with a visible "loading model…" state (first init takes
  seconds). Stays warm afterward; closed in `onCleared`.
- One generation at a time: the button disables while generating; a new
  note replaces the previous one. Output length-capped (max tokens set at
  plan time, on the order of 100 — the note is 2–3 sentences).
- Interface (final signatures at plan time):
  `ExplainerState` = Hidden / Ready / LoadingModel / Generating /
  Note(text) / Error(msg); `fun explain(context: PredictionContext)`.

### Prompt construction (pure, unit-testable)

`PredictionContext` carries only replayed-data facts: current mg/dL,
window min/max/net change over the hour, predicted class name +
confidence, trace label (synthetic/real). The prompt template instructs:
describe this pattern in 2–3 plain sentences; do not give advice,
recommendations, or dosing; do not address the reader as a patient. The
template is a constant with a unit test asserting the exact rendered
string for a known context.

### UI (one card in `MainScreen`)

- "Explain" button in the controls area, rendered only in states other
  than Hidden.
- Note card shows the generated text beneath a **static caption rendered
  by the app**: "On-device demo note — not medical guidance." The model's
  output is never trusted to carry the disclaimer.
- Loading/generating states show progress text; Error shows the existing
  non-crashing banner pattern.

## Failure handling

- Model file absent: feature Hidden (not an error).
- Engine init failure (corrupt/wrong file, unsupported device): Error
  state with a short message; the rest of the app unaffected; retry on
  next tap.
- Generation failure/timeout: same Error pattern. Nothing here may crash
  the app or block the replay pipeline.
- Emulators: unsupported-but-not-blocked. The feature gates only on model
  presence; the documented path is a physical device. If LiteRT-LM's
  native init misbehaves on Apple-Silicon AVDs (as LiteRT's CompiledModel
  did), we document it in the SIGILL-decision-record style rather than
  fight it.

## Testing

- **Unit (CI):** model-file gating (fake file-presence provider), prompt
  construction (exact string), ExplainerState reducer transitions,
  button-visibility logic. No LLM in CI, ever.
- **On-device (documented, manual + smoke):** with the model pushed on
  the S22 Ultra — first-tap init completes; a note generates within the
  length cap and renders under the static caption; airplane-mode
  generation works; APK size delta vs. pre-feature build recorded.
- LLM output **content** is not golden-tested: it is non-deterministic
  and explicitly a demo. The mechanics around it are what we pin.

## Docs

README: un-stretch the roadmap item with setup instructions (official
model download on a computer + the adb push above), the measured APK size
delta, and the same honest framing as the rest of the project. MODELS.md
is unchanged (it documents bundled assets; this model is deliberately not
one).

## Constraints carried from the project brief

- No INTERNET permission — unchanged, enforced by the existing CI check.
- No GlucoBench data committed; no model weights committed.
- No output, string, or doc may imply clinical validity or treatment
  guidance; the demo caption is mandatory static UI.
- The core app must remain fully functional with no model present.

## Out of scope (YAGNI)

Product flavors, GPU backend tuning, model download UX, multi-model
support, conversation/chat UI, prompt experimentation framework,
streaming-token UI polish beyond a simple progress state.

## Open items pinned at plan time (not open questions)

- Exact `litertlm-android` Maven version; exact official Gemma 3 1B int4
  `.litertlm` filename and download source.
- Exact Kotlin generation API shape (Engine → conversation/session →
  generate) from the pinned version's docs, and max-token parameter name.
- Whether an instrumented smoke test is practical (model file too large
  for CI; likely a documented manual gate like the parity suite's
  device run).
