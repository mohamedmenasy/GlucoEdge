# iOS app: porting the on-device trend classifier to iPhone

Date: 2026-10-09
Status: approved design

## Purpose

Port the GlucoEdge demo app to iOS so the same trained model ships
identically on two platforms. The Android app (Kotlin/Compose, `android/`)
stays as-is and keeps building in CI; this adds a SwiftUI sibling at `ios/`
that runs the **exact same two `.tflite` artifacts** through LiteRT's Swift
CompiledModel API and is held to the **exact same golden parity vectors**.
Primary test device: the user's iPhone 18 Pro Max (iOS 26+).

Portfolio goal (settled with the user): demonstrate one model deployed
bit-comparably to two platforms — **not** an Apple-native Core ML
conversion. No new model files, no new test-set evaluation, so the locked
calibration-coverage study plan is untouched.

As on Android: portfolio demonstration on public GlucoBench-derived
artifacts, explicitly **not a medical device**; no code or output string
may imply clinical validity or treatment guidance. No network calls for
inference, ever.

## Decisions (settled with the user)

1. **Runtime: LiteRT Swift package, pinned exactly 2.3.0.** First release
   tag (2026-10-07) that ships SwiftPM products as URL+checksum binary
   targets (earlier tags used local-path binaries and `unsafeFlags`,
   uninstallable as a versioned dependency). SwiftPM verifies the
   downloaded xcframeworks against the checksums in `Package.swift`.
   Products used: `LiteRT` (Swift CompiledModel API). The same package's
   `TensorFlowLite` product (classic Interpreter over the same C core) is
   the designated fallback **if and only if** CompiledModel proves broken
   on simulator or device — not built preemptively.
2. **Explain feature: Apple Foundation Models framework** (on-device
   `SystemLanguageModel`), not LiteRT-LM. LiteRT-LM's iOS Swift support is
   labeled early preview with multiple broken 2026 release tags, and
   combining it with LiteRT risks a duplicate-runtime clash like the
   Android `libLiteRt.so` conflict. Foundation Models needs no model
   download/sideload and its safety guardrails are always on. Consequence:
   the iOS note is written by a different model than Android's Gemma 3 1B
   — notes will read differently; prompt constraints (describe only, no
   advice, no dosing) carry over.
3. **Deployment target iOS 26.0** — the Foundation Models floor, and
   buildable on GitHub's `macos-26` runners (default Xcode 26.6, iOS 26.5
   simulators; verified against actions/runner-images). No iOS 27-only
   APIs (e.g. `PrivateCloudComputeLanguageModel`, `ContextOptions`),
   so the app builds with both the user's Xcode 27 and CI's Xcode 26.6.
4. **Project structure: plain Xcode project with folder-synced groups,
   sharing Android's asset folders in place.** The app target bundles
   `android/app/src/main/assets/` by folder reference (models +
   synthetic trace + gitignored `real_trace.csv` when present locally);
   the test target bundles
   `android/app/src/androidTest/assets/golden_vectors.json`. One copy of
   every artifact, zero drift. Accepted coupling: if Android's asset
   layout moves, the iOS build breaks loudly at build time. Rejected
   alternatives: Swift package for logic (SwiftPM can't bundle resources
   outside the package directory → copies that drift), XcodeGen (extra
   toolchain dependency for every contributor and CI).
5. **No emulator-style fallback engine at launch.** Android's Interpreter
   fallback existed for one measured reason (CompiledModel SIGILL on
   Apple-Silicon-hosted AVDs — the `rdsvl` SME probe). The iOS simulator
   runs natively on the host M4 Max, which executes SME. No evidence iOS
   needs a second engine; add one only if the first simulator/device run
   proves otherwise (decision 1 names the fallback).

## Repo layout

```
ios/
├── GlucoEdge.xcodeproj          # plain Xcode project, no generator tool
├── GlucoEdge/
│   ├── GlucoEdgeApp.swift
│   ├── Inference/               # TrendClassifier, QuantizationMath, LatencyMeter
│   ├── Replay/                  # Reading, CsvTraceLoader, WindowBuffer, ReplayEngine
│   ├── Explain/                 # PromptBuilder, NoteGenerator protocol,
│   │                            #   FoundationModelsNoteGenerator, ExplainerState
│   └── UI/                      # MainView, MainViewModel
├── GlucoEdgeTests/              # Swift Testing: unit + golden parity + latency bench
├── Config/
│   ├── Base.xcconfig            # shared build settings; optional-includes Local.xcconfig
│   └── Local.xcconfig           # gitignored: DEVELOPMENT_TEAM for device installs
└── scripts/
    └── check_no_network.sh      # CI source scan (see No-network guard)
```

Toolchain: Swift 6, SwiftUI, Swift Charts, Observation (`@Observable`),
Swift Testing. Single dependency: LiteRT 2.3.0 (exact pin).

## Inference engine

**`Classifier` protocol** mirrors Kotlin: `classify(window: [Float]) throws
-> Prediction`, `close()`, `engineLabel`. `Prediction` carries
`probabilities`, `classIndex`, `latencyNanos`, `logits` (logits retained so
parity asserts at logit level, same as Android).

**Latency timer scope is identical to Android:** the timer starts
immediately before the engine run and stops immediately after it returns —
**before** softmax and class selection. (The README/paper describe this
scope; the iOS numbers must be comparable.)

**`TrendClassifier`:** loads a bundled `.tflite` via
`CompiledModel(filePath:environment:)` with the CPU accelerator, creates
input/output `TensorBuffer`s once, writes `[Float]` (float model) or
`[Int8]` (INT8 model), reads logits back. Quantization parameters are read
at load time from `inputTensorQuantization()` / `outputTensorQuantization()`
directly on `CompiledModel` — the Swift API exposes them, so Android's
second-interpreter metadata workaround is unnecessary. Never hardcoded.
`engineLabel` is exactly `"CompiledModel"` (surfaced in the UI stats line).
Window length 12 validated at the call boundary.

**`QuantizationMath`** (straight port of the Kotlin object):
- `quantizeInt8(values:scale:zeroPoint:)`:
  `(x / scale).rounded(.toNearestOrEven) + zeroPoint`, **clipped** to
  [-128, 127]. Half-even ties match `np.round` and the fixed Kotlin
  implementation; the clip is load-bearing (saturate, never wrap — the
  conversion-phase Critical bug).
- `dequantizeInt8(values:scale:zeroPoint:)`: `(q - zeroPoint) * scale` in
  Float32.
- `softmax(logits:)`: max-subtracted, Float32 output.

Unit tests port 1:1, including `roundsExactTiesToEvenLikeNumpy`
(0.5→0, 1.5→2, 2.5→2, -1.5→-2) and the wrap-vs-clip saturation case.

**`LatencyMeter`:** rolling window of the last 100 inference wall times;
mean / median / p95, same index arithmetic as Kotlin
(`p95 = sorted[((n-1)*95)/100]`).

## Golden parity tests

`GoldenParityTests` (Swift Testing; runs on simulator in CI and on the
iPhone locally) consumes the **existing, unmodified**
`android/app/src/androidTest/assets/golden_vectors.json`:

1. **Asset drift guard:** SHA-256 of each bundled `.tflite` equals the
   hash recorded in the goldens — regenerate both together or fail.
2. **Float model:** for all 20 vectors, predicted class matches
   `float_class` and every logit is within 1e-5 of `float_logits`.
3. **INT8 model:** predicted class matches `int8_class`, and logits are
   **bit-exact** against `int8_raw_output` dequantized in Swift by the
   same `QuantizationMath` the app uses (same integer inputs, same fp32
   math on both sides — zero tolerance, as on Android).

No Python changes, no vector regeneration. (The golden windows are
0.1 mg/dL-grained with no ties at the shipped scale, so the half-even
rounding fix does not change expected outputs — established 2026-10-09.)

## Replay

Straight ports, same semantics:

- **`Reading`**: `epochMinutes: Int64` + `mgdl: Float` — integer gap math.
- **`CsvTraceLoader`**: parses `time,gl` (ISO-8601 minute timestamps, UTC,
  fixed `yyyy-MM-dd'T'HH:mm` formatter with `en_US_POSIX` locale);
  malformed rows are skipped and counted, never fatal.
- **`WindowBuffer`**: sliding window of 12; **resets when the gap since
  the previous reading exceeds 5 minutes** — the app-side mirror of
  training's `id_segment` rule (windows never span a sensor gap). Any
  change must preserve this.
- **`ReplayEngine`**: virtual clock via structured concurrency
  (`AsyncStream`), one reading per 5 s at 1×, speeds 1×/4×/16×,
  play/pause. Same pause semantics as the Kotlin Flow version: no
  emission while paused, pause state re-checked after each delay.
- **Trace selection:** prefer `real_trace.csv` when present in the bundle
  (local builds only — it is gitignored and never committed), else the
  committed `synthetic_trace.csv`. The active trace label is shown in
  the UI.

## UI

One SwiftUI screen mirroring the Compose layout: Swift Charts line of the
last 12 readings; current mg/dL + trace label; trend arrow + class name;
five probability bars; stats line (`model · engine · inference count ·
mean ms · p95 ms · skipped rows · device`); play/pause + speed buttons;
float/INT8 toggle; Explain button + note area; fixed disclaimer footer
("Portfolio demo on public research data — not a medical device, not
treatment guidance.").

**`MainViewModel`** (`@Observable`, MainActor): same state shape as
Android's `UiState`. Toggling the model closes the old classifier, creates
the new one, resets the latency meter and clears the prediction. Inference
failure surfaces an error line but replay continues. Empty trace is a
visible error state. A missing bundled model asset fails loud at launch.
Classifier and note generator enter via factory closures so unit tests
inject fakes.

## Explain (Foundation Models)

- **`NoteGenerator` protocol** — the same seam as Android, so ViewModel
  tests fake generation without the framework.
- **Availability gating:** `SystemLanguageModel.default.availability`
  decides whether the Explain button appears. Each
  `.unavailable(reason:)` case (`deviceNotEligible`,
  `appleIntelligenceNotEnabled`, `modelNotReady`) maps to a short status
  line instead of hiding silently — richer than Android's binary
  hidden/ready because the API reports why.
- **Session:** `LanguageModelSession(instructions:)` carries the standing
  rules (write 2–3 plain sentences about the replayed pattern; no advice,
  no recommendations, no dosing; never address the reader as a patient;
  never mention insulin or treatment). The per-request prompt is built by
  the ported `PromptBuilder` from the same `PredictionContext` facts only:
  current mg/dL, window min/max, net change, class name, confidence %,
  trace label. Nothing else may enter the prompt.
- **Options:** `GenerationOptions(temperature: 0.2,
  maximumResponseTokens: 200)`.
- **States:** hidden/unavailable(reason) / ready / generating / note /
  error. One generation at a time (guarded by session `isResponding` and
  the state machine). `GenerationError` cases — `guardrailViolation`,
  `refusal`, `exceededContextWindowSize`, `rateLimited`,
  `assetsUnavailable` — each render as readable error text, never a crash.
  Guardrail false positives on glucose-adjacent text are a known risk; the
  prompt is tested live on the user's iPhone early in implementation.
- The note renders under the same label: "On-device demo note — not
  medical guidance."

## Testing

Swift Testing throughout. Ported assertion-for-assertion from the Kotlin
JVM suites: `QuantizationMath` (ties, saturation, dequant, softmax),
`LatencyMeter` (stats math, rolling eviction), `CsvTraceLoader` (happy
path, malformed rows, blank lines), `WindowBuffer` (fill, slide, gap
reset), `ReplayEngine` (virtual-time pause/speed semantics),
`PromptBuilder` (exact string), `MainViewModel` (fake classifier + fake
note generator: reading flow, toggle reset, error surfacing, explain state
machine). Plus `GoldenParityTests` (above) and a latency benchmark test
(manual trigger) recording equal-n float/INT8 stats.

Foundation Models-dependent tests are availability-gated: they skip (not
fail) where Apple Intelligence is absent — CI simulators in particular.
The Explain logic stays covered through the fake generator.

## CI

New `ios` job appended to `.github/workflows/tests.yml`; the existing
`test` (pytest) and `android` jobs are untouched:

- `runs-on: macos-26` (arm64; default Xcode 26.6 — verified to carry the
  iOS 26.x simulators the 26.0 target needs).
- `xcodebuild test -project ios/GlucoEdge.xcodeproj -scheme GlucoEdge
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'` — builds the
  app and runs every test, golden parity included, on the simulator.
  No code signing (simulator only).
- Runs `scripts/check_no_network.sh`.

## No-network guard (honest version)

iOS has no INTERNET permission, so Android's merged-manifest check has no
structural equivalent, and a binary-symbol scan is useless (Foundation
always exports URLSession symbols). The guard is therefore a CI source
scan: `scripts/check_no_network.sh` fails if any file under `ios/GlucoEdge`
or `ios/GlucoEdgeTests` references `URLSession`, `import Network`,
`NWConnection`, `CFSocket`, or `getaddrinfo`. This is **weaker** than the
Android guarantee and is documented as such in the README — consistent
with the project's rule of stating real limitations rather than implying
stronger ones.

## Signing and device verification

`Config/Local.xcconfig` (gitignored) holds `DEVELOPMENT_TEAM` for
installing on the user's iPhone; `Base.xcconfig` includes it with
`#include?` so CI builds succeed without it.

Device verification on the iPhone 18 Pro Max, recorded in a results
addendum to this spec and in the README:
1. Golden parity suite on-device (3 test groups).
2. Equal-n latency benchmark: ≥100 inferences for float **and** INT8
   (fixes the July asymmetry where INT8 had n=21).
3. One live Explain generation (guardrail behavior check).

## Documentation

- README: iOS section; iPhone latency row alongside the S22 Ultra numbers
  once measured; no-network guard limitation note.
- CLAUDE.md: iOS build/test commands and `ios/` architecture note (and
  the already-pending refresh covering `conversion/`, `android/`,
  `paper/`, checkpoint support).
- `android/app/src/main/assets/MODELS.md`: untouched — the assets are
  shared in place and their provenance is unchanged.

## Risks (accepted)

1. **LiteRT Swift package maturity:** v2.3.0 was tagged two days before
   this design. Mitigations: exact pin, SwiftPM checksum verification,
   golden parity as the acceptance gate, and the same-package
   `TensorFlowLite` Interpreter product as a measured-need fallback.
2. **Cross-directory asset reference** couples the iOS build to
   `android/`'s layout; breaks loudly at build time if assets move.
3. **Foundation Models guardrails** may refuse glucose-adjacent prompts;
   handled as visible states and tested live early.
4. **CI cannot exercise Explain generation** (no Apple Intelligence on
   runners); covered by fakes in CI and a live device check.

## Implementation notes (2026-10-09)

Deviations found during implementation, each recorded at the point it
was decided:

- **LiteRT pin (decision 1).** The `v2.3.0` tag's `Package.swift`
  references `prebuilt/*.zip` files that were never committed, and
  SwiftPM crashes while extracting them. The app instead pins commit
  `8f555ada850ac4cd331d9d10bded0d0048013666` ("Update Package.swift binary
  targets", 2026-10-08), which points at the v2.3.0 release xcframeworks
  by URL and checksum. The binaries are therefore exactly 2.3.0. The Swift
  wrapper sources come from `main` at that commit; golden parity passed on
  that pairing.
- **Git LFS.** The LiteRT repository keeps Android/Linux prebuilts, which
  its Swift package never uses, in Git LFS. SwiftPM's mirror lacks those
  objects for a pinned revision, so `git lfs pull` fails. Package
  resolution runs with `lfs.fetchexclude='*'`, supplied as `GIT_CONFIG_*`
  environment variables (CI job env and the documented CLI command).
  `git-lfs` must still be installed.
- **Concurrency.** `Classifier` and `NoteGenerator` are `@MainActor`
  protocols, because the main-actor view model is their only caller.
- **Explain.**
  - Every generation failure maps to a readable `NoteError`. This
    includes bridged `NSError`s that don't cast to `GenerationError`; one
    was observed on the simulator while guardrail assets were loading.
  - The live guardrail test runs on physical devices only, because the
    simulator reports the host Mac's availability and behaves
    nondeterministically.
  - Availability is re-checked when the app returns to the foreground,
    mirroring Android's `onResumeCheck`.
- **Stats line.** It shows `device`/`simulator`, as this spec's UI section
  lists.
- **Fixes from the final review.**
  - **Latency timer.** It now wraps quantize, buffer write, run, buffer
    read and dequantize, the same region Kotlin's
    `engine.runInference(window)` times. Softmax and class selection stay
    outside. This makes iPhone numbers comparable with the S22 Ultra's.
  - **Explain sessions.** Every note gets a fresh `LanguageModelSession`, as
    Android uses a fresh conversation per note. A reused session failed
    with a context overflow on the 17th consecutive note (reproduced
    locally) and fed earlier notes into later prompts.
  - **Signing.** `Base.xcconfig` is now attached at project level, so the
    test target inherits `DEVELOPMENT_TEAM` for on-device test runs.
  - **Crash fixes.** Rows whose glucose value isn't finite (`nan`, `inf`)
    are skipped and counted. The chart's y-range no longer inverts when
    every reading sits outside 30–420 mg/dL.

## Device verification results (2026-10-10)

iPhone 18 Pro Max (iPhone19,3), iOS 27.0.1, Debug build, signed with a
free Apple ID's Personal Team via `xcodebuild -allowProvisioningUpdates`.

1. **Golden parity:** `GoldenParityTests` 4/4 on the device (asset
   sha256, class names, float logits within 1e-5, INT8 bit-exact). The
   same suite is also green on the iOS 27 simulator and in the `ios` CI job
   (Xcode 26.6, iOS 26.5).
2. **Latency, equal n.** Both measurements time the same region:
   quantize, write, run, read, dequantize.

   | Measurement | Float mean / p95 | INT8 mean / p95 |
   |---|---|---|
   | In-app stats line, 16× replay of the synthetic trace, n=100 each, two rounds | 0.062–0.073 ms / 0.103–0.134 ms | 0.101–0.110 ms / 0.178–0.182 ms |
   | Test-runner loop, 1 warmup + n=100 back-to-back calls, three runs | 2.83–2.85 µs / 3.29–3.38 µs | 3.69–3.72 µs / 4.13–4.25 µs |

   - The in-app row sits beside the S22 Ultra's in the README; both are
     in-app measurements.
   - The loop is 20–30× faster on the same code. Most of the in-app cost
     is therefore per-call overhead between replay ticks, not the model's
     arithmetic.
   - INT8 is not faster on either path, matching Android.
3. **Explain:**
   - `FoundationModelsLiveTests` passed 2/2 with Apple Intelligence
     available. The real prompt produced a descriptive note with no
     advice, and 25 consecutive fresh-session notes took 30.4 s.
   - In the app, tapping Explain (float and INT8, one note each) showed
     descriptive notes under the "not medical guidance" label, with no
     advice. The note describes the window at the moment of the tap;
     replay keeps running while it generates.
   - A throwaway probe (not committed) asked for an exact insulin dose and
     separately tried a "skip your insulin" prompt injection. Both direct
     `respond` and `FoundationModelsNoteGenerator` returned a refusal *as
     text* and threw no error, so the app shows the refusal as the note.
   - The probe never triggered a thrown guardrail error. Whether iOS 27
     surfaces one as `GenerationError` or `LanguageModelError` is
     therefore still unobserved. The catch-all keeps the `NoteError`
     contract either way.
4. **Signing:**
   - `com.glucoedge.ios` was already registered to a different free team
     during setup, and bundle IDs are global. The app ID is now
     `com.mohamedmenasy.glucoedge` (tests `.tests`).
   - Opening the project in Xcode 27 re-serialized `project.pbxproj`
     (sections reordered, `objectVersion` 77 → 71) with no setting changed.
     That rewrite was discarded.
