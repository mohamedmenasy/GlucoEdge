# GlucoEdge

An on-device glucose trend classifier for Android — a portfolio project
demonstrating production on-device ML deployment (privacy, offline
inference, latency), **not a medical device**. It does not diagnose,
monitor, or advise on diabetes management, and none of its outputs should
be used to make treatment decisions.

Built entirely on [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench),
a public academic CGM research benchmark — no proprietary data, schemas,
or algorithms.

## What this is, right now

The task: predict a 5-class glucose trend arrow (`falling_fast` /
`falling` / `stable` / `rising` / `rising_fast`) 15 minutes ahead, from the
last hour of continuous glucose monitor readings. It's a genuine forecast —
the label is computed from a point after the input window ends, not
recomputed from data the model can already see.

Currently built: the training pipeline (`training/`), the decision on how
many output classes the model needs, LiteRT conversion + INT8 quantization
(with a measured size/accuracy tradeoff — see
[the results doc](docs/superpowers/specs/2026-07-01-litert-conversion-results.md)),
and the Android app itself (a replay demo bundling both the float and INT8
models — see [Android app](#android-app-android) below, with the
`CompiledModel` path verified on a physical device). Not yet built: the
on-device explanation stretch goal — see [Roadmap](#roadmap).

**The 5-class question is settled with real data, not assumed.** On
GlucoBench's small 4-patient iglu config, the two "fast" trend classes had
too few examples to learn from (15 and 64). Rather than assume that meant
the classes needed collapsing, the pipeline was rerun on GlucoBench's full
200-patient weinstock config: both fast classes turned out to have
13,000+ training examples and ~53-54% test recall — well past the
thresholds that would have triggered a collapse to 3 classes. Full numbers
and the decision procedure: `docs/superpowers/specs/2026-07-01-weinstock-class-count-decision.md`.

## Architecture

```
training/
├── labeling.py   # label_trend(): rate-of-change -> trend class, clinically-anchored thresholds
├── dataset.py    # GlucoseTrendDataset: segment-aware sliding windows over CGM data
├── model.py      # TrendCNN: a small 1D-CNN (~2,900 params), chosen over a
│                 #   GRU/LSTM specifically because it converts to a fixed-input-shape,
│                 #   stateless inference format with no recurrent state to manage
└── train.py      # CLI: wires GlucoBench's DataFormatter into the above
```

`GlucoseTrendDataset` windows respect GlucoBench's `id_segment` boundaries —
it never slides a window across a gap that GlucoBench's own interpolation
step dropped, which would otherwise silently splice unrelated time periods
together.

## Running it

```bash
# 1. Clone the data dependency (not vendored - see .gitignore)
git clone https://github.com/IrinaStatsLab/GlucoBench.git
cd GlucoBench && unzip raw_data.zip && cd ..

# 2. Set up the environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the tests (fast - no CGM data needed, GlucoBench doesn't even
#    need to be cloned for this step)
pytest -v

# 4. Run the actual training CLI
python -m training.train --dataset iglu --classes 5 --epochs 2       # fast smoke test
python -m training.train --dataset weinstock --classes 5 --epochs 20 # full run, several minutes
```

Each run writes `results/{dataset}_{classes}class_report.json`: per-class
precision/recall/f1 and the training-set class counts, so any classifier
behavior is traceable back to real numbers instead of a single accuracy
figure. (Accuracy alone is misleading here — `stable` is the large
majority of the data, so a model that always predicts `stable` beats a
real classifier on raw accuracy while being useless for the actual
product. Per-class recall is the metric that matters.)

## Android app (`android/`)

Kotlin + Jetpack Compose app that replays a CGM trace through LiteRT
entirely on-device. On real hardware it runs inference via the LiteRT
**CompiledModel** API; on emulators it falls back to the classic
`Interpreter` (XNNPACK off), because `CompiledModel` natively crashes
(SIGILL, in a CPU feature-probe instruction) on Apple-Silicon-hosted AVDs.
The UI shows which engine and which environment (emulator/device) are
active. The manifest requests **no INTERNET permission** — the OS itself
guarantees no network calls for inference, and CI fails if any dependency
tries to merge that permission in.

- Bundled models: `trend_float.tflite` (default) and `trend_int8.tflite`
  (toggle in-app) — provenance and hashes in
  `android/app/src/main/assets/MODELS.md`.
- Replay data: a committed synthetic trace; optionally extract a real
  GlucoBench segment locally (never committed) with
  `python -m conversion.export_replay_trace`. An APK built on a machine
  where `real_trace.csv` has been extracted bundles that (GlucoBench-derived)
  file as an asset — don't distribute such builds; the committed repo and CI
  builds contain only the synthetic trace.
- Build: `cd android && ./gradlew :app:assembleDebug`
- Unit tests (run in CI): `./gradlew :app:testDebugUnitTest`
- **Parity tests (local-only gate, needs an emulator/device):**
  `./gradlew :app:connectedDebugAndroidTest` — proves the Kotlin inference
  path reproduces the Python benchmark's outputs (golden vectors) for both
  models, including round-and-clip INT8 input quantization. On an emulator
  this exercises the `Interpreter` fallback; on a physical device it
  exercises `CompiledModel` — verified 3/3 on a Galaxy S22 Ultra
  (Android 16): float logits within 1e-5, INT8 outputs bit-exact.

The app shows measured on-device inference latency; numbers from an
emulator are labeled as such and are not device measurements.

## The quantization tradeoff, measured

The headline comparison across the three model artifacts, on the full
127,165-window weinstock test set (accuracy/recall) and a real phone
(latency):

| | Size | Latency mean / p95 (device) | Accuracy | Macro-avg recall |
|---|---|---|---|---|
| PyTorch checkpoint | 14.4 KB | — (not deployable) | 0.5184 | 0.5060 |
| Float `.tflite` (shipped default) | 16.95 KB | 0.334 ms / 0.540 ms | 0.5184 | 0.5060 |
| INT8 `.tflite` | 11.70 KB | 0.484 ms / 0.665 ms | 0.5866 | 0.4135 |

Latency is measured in-app on a Samsung Galaxy S22 Ultra (Android 16) via
the CompiledModel path — the full Kotlin classify path (buffer writes,
quantization, softmax), not just kernel invoke; float over a full
100-inference rolling window, INT8 over a 21-inference window. The
conversion phase's dev-machine CPU proxy reached the same qualitative
conclusion: INT8 is not faster at this model size — per-call overhead
dominates a ~2,900-parameter network.

Float conversion is exactly lossless: its classification report is
byte-identical to the PyTorch checkpoint's. INT8's *higher* accuracy is
not an improvement — it's majority-class bias, visible immediately in
per-class recall:

| Class | Support (% of test) | Float recall | INT8 recall | Change |
|---|---|---|---|---|
| `falling_fast` | 3,692 (2.9%) | 0.5119 | 0.1907 | −0.3212 (−62.7% relative) |
| `falling` | 11,387 (9.0%) | 0.5484 | 0.4306 | −0.1178 |
| `stable` | 95,998 (75.5%) | 0.5225 | 0.6529 | +0.1304 |
| `rising` | 11,012 (8.7%) | 0.4761 | 0.3845 | −0.0916 |
| `rising_fast` | 5,076 (4.0%) | 0.4712 | 0.4090 | −0.0622 |

`stable` — 75.5% of the test set — is the only class INT8 helps;
`falling_fast`, the rarest class and half the reason a trend classifier
exists, loses 62.7% of its recall. (A degenerate model that always
predicts `stable` scores ~75.5% accuracy with a macro-avg recall of
exactly 0.20 — which is why this project never reports accuracy alone.)
Part of the INT8 degradation is calibration coverage: static quantization
calibrated on 200 validation windows learned an input ceiling of
~240 mg/dL while the test data reaches 401 mg/dL, so ~28% of test windows
saturate at the INT8 input boundary even with correct clipping.

So the measured tradeoff is: INT8 buys a ~31% smaller file (5.25 KB) and
costs 9.25 macro-recall points, with no latency benefit — which is why
the app ships float as its default and offers INT8 as a toggle. Full
analysis:
[`docs/superpowers/specs/2026-07-01-litert-conversion-results.md`](docs/superpowers/specs/2026-07-01-litert-conversion-results.md).

## Roadmap

Per the original project plan, in order:

1. ~~Rebuild the training pipeline; resolve 5-class vs 3-class on
   weinstock~~ — done, see above.
2. ~~Convert the trained model to [LiteRT](https://ai.google.dev/edge/litert)
   (the current name for what used to be called TFLite) and post-training
   quantize to INT8, reporting the size/latency/accuracy tradeoff~~ — done:
   float conversion is lossless, INT8 is ~31% smaller with no measurable
   latency change but a real macro-avg-recall cost from majority-class
   bias. Full numbers:
   [`docs/superpowers/specs/2026-07-01-litert-conversion-results.md`](docs/superpowers/specs/2026-07-01-litert-conversion-results.md).
3. ~~Build an Android app (Kotlin + Jetpack Compose) that runs inference
   against a replayed CGM stream~~ — done, see
   [Android app](#android-app-android) above.
4. ~~Expand this README with the quantization tradeoff table and finalize
   the disclaimer for the shipped app~~ — done, see
   [The quantization tradeoff, measured](#the-quantization-tradeoff-measured)
   and [Disclaimer](#disclaimer) above.
5. ~~Verify the app's `CompiledModel` inference path on a physical
   device~~ — done: golden-vector parity 3/3 on a Galaxy S22 Ultra
   (Android 16) via `CompiledModel`, no SIGILL; in-app latency float
   mean 0.334 ms / INT8 mean 0.484 ms (full Kotlin classify path — INT8
   is again not faster). Details appended to
   [the SIGILL decision record](docs/superpowers/specs/2026-07-05-emulator-compiledmodel-sigill.md).

Optional stretch, once the above works end to end: a fully local
on-device explanation layer (LiteRT-LM + a small open-weight model) that
turns a raw trend prediction into a plain-language note — labeled clearly
as a demo feature, not medical guidance.

## Disclaimer

GlucoEdge is a portfolio demonstration of on-device ML engineering. It is
**not a medical device**: it does not diagnose, monitor, or advise on
diabetes management, and no output may inform a treatment decision. No
reported number has been clinically validated — accuracy and recall
figures describe performance on a public research benchmark, nothing
more. The app states this on-screen, verbatim: *"Portfolio demo on public
research data — not a medical device, not treatment guidance."* All data
comes from the public GlucoBench research benchmark; no proprietary data,
schemas, or algorithms are used anywhere in the project.

## Data attribution

CGM data via [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench)
(Sergazinov, Rogovchenko, Chun, Fernandes, and Gaynanova, "GlucoBench:
Curated List of Continuous Glucose Monitoring Datasets with Prediction
Benchmarks," arXiv, 2023). See that repository for the per-dataset
licenses of the five CGM datasets it bundles.
