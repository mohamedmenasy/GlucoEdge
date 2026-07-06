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

Currently built: the training pipeline (`training/`) and the decision on
how many output classes the model needs. Not yet built: on-device model
conversion/quantization, the Android app, and the on-device explanation
stretch goal — see [Roadmap](#roadmap).

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
  `python -m conversion.export_replay_trace`.
- Build: `cd android && ./gradlew :app:assembleDebug`
- Unit tests (run in CI): `./gradlew :app:testDebugUnitTest`
- **Parity tests (local-only gate, needs an emulator/device):**
  `./gradlew :app:connectedDebugAndroidTest` — proves the Kotlin inference
  path reproduces the Python benchmark's outputs (golden vectors) for both
  models, including round-and-clip INT8 input quantization. This currently
  runs against the emulator fallback path; the CompiledModel path is
  compile-verified but stays unverified until this suite runs on a physical
  device.

The app shows measured on-device inference latency; numbers from an
emulator are labeled as such and are not device measurements.

## Roadmap

Per the original project plan, in order:

1. ~~Rebuild the training pipeline; resolve 5-class vs 3-class on
   weinstock~~ — done, see above.
2. Convert the trained model to [LiteRT](https://ai.google.dev/edge/litert)
   (the current name for what used to be called TFLite) and post-training
   quantize to INT8. Report the size/latency/accuracy tradeoff, since that
   comparison matters more to this project's story than squeezing out
   extra accuracy. *Not started — this run kept no checkpoint or fixed
   seed, so it starts with a fresh, reproducible retrain.*
3. Scaffold an Android app (Kotlin + Jetpack Compose) that loads the
   quantized model via LiteRT's `CompiledModel` API and runs inference
   against a replayed CGM stream (a CSV replayed at real or sped-up
   5-minute intervals — no real sensor, no network calls, ever).
4. Expand this README with the quantization tradeoff table and finalize
   the disclaimer for the shipped app.

Optional stretch, once the above works end to end: a fully local
on-device explanation layer (LiteRT-LM + a small open-weight model) that
turns a raw trend prediction into a plain-language note — labeled clearly
as a demo feature, not medical guidance.

## Data attribution

CGM data via [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench)
(Sergazinov, Rogovchenko, Chun, Fernandes, and Gaynanova, "GlucoBench:
Curated List of Continuous Glucose Monitoring Datasets with Prediction
Benchmarks," arXiv, 2023). See that repository for the per-dataset
licenses of the five CGM datasets it bundles.
