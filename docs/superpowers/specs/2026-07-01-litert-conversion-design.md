# GlucoEdge: LiteRT Conversion and INT8 Quantization

Date: 2026-07-01
Status: Approved, pending implementation plan

## Context

Sub-project 1 (the training pipeline, merged via PR #1) settled the
5-class-vs-3-class question with real weinstock data but deliberately
persisted no model checkpoint and set no random seed — its purpose was
answering that question, not producing a deployable artifact. This
sub-project converts a real, reproducibly-trained model to
[LiteRT](https://ai.google.dev/edge/litert/overview) (the current name for
what used to be TensorFlow Lite) and post-training quantizes it to INT8,
per the original project plan's step 2. The comparison this produces
(size/latency/accuracy before vs. after quantization) matters more to this
project's portfolio story than squeezing out extra accuracy — the
underlying model is already tiny (~2,900 params), so this is about
correctly measuring and honestly reporting whatever real difference
quantization makes, not assuming a dramatic win.

## Grounding facts (from current documentation, not assumption)

The PyTorch-to-LiteRT conversion tooling has been renamed since it was
last discussed anywhere in this project's history — the same kind of
rename the original brief already flagged for TFLite→LiteRT itself:

- The pip package is `litert-torch` (nightly: `litert-torch-nightly`).
  Its predecessor, `ai-edge-torch`, is deprecated and no longer updated.
- Conversion API: `edge_model = litert_torch.convert(model.eval(), sample_inputs)`
  where `sample_inputs` is a tuple of tensors, followed by
  `edge_model.export('model.tflite')`. The source model must be
  `torch.export`-compliant (PyTorch ≥ 2.1.0 — GlucoEdge already pins
  `torch==2.3.0`, so no version change needed). The output file extension
  is still `.tflite` even though the framework is now called LiteRT.
- Static (weights + activations) INT8 post-training quantization is a
  distinct flow from a plain `convert()` call: `torch.export.export(...)`
  → `prepare_pt2e(model, quantizer)` (quantizer configured via
  `get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)`)
  → run real calibration data through the prepared model → `convert_pt2e(...)`
  → `litert_torch.convert(..., quant_config=QuantConfig(pt2e_quantizer=...))`.
  `prepare_pt2e`/`convert_pt2e` come from `torchao`, a dependency this
  project doesn't have yet.
- Running/benchmarking a `.tflite` file (as opposed to converting to one)
  needs a separate runtime package, `ai-edge-litert`.
- None of `litert-torch`, `torchao`, or `ai-edge-litert` are needed by
  `training/` — bundling them into the root `requirements.txt` would force
  the existing CI job to install nightly/pre-release edge-ML packages just
  to run label-threshold unit tests that never touch them.

## Decisions

### Code structure

A new top-level `conversion/` package, separate from `training/`:

```
GlucoEdge/
├── training/                 # unchanged except for two new CLI flags (below)
├── conversion/
│   ├── __init__.py
│   ├── requirements.txt      # litert-torch, torchao, ai-edge-litert - NOT
│   │                         #   merged into the root requirements.txt
│   ├── convert.py            # checkpoint -> float .tflite -> INT8 .tflite
│   └── benchmark.py          # size/latency/accuracy comparison -> report
└── docs/superpowers/specs/
```

Rejected alternatives: a single monolithic script (couples conversion and
benchmarking, which someone may want to run independently — e.g., re-run
just the benchmark against an already-converted file); adding this code
into `training/` directly (would pull `litert-torch`/`torchao`/
`ai-edge-litert` into the fast CI job's install surface for no benefit,
undoing the exact separation the CI work just established).

### Retraining with a fixed seed

`training/train.py` gains two CLI flags:

- `--seed INT`: calls `torch.manual_seed(seed)` before model construction
  and before the training loop starts, so the same seed reproduces the
  same trained weights.
- `--save-checkpoint`: after training, writes
  `results/{dataset}_{classes}class_model.pt` via `torch.save(model.state_dict(), ...)`.

This reuses already-tested training code rather than duplicating the
training loop inside `conversion/`. The conversion pipeline starts from:

```bash
python -m training.train --dataset weinstock --classes 5 --epochs 20 --seed 0 --save-checkpoint
```

matching the exact dataset/classes/epochs of the already-documented
weinstock decision, so the resulting accuracy numbers stay comparable to
that record.

### Conversion (`conversion/convert.py`)

1. Load the checkpoint into a fresh `TrendCNN(num_classes=5)`, `.eval()` it.
2. Float conversion: `litert_torch.convert(model, (sample_input,))` where
   `sample_input` is a `(1, 1, 12)` tensor, then `.export(...)` to
   `results/model_float.tflite`.
3. Static INT8 quantization, following the grounding facts above exactly:
   `torch.export.export` → `prepare_pt2e` with
   `get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)`
   → **calibrate by running real windows from `formatter.val_data` (not
   `test_data`) through the prepared model** → `convert_pt2e` →
   `litert_torch.convert(..., quant_config=...)` → export to
   `results/model_int8.tflite`.

Calibration uses `val_data`, never `test_data`: calibration statistics
shape the quantized model in the same sense training data shapes a
trained model, so using `test_data` for calibration would contaminate the
accuracy-delta comparison the same way training on your test set would.

All `litert_torch`/`torchao` imports are lazy (inside the functions that
use them), matching the pattern already established in `training/train.py`
for its GlucoBench import — so importing `conversion.convert` itself never
requires these dependencies to be installed.

### Benchmarking (`conversion/benchmark.py`)

- **Size:** file size in KB of `model_float.tflite` vs `model_int8.tflite`.
- **Latency:** load each `.tflite` via `ai-edge-litert`'s runtime, run a
  fixed number of warmup calls (discarded), then a fixed number of timed
  calls; report mean/median/p95 wall-clock latency per call for each
  model. Exact warmup/timed-call counts are an implementation-level
  parameter finalized in the implementation plan, not an architectural
  decision — this spec fixes the method (warmup-then-measure,
  mean/median/p95), not the exact iteration counts. This measures CPU
  latency on the development machine as a proxy
  — the report states this explicitly, since real Android on-device
  numbers only become available once the app exists (next sub-project),
  and this document must not imply a phone measurement that didn't happen.
- **Accuracy delta:** run the same weinstock test set (`formatter.test_data`)
  through three things — the original PyTorch checkpoint, the float
  `.tflite`, and the INT8 `.tflite` — and compare `classification_report`
  output across all three. The float `.tflite` is expected to match the
  PyTorch model almost exactly (format conversion, not precision loss);
  the INT8 model is where a real accuracy delta can appear.

Output: `results/conversion_report.json` (sizes, latency stats, all three
classification reports), gitignored like the training reports. Once real
numbers exist, they get written into a committed decision-record markdown
in `docs/superpowers/specs/`, matching the pattern established by the
weinstock class-count decision.

### Testing

Like `training/train.py`, this phase is verified by real runs, not
pytest — there's little pure logic to unit-test in "call
`litert_torch.convert()` and check a file exists," and `litert-torch` is
genuinely pre-release nightly software that doesn't belong gating every
PR any more than the real training runs do. The existing CI job
(`.github/workflows/tests.yml`) is unchanged by this sub-project — it
continues to run only `training/`'s unit tests.

## Out of scope for this spec

- The Android app itself (original plan step 3)
- Real on-device (phone/emulator) latency measurement
- README updates with the quantization tradeoff table (deferred until
  real numbers exist to put in it)
- The optional on-device LLM explanation stretch goal
