# LiteRT Conversion and INT8 Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the glucose-trend classifier with a fixed seed, convert it to LiteRT, post-training quantize it to static INT8, and produce a real, documented size/latency/accuracy comparison across the PyTorch/float-tflite/int8-tflite versions.

**Architecture:** `training/train.py` gains `--seed`/`--save-checkpoint` flags to produce a reproducible checkpoint. A new `conversion/` package (separate dependencies from `training/`, kept out of the existing CI job) converts that checkpoint to a float `.tflite` and a static-INT8 `.tflite` via `litert_torch` + `torchao`'s PT2E flow, then benchmarks all three representations on the same held-out test set.

**Tech Stack:** `litert-torch==0.9.1` (PyTorch→LiteRT conversion), `torchao==0.17.0` (PT2E quantization), `ai-edge-litert==2.1.5` (Python runtime for running `.tflite` files — pinned to `2.1.5`, not the originally-planned `2.1.6`, per Task 2's actual install: `litert-torch==0.9.1` pulls in `ai-edge-quantizer==0.7.0`, which requires exactly `ai-edge-litert==2.1.5`).

## Global Constraints

- The model must stay `torch.export`-compliant (PyTorch ≥2.1.0 — already satisfied by `torch==2.3.0` and `TrendCNN`'s plain conv/relu/pool/linear architecture).
- `conversion/requirements.txt` holds exactly `litert-torch==0.9.1`, `torchao==0.17.0`, `ai-edge-litert==2.1.5` (see Tech Stack note on the version) — these must never be added to the root `requirements.txt`, since the existing CI job (`.github/workflows/tests.yml`) must keep working with only the root install.
- Every `litert_torch`/`torchao`/`ai_edge_litert` import inside `conversion/` must be inside a function, not at module level — matching the lazy-import pattern already established in `training/train.py` for GlucoBench, so importing `conversion.convert` or `conversion.benchmark` never requires these dependencies to be installed.
- INT8 calibration uses `formatter.val_data`, never `formatter.test_data` — the test set is reserved for the accuracy-delta comparison and must stay uncontaminated by anything that shaped the quantized model.
- Latency numbers are measured on this development machine's CPU via `ai_edge_litert`'s `Interpreter`, explicitly labeled as a proxy — nothing in code output or docs may imply this is a real Android on-device measurement (that comes in the next sub-project).
- This is a portfolio piece, not a medical device — no clinical claims in code comments, docstrings, or output text.
- This external tooling (`litert_torch`, `torchao`) is genuinely fast-moving; if actual runtime behavior (exact tensor dtypes, exact API signatures) differs from what a task describes, adapt the code to match observed reality and document what you found — don't force reality to match this plan's assumptions.

---

## Task 1: Reproducible retraining (`--seed` / `--save-checkpoint`)

**Files:**
- Modify: `training/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produces: `set_seed(seed: int) -> None` in `training/train.py`; `train.py`'s CLI gains `--seed INT` (default `None`, no-op if unset) and `--save-checkpoint` (flag; when set, writes `results/{dataset}_{classes}class_model.pt` via `torch.save(model.state_dict(), ...)`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py` (append; keep the existing two tests above it):

```python
from training.train import set_seed


def test_set_seed_makes_model_init_reproducible():
    from training.model import TrendCNN

    set_seed(42)
    model_a = TrendCNN(num_classes=5)

    set_seed(42)
    model_b = TrendCNN(num_classes=5)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)
```

This needs `torch` imported in the test file — check the top of `tests/test_train.py`; if `import torch` isn't already there, add it.

- [ ] **Step 2: Run the test to verify it fails**

```bash
source venv/bin/activate
pytest tests/test_train.py -v
```

Expected: FAIL with `ImportError: cannot import name 'set_seed' from 'training.train'`.

- [ ] **Step 3: Implement `set_seed` and wire the CLI flags**

In `training/train.py`, add this function after the `GLUCOBENCH_ROOT = ...` line (line 19) and before `def load_formatter`:

```python
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
```

Then replace the `main()` function (lines 93-136) with:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iglu", "weinstock"], required=True)
    parser.add_argument("--classes", choices=["5", "3"], default="5")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    formatter = load_formatter(args.dataset)

    if args.classes == "5":
        classes, collapse_map = FIVE_CLASSES, None
    else:
        classes, collapse_map = THREE_CLASSES, THREE_CLASS_MAP

    train_ds = GlucoseTrendDataset(formatter.train_data, classes=classes, collapse_map=collapse_map)
    val_ds = GlucoseTrendDataset(formatter.val_data, classes=classes, collapse_map=collapse_map)
    test_ds = GlucoseTrendDataset(formatter.test_data, classes=classes, collapse_map=collapse_map)

    print(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    train_label_counts = {
        c: int(np.sum(np.array(train_ds.labels) == i)) for i, c in enumerate(classes)
    }
    print(f"train class counts: {train_label_counts}")

    model = train_model(train_ds, val_ds, len(classes), args.epochs, device)

    results_dir = GLUCOEDGE_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if args.save_checkpoint:
        checkpoint_path = results_dir / f"{args.dataset}_{args.classes}class_model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"wrote {checkpoint_path}")

    report = evaluate(model, test_ds, classes, device)

    out_path = results_dir / f"{args.dataset}_{args.classes}class_report.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "num_classes": len(classes),
                "classes": classes,
                "train_label_counts": train_label_counts,
                "classification_report": report,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

(The only substantive changes from the existing file: the two new `parser.add_argument` calls, the `if args.seed is not None: set_seed(args.seed)` block, moving `results_dir = ...` / `results_dir.mkdir(...)` up so the checkpoint save can use it, and the new `if args.save_checkpoint:` block.)

- [ ] **Step 4: Run the test to verify it passes**

```bash
source venv/bin/activate
pytest tests/test_train.py -v
```

Expected: all 3 tests PASS (2 pre-existing + 1 new).

- [ ] **Step 5: Verify determinism end-to-end with a real run**

```bash
source venv/bin/activate
python -m training.train --dataset iglu --classes 5 --epochs 2 --seed 0 --save-checkpoint
mv results/iglu_5class_model.pt /tmp/checkpoint_run1.pt
python -m training.train --dataset iglu --classes 5 --epochs 2 --seed 0 --save-checkpoint
shasum -a 256 /tmp/checkpoint_run1.pt results/iglu_5class_model.pt
```

Expected: the two `sha256` hashes are identical — same seed, same data, same result. If they differ, stop and investigate before proceeding (something is non-deterministic that the `--seed` flag isn't controlling) rather than continuing to Task 3 on top of a checkpoint mechanism that doesn't actually reproduce.

- [ ] **Step 6: Run the full suite and commit**

```bash
pytest -v
```

Expected: all tests pass (24 total: 23 pre-existing + 1 new).

```bash
git add training/train.py tests/test_train.py
git commit -m "Add --seed and --save-checkpoint flags to the training CLI"
```

---

## Task 2: `conversion/` package scaffold

**Files:**
- Create: `conversion/__init__.py`
- Create: `conversion/requirements.txt`

**Interfaces:**
- Produces: an installable `conversion` package (empty for now) and a pinned dependency list, installed alongside (not instead of) the root `requirements.txt` in the same venv — `conversion/convert.py` (Task 4) will import both `training.model` and `litert_torch`, so both need to be on the same Python path.

- [ ] **Step 1: Create the package stub**

`conversion/__init__.py`: empty file.

- [ ] **Step 2: Write the pinned requirements**

`conversion/requirements.txt`:

```
litert-torch==0.9.1
torchao==0.17.0
ai-edge-litert==2.1.5
```

- [ ] **Step 3: Install and verify**

```bash
source venv/bin/activate
pip install -r conversion/requirements.txt
python -c "import litert_torch; import torchao; import ai_edge_litert; print('ok')"
```

Expected: prints `ok` with no `ImportError`. If any of these three pinned versions fails to resolve (this ecosystem moves fast), check `pip index versions <package>` for the current stable release, use that instead, and note the substitution in your task report — that's a real, expected possibility with fast-moving pre-1.0/nightly-adjacent tooling, not a sign anything else is wrong.

- [ ] **Step 4: Confirm the existing test suite is unaffected**

```bash
pytest -v
```

Expected: still 24/24 passing — installing `conversion/`'s dependencies must not change anything about `training/`'s test results.

- [ ] **Step 5: Commit**

```bash
git add conversion/__init__.py conversion/requirements.txt
git commit -m "Add conversion package scaffold and pinned LiteRT/torchao/ai-edge-litert deps"
```

---

## Task 3: Produce a reproducible weinstock checkpoint

**Files:** none created or modified — this task runs Task 1's CLI flags for real, producing artifacts that Tasks 4-5 depend on.

**Interfaces:**
- Produces: `results/weinstock_5class_model.pt` (the checkpoint Task 4 loads) and a fresh `results/weinstock_5class_report.json` (this run's own accuracy numbers, for sanity-checking against the original weinstock decision record).

- [ ] **Step 1: Run the seeded weinstock retrain**

```bash
source venv/bin/activate
python -m training.train --dataset weinstock --classes 5 --epochs 20 --seed 0 --save-checkpoint
```

Expected: completes (200 patients — several minutes, same as the original weinstock decision run), ends with two `wrote results/...` lines (the checkpoint, then the report).

- [ ] **Step 2: Sanity-check against the original weinstock decision**

```bash
python -c "
import json
r = json.load(open('results/weinstock_5class_report.json'))
cr = r['classification_report']
print('falling_fast recall:', cr['falling_fast']['recall'], 'support:', cr['falling_fast']['support'])
print('rising_fast recall:', cr['rising_fast']['recall'], 'support:', cr['rising_fast']['support'])
print('accuracy:', cr['accuracy'])
"
```

Expected: numbers in the same ballpark as the original decision record (`docs/superpowers/specs/2026-07-01-weinstock-class-count-decision.md`: falling_fast recall 0.5333, rising_fast recall 0.5380, accuracy 0.5508) — exact match isn't expected (this is a different random seed and a fresh random weight initialization), but a wildly different result (e.g., accuracy near the 5-class chance baseline of ~20%) would mean something is wrong with this run, and you should stop and investigate rather than converting a broken model.

- [ ] **Step 3: Confirm the checkpoint file exists and is non-trivial**

```bash
ls -la results/weinstock_5class_model.pt
```

Expected: file exists, size consistent with a ~2,900-parameter float32 model (a few tens of KB) plus some torch serialization overhead.

No commit for this task — `results/` is gitignored, and no source files changed. Task 4 depends on this checkpoint existing on disk.

---

## Task 4: Convert to LiteRT (float + static INT8)

**Files:**
- Create: `conversion/common.py`
- Create: `conversion/convert.py`

**Interfaces:**
- Consumes: `training.model.TrendCNN(num_classes)` (Task 4 of the training-pipeline plan); `training.train.load_formatter(dataset) -> DataFormatter` and `training.dataset.GlucoseTrendDataset` (for calibration data); the checkpoint from Task 3 (`results/{dataset}_{classes}class_model.pt`).
- Produces: `conversion.common.load_checkpoint(checkpoint_path: Path, num_classes: int) -> TrendCNN` (Task 5's `benchmark.py` reuses this — don't redefine it there). `results/model_float.tflite` and `results/model_int8.tflite` when run as `python -m conversion.convert --dataset weinstock --classes 5`.

This task is an integration entry point verified by a real run, like `training/train.py` — there's no meaningful pytest coverage for "call an external converter and check a file appears."

- [ ] **Step 1: Implement `conversion/common.py`**

```python
"""Shared helpers for the conversion/ CLIs."""
from pathlib import Path

import torch

from training.model import TrendCNN


def load_checkpoint(checkpoint_path: Path, num_classes: int) -> TrendCNN:
    model = TrendCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model
```

- [ ] **Step 2: Implement `conversion/convert.py`**

```python
"""Converts a trained TrendCNN checkpoint to LiteRT: a float .tflite and a
static-INT8-quantized .tflite, calibrated on real held-out CGM windows."""
import argparse
import sys
from pathlib import Path

import torch

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GLUCOEDGE_ROOT))

from conversion.common import load_checkpoint  # noqa: E402


def convert_float(model, sample_input: torch.Tensor, out_path: Path) -> None:
    import litert_torch

    edge_model = litert_torch.convert(model, (sample_input,))
    edge_model.export(str(out_path))


def convert_int8(model, sample_input, calibration_inputs, out_path: Path) -> None:
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    exported = torch.export.export(model, (sample_input,)).module()

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    prepared = prepare_pt2e(exported, quantizer)

    for calib_input in calibration_inputs:
        prepared(calib_input)

    quantized = convert_pt2e(prepared, fold_quantize=False)

    edge_model = litert_torch.convert(
        quantized, (sample_input,), quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    edge_model.export(str(out_path))


def build_calibration_inputs(val_ds, max_samples: int = 200) -> list:
    inputs = []
    for i in range(min(max_samples, len(val_ds))):
        x, _ = val_ds[i]
        inputs.append(x.unsqueeze(0))
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iglu", "weinstock"], required=True)
    parser.add_argument("--classes", choices=["5", "3"], default="5")
    args = parser.parse_args()

    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP
    from training.train import load_formatter

    results_dir = GLUCOEDGE_ROOT / "results"
    num_classes = 5 if args.classes == "5" else 3
    checkpoint_path = results_dir / f"{args.dataset}_{args.classes}class_model.pt"

    model = load_checkpoint(checkpoint_path, num_classes)
    sample_input = torch.zeros(1, 1, 12, dtype=torch.float32)

    float_path = results_dir / "model_float.tflite"
    convert_float(model, sample_input, float_path)
    print(f"wrote {float_path}")

    classes, collapse_map = (FIVE_CLASSES, None) if args.classes == "5" else (THREE_CLASSES, THREE_CLASS_MAP)
    formatter = load_formatter(args.dataset)
    val_ds = GlucoseTrendDataset(formatter.val_data, classes=classes, collapse_map=collapse_map)
    calibration_inputs = build_calibration_inputs(val_ds)
    print(f"calibration windows: {len(calibration_inputs)}")

    int8_path = results_dir / "model_int8.tflite"
    convert_int8(model, sample_input, calibration_inputs, int8_path)
    print(f"wrote {int8_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the conversion against the Task 3 checkpoint**

```bash
source venv/bin/activate
python -m conversion.convert --dataset weinstock --classes 5
```

Expected: prints the calibration window count (should be 200, matching `max_samples`), then two `wrote results/model_*.tflite` lines. If any step raises an error from `litert_torch`/`torchao` — e.g., `torch.export` rejecting `TrendCNN`, or a different call signature than what's shown above — that's the external-library-behavior risk named in the Global Constraints: read the actual error, check the installed package's real API (`help(litert_torch.convert)`, or the package's own source under `venv/lib/*/site-packages/litert_torch/`), fix the call to match reality, and note what changed in your task report.

- [ ] **Step 4: Confirm both files exist and compare sizes**

```bash
ls -la results/model_float.tflite results/model_int8.tflite
```

Expected: both files exist; `model_int8.tflite` should generally be smaller than `model_float.tflite`, though given how tiny this model already is (~2,900 params), don't be surprised if the difference is modest — that's a real, reportable finding, not a bug to chase.

- [ ] **Step 5: Commit**

```bash
git add conversion/common.py conversion/convert.py
git commit -m "Add LiteRT conversion: float export + static INT8 via PT2E"
```

(`results/model_*.tflite` stay untracked per `.gitignore`'s existing `*.tflite` pattern.)

---

## Task 5: Benchmark size, latency, and accuracy

**Files:**
- Create: `conversion/benchmark.py`

**Interfaces:**
- Consumes: `conversion.common.load_checkpoint` (Task 4 — reuse this, don't redefine it); `results/model_float.tflite` and `results/model_int8.tflite` (Task 4); the checkpoint from Task 3; `training.dataset.GlucoseTrendDataset`, `training.train.load_formatter`.
- Produces: `results/conversion_report.json` when run as `python -m conversion.benchmark --dataset weinstock --classes 5`.

Like Task 4, this is verified by a real run, not pytest.

- [ ] **Step 1: Implement `conversion/benchmark.py`**

```python
"""Benchmarks the converted LiteRT models against the original PyTorch
checkpoint: file size, inference latency, and accuracy (classification
report) on the same held-out test set for all three representations."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GLUCOEDGE_ROOT))

from conversion.common import load_checkpoint  # noqa: E402

WARMUP_RUNS = 20
TIMED_RUNS = 200


def file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def _quantize_input(window: np.ndarray, input_detail: dict) -> np.ndarray:
    x = window.reshape(1, 1, 12).astype(np.float32)
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return x
    scale, zero_point = input_detail["quantization"]
    return ((x / scale) + zero_point).astype(dtype)


def _dequantize_output(raw_output: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return raw_output
    scale, zero_point = output_detail["quantization"]
    return (raw_output.astype(np.float32) - zero_point) * scale


def benchmark_tflite_latency(tflite_path: Path, sample_window: np.ndarray) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x = _quantize_input(sample_window, input_details[0])

    for _ in range(WARMUP_RUNS):
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

    timings_ms = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": statistics.quantiles(timings_ms, n=100)[94],
    }


def predict_tflite(tflite_path: Path, windows: list) -> list:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    for window in windows:
        x = _quantize_input(window, input_details[0])
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])
        output = _dequantize_output(raw_output, output_details[0])
        preds.append(int(np.argmax(output[0])))
    return preds


def predict_pytorch(model, windows: list) -> list:
    preds = []
    with torch.no_grad():
        for window in windows:
            x = torch.from_numpy(window).float().reshape(1, 1, 12)
            out = model(x)
            preds.append(int(out.argmax(dim=1).item()))
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iglu", "weinstock"], required=True)
    parser.add_argument("--classes", choices=["5", "3"], default="5")
    args = parser.parse_args()

    from sklearn.metrics import classification_report

    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP
    from training.train import load_formatter

    results_dir = GLUCOEDGE_ROOT / "results"
    num_classes = 5 if args.classes == "5" else 3
    checkpoint_path = results_dir / f"{args.dataset}_{args.classes}class_model.pt"
    float_path = results_dir / "model_float.tflite"
    int8_path = results_dir / "model_int8.tflite"

    model = load_checkpoint(checkpoint_path, num_classes)

    classes, collapse_map = (FIVE_CLASSES, None) if args.classes == "5" else (THREE_CLASSES, THREE_CLASS_MAP)
    formatter = load_formatter(args.dataset)
    test_ds = GlucoseTrendDataset(formatter.test_data, classes=classes, collapse_map=collapse_map)
    test_windows = [test_ds.windows[i] for i in range(len(test_ds))]
    test_labels = list(test_ds.labels)
    print(f"test windows: {len(test_windows)}")

    pytorch_preds = predict_pytorch(model, test_windows)
    float_preds = predict_tflite(float_path, test_windows)
    int8_preds = predict_tflite(int8_path, test_windows)

    def report(preds):
        return classification_report(
            test_labels,
            preds,
            target_names=classes,
            labels=list(range(len(classes))),
            output_dict=True,
            zero_division=0,
        )

    output = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "size_kb": {
            "float_tflite": file_size_kb(float_path),
            "int8_tflite": file_size_kb(int8_path),
        },
        "latency_ms": {
            "float_tflite": benchmark_tflite_latency(float_path, test_windows[0]),
            "int8_tflite": benchmark_tflite_latency(int8_path, test_windows[0]),
        },
        "classification_report": {
            "pytorch": report(pytorch_preds),
            "float_tflite": report(float_preds),
            "int8_tflite": report(int8_preds),
        },
    }

    out_path = results_dir / "conversion_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

```bash
source venv/bin/activate
python -m conversion.benchmark --dataset weinstock --classes 5
```

Expected: prints the test window count (should match the count from the original weinstock decision record's test set, ~127,165), then `wrote results/conversion_report.json`. This will take a while — it runs every test window through three models sequentially, one at a time (not batched), so expect noticeably longer than the training runs. If it's impractically slow, that's worth noting in your report, but let it finish rather than reducing test-set coverage on your own judgment — the accuracy-delta comparison should be over the same full test set used for the original decision, not a subsample.

- [ ] **Step 3: Sanity-check the report**

```bash
python -c "
import json
r = json.load(open('results/conversion_report.json'))
print('sizes (KB):', r['size_kb'])
print('latency (ms):', r['latency_ms'])
print('pytorch accuracy:', r['classification_report']['pytorch']['accuracy'])
print('float_tflite accuracy:', r['classification_report']['float_tflite']['accuracy'])
print('int8_tflite accuracy:', r['classification_report']['int8_tflite']['accuracy'])
"
```

Expected: `pytorch` and `float_tflite` accuracy should be nearly identical (format conversion, not precision loss) — if they differ substantially, something in the float conversion path is wrong and needs investigation before trusting the INT8 comparison. `int8_tflite` accuracy may differ more; that real difference is the finding this whole sub-project exists to measure and report, not something to "fix" by adjusting the comparison.

- [ ] **Step 4: Commit**

```bash
git add conversion/benchmark.py
git commit -m "Add conversion benchmark: size, latency, and 3-way accuracy comparison"
```

---

## Task 6: Document the real results

**Files:**
- Create: `docs/superpowers/specs/<today's-date>-litert-conversion-results.md`

No new source code — this task turns Task 5's real numbers into a committed, durable record, matching the pattern established by the weinstock class-count decision.

- [ ] **Step 1: Write the results record**

Create `docs/superpowers/specs/<today's-date>-litert-conversion-results.md` (use the actual current date, `YYYY-MM-DD`, matching the convention of the other files in this directory) with the real numbers from `results/conversion_report.json` filled in — no bracketed placeholder may remain in the committed file:

```markdown
# LiteRT conversion and INT8 quantization results

Retrained weinstock/5-class with a fixed seed (`--seed 0`) and converted
the checkpoint to LiteRT, then post-training quantized to static INT8
(PT2E, calibrated on 200 real windows from the validation split).

## Size

- Float `.tflite`: <actual size_kb.float_tflite> KB
- INT8 `.tflite`: <actual size_kb.int8_tflite> KB

## Latency (this development machine's CPU, not an Android device)

|  | mean (ms) | median (ms) | p95 (ms) |
|---|---|---|---|
| Float `.tflite` | <mean_ms> | <median_ms> | <p95_ms> |
| INT8 `.tflite` | <mean_ms> | <median_ms> | <p95_ms> |

Real on-device Android latency will be measured once the app exists
(next sub-project) — these are a CPU proxy, not a phone measurement.

## Accuracy (same weinstock test set as the original class-count decision)

|  | accuracy | macro-avg recall |
|---|---|---|
| Original PyTorch checkpoint | <value> | <value> |
| Float `.tflite` | <value> | <value> |
| INT8 `.tflite` | <value> | <value> |

<One or two sentences on what the real numbers show: e.g., whether the
float conversion preserved accuracy as expected, and whether INT8
quantization's accuracy cost is small/large relative to its size
reduction, stated plainly from the observed numbers - not assumed.>
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/*-litert-conversion-results.md
git commit -m "Document LiteRT conversion size/latency/accuracy results"
```

---

## Definition of done

- `pytest` passes with 24/24 tests green (unchanged CI job).
- `results/weinstock_5class_model.pt`, `results/model_float.tflite`, `results/model_int8.tflite`, and `results/conversion_report.json` exist locally with real data behind them.
- A committed results record states, with real numbers, the size/latency/accuracy tradeoff of INT8 quantization — this is what the eventual README's quantization tradeoff table (original plan step 4) will be built from, and what the Android app (next sub-project) will bundle: `model_int8.tflite` unless the results record's numbers say otherwise.
