# Training Pipeline Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild GlucoEdge's training pipeline from scratch (the described starting file did not exist anywhere on disk) and use it to resolve, with real weinstock numbers, whether the glucose trend classifier needs 5 or 3 output classes.

**Architecture:** A small `training/` package (labeling, dataset, model, CLI) sits alongside a cloned, unmodified `GlucoBench/` checkout (already present at the repo root). `training/train.py` wires GlucoBench's `DataFormatter` output into a custom windowing `Dataset` and a small 1D-CNN, then writes per-class metrics to a JSON artifact.

**Tech Stack:** Python 3.12, PyTorch 2.3, pandas/numpy/scikit-learn, PyYAML, pytest.

## Global Constraints

- Public GlucoBench data only — no proprietary code, data schemas, or algorithms from prior FDA-regulated work, even reconstructed from memory.
- `gl` stays unscaled (mg/dL) for both the iglu and weinstock configs; label thresholds are defined in mg/dL/min.
- Windows must never cross an `(id, id_segment)` boundary — `id_segment` is GlucoBench's marker for a contiguous gap-free run after interpolation.
- Windowing stride is 1 (a new sample every 5 minutes), matching how the model will be queried once deployed.
- `requirements.txt` is limited to `numpy==1.26.4`, `pandas==2.2.2`, `torch==2.3.0`, `scikit-learn==1.4.2`, `PyYAML==6.0.1`, `pytest>=7.0` — none of GlucoBench's heavier benchmark-only dependencies (darts, optuna, pytorch-lightning, xgboost, statsforecast, pmdarima).
- `GlucoBench/` is gitignored and never vendored into GlucoEdge's own git history (already set up).
- This is a portfolio piece, not a medical device — no clinical claims in code comments, docstrings, or output text.

---

## Task 1: Environment, scaffolding, and data setup

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `training/__init__.py`

**Interfaces:**
- Produces: a working venv at `venv/`, unzipped raw CSVs at `GlucoBench/raw_data/*.csv`, an importable empty `training` package.

- [ ] **Step 1: Write `requirements.txt`**

```
numpy==1.26.4
pandas==2.2.2
torch==2.3.0
scikit-learn==1.4.2
PyYAML==6.0.1
pytest>=7.0
```

- [ ] **Step 2: Create the venv and install dependencies**

Run from `/Users/mohamednabil/Documents/vibe/GlucoEdge`:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: install completes with no errors.

- [ ] **Step 3: Verify the interpreter and key imports**

```bash
source venv/bin/activate
python -c "import torch, pandas, numpy, sklearn, yaml; print(torch.__version__)"
```

Expected: prints `2.3.0` with no `ImportError`.

- [ ] **Step 4: Unzip the raw CGM data**

```bash
cd GlucoBench && unzip -o raw_data.zip && cd ..
```

Expected output includes `inflating: raw_data/iglu.csv` and `inflating: raw_data/weinstock.csv`.

- [ ] **Step 5: Verify the CSVs landed correctly**

```bash
ls -la GlucoBench/raw_data/
```

Expected: `iglu.csv`, `weinstock.csv`, `hall.csv`, `colas.csv`, `dubosson.csv` all present with non-zero size.

- [ ] **Step 6: Write `pytest.ini` and the empty `training` package**

`pytest.ini`:
```ini
[pytest]
pythonpath = .
```

`training/__init__.py`: empty file.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini training/__init__.py
git commit -m "Add project scaffolding: venv deps, pytest config, training package stub"
```

---

## Task 2: `label_trend()` and class vocabulary

**Files:**
- Create: `training/labeling.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Produces:
  - `FIVE_CLASSES: list[str]` = `["falling_fast", "falling", "stable", "rising", "rising_fast"]`
  - `THREE_CLASSES: list[str]` = `["falling", "stable", "rising"]`
  - `THREE_CLASS_MAP: dict[str, str]` mapping each of `FIVE_CLASSES` to its `THREE_CLASSES` collapse target
  - `label_trend(last_value: float, future_value: float, horizon_minutes: float = 15.0) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_labeling.py`:

```python
import pytest

from training.labeling import (
    FIVE_CLASSES,
    THREE_CLASSES,
    THREE_CLASS_MAP,
    label_trend,
)


@pytest.mark.parametrize("last_value,future_value,expected", [
    (100.0, 100.0 - 2 * 15, "falling_fast"),   # rate == -2 (boundary)
    (100.0, 100.0 - 3 * 15, "falling_fast"),   # rate == -3 (interior)
    (100.0, 100.0 - 1 * 15, "falling"),        # rate == -1 (boundary)
    (100.0, 100.0 - 1.5 * 15, "falling"),      # rate == -1.5 (interior)
    (100.0, 100.0 - 0.5 * 15, "stable"),       # rate == -0.5 (interior)
    (100.0, 100.0, "stable"),                  # rate == 0
    (100.0, 100.0 + 0.99 * 15, "stable"),      # rate == 0.99 (just under boundary)
    (100.0, 100.0 + 1 * 15, "rising"),         # rate == 1 (boundary)
    (100.0, 100.0 + 1.5 * 15, "rising"),       # rate == 1.5 (interior)
    (100.0, 100.0 + 2 * 15, "rising_fast"),    # rate == 2 (boundary)
    (100.0, 100.0 + 3 * 15, "rising_fast"),    # rate == 3 (interior)
])
def test_label_trend_thresholds(last_value, future_value, expected):
    assert label_trend(last_value, future_value, horizon_minutes=15.0) == expected


def test_five_classes_order():
    assert FIVE_CLASSES == ["falling_fast", "falling", "stable", "rising", "rising_fast"]


def test_three_class_map_covers_all_five_classes():
    assert set(THREE_CLASS_MAP.keys()) == set(FIVE_CLASSES)
    assert set(THREE_CLASS_MAP.values()) == set(THREE_CLASSES)


def test_three_class_map_collapses_fast_tiers():
    assert THREE_CLASS_MAP["falling_fast"] == "falling"
    assert THREE_CLASS_MAP["falling"] == "falling"
    assert THREE_CLASS_MAP["stable"] == "stable"
    assert THREE_CLASS_MAP["rising"] == "rising"
    assert THREE_CLASS_MAP["rising_fast"] == "rising"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_labeling.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'training.labeling'`.

- [ ] **Step 3: Implement `training/labeling.py`**

```python
FIVE_CLASSES = ["falling_fast", "falling", "stable", "rising", "rising_fast"]

THREE_CLASSES = ["falling", "stable", "rising"]

THREE_CLASS_MAP = {
    "falling_fast": "falling",
    "falling": "falling",
    "stable": "stable",
    "rising": "rising",
    "rising_fast": "rising",
}


def label_trend(last_value: float, future_value: float, horizon_minutes: float = 15.0) -> str:
    rate = (future_value - last_value) / horizon_minutes
    if rate <= -2:
        return "falling_fast"
    elif rate <= -1:
        return "falling"
    elif rate < 1:
        return "stable"
    elif rate < 2:
        return "rising"
    else:
        return "rising_fast"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
source venv/bin/activate
pytest tests/test_labeling.py -v
```

Expected: all tests PASS (14 total: 11 parametrized + 3 standalone).

- [ ] **Step 5: Commit**

```bash
git add training/labeling.py tests/test_labeling.py
git commit -m "Add label_trend() and 5-class/3-class label vocabulary"
```

---

## Task 3: `GlucoseTrendDataset`

**Files:**
- Create: `training/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `training.labeling.label_trend(last_value, future_value, horizon_minutes) -> str` (Task 2)
- Produces:
  - `GlucoseTrendDataset(df, classes, collapse_map=None, id_col="id", segment_col="id_segment", time_col="time", target_col="gl", input_length=12, horizon=3, horizon_minutes=15.0)` — a `torch.utils.data.Dataset`
  - Public attributes: `.windows: list[np.ndarray]`, `.labels: list[int]`
  - `__len__() -> int`
  - `__getitem__(idx) -> tuple[torch.FloatTensor, int]` where the tensor has shape `(1, input_length)`

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset.py`:

```python
import pandas as pd
import torch

from training.dataset import GlucoseTrendDataset
from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP


def _make_df(segment_lengths, start_value=100.0, step=0.0):
    """Synthetic single-patient dataframe. Each entry in segment_lengths
    becomes its own id_segment; values increase by `step` per row within
    a segment, restarting at start_value at the top of each segment."""
    rows = []
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    for seg_id, length in enumerate(segment_lengths):
        for i in range(length):
            rows.append({
                "id": "p1",
                "id_segment": seg_id,
                "time": t0 + pd.Timedelta(minutes=5 * i),
                "gl": start_value + step * i,
            })
    return pd.DataFrame(rows)


def test_windows_never_cross_segment_boundary():
    # Two 10-point segments. span = input_length(12) + horizon(3) = 15, so
    # neither segment alone can produce a window. If the boundary were
    # ignored and this were treated as one 20-point run, it would produce
    # 20 - 15 + 1 = 6 windows instead.
    df = _make_df([10, 10])
    dataset = GlucoseTrendDataset(df, classes=FIVE_CLASSES)
    assert len(dataset) == 0


def test_sample_count_for_known_segment_length():
    # One 17-point segment: span = 15, so valid windows = 17 - 15 + 1 = 3.
    df = _make_df([17])
    dataset = GlucoseTrendDataset(df, classes=FIVE_CLASSES)
    assert len(dataset) == 3


def test_window_and_label_alignment():
    # One 15-point segment (exactly span), so exactly 1 window.
    # values[k] = 100 + 20*k -> last window point (idx 11) = 320,
    # future point (idx 14, 3 steps/15min later) = 380,
    # rate = (380 - 320) / 15 = 4.0 mg/dL/min -> rising_fast.
    df = _make_df([15], start_value=100.0, step=20.0)
    dataset = GlucoseTrendDataset(df, classes=FIVE_CLASSES)
    assert len(dataset) == 1

    x, y = dataset[0]
    assert x.shape == (1, 12)
    expected_window = [100.0 + 20.0 * k for k in range(12)]
    assert torch.allclose(x.squeeze(0), torch.tensor(expected_window, dtype=torch.float32))
    assert FIVE_CLASSES[y] == "rising_fast"


def test_collapse_map_reduces_to_three_classes():
    df = _make_df([15], start_value=100.0, step=20.0)  # same rising_fast case
    dataset = GlucoseTrendDataset(df, classes=THREE_CLASSES, collapse_map=THREE_CLASS_MAP)
    assert len(dataset) == 1

    _, y = dataset[0]
    assert THREE_CLASSES[y] == "rising"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'training.dataset'`.

- [ ] **Step 3: Implement `training/dataset.py`**

```python
import numpy as np
import torch
from torch.utils.data import Dataset

from training.labeling import label_trend


class GlucoseTrendDataset(Dataset):
    def __init__(
        self,
        df,
        classes,
        collapse_map=None,
        id_col="id",
        segment_col="id_segment",
        time_col="time",
        target_col="gl",
        input_length=12,
        horizon=3,
        horizon_minutes=15.0,
    ):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.collapse_map = collapse_map
        self.input_length = input_length
        self.horizon = horizon
        self.horizon_minutes = horizon_minutes

        self.windows = []
        self.labels = []

        span = input_length + horizon
        for _, group in df.groupby([id_col, segment_col]):
            group = group.sort_values(time_col)
            values = group[target_col].to_numpy(dtype=np.float32)
            n_valid = len(values) - span + 1
            for i in range(max(n_valid, 0)):
                window = values[i:i + input_length]
                last_value = window[-1]
                future_value = values[i + input_length + horizon - 1]
                raw_label = label_trend(float(last_value), float(future_value), horizon_minutes)
                label = collapse_map[raw_label] if collapse_map else raw_label
                self.windows.append(window)
                self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx]).float().unsqueeze(0)
        y = self.labels[idx]
        return x, y
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
source venv/bin/activate
pytest tests/test_dataset.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py tests/test_dataset.py
git commit -m "Add GlucoseTrendDataset with segment-aware sliding windows"
```

---

## Task 4: `TrendCNN`

**Files:**
- Create: `training/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `TrendCNN(num_classes: int = 5)` — an `nn.Module` with `forward(x)` accepting `FloatTensor[batch, 1, 12]` and returning `FloatTensor[batch, num_classes]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:

```python
import torch

from training.model import TrendCNN


def test_forward_pass_output_shape():
    model = TrendCNN(num_classes=5)
    x = torch.randn(4, 1, 12)
    out = model(x)
    assert out.shape == (4, 5)


def test_param_count_five_class_budget():
    model = TrendCNN(num_classes=5)
    count = sum(p.numel() for p in model.parameters())
    assert count == 2909


def test_param_count_three_class_budget():
    model = TrendCNN(num_classes=3)
    count = sum(p.numel() for p in model.parameters())
    assert count == 2835
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_model.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'training.model'`.

- [ ] **Step 3: Implement `training/model.py`**

```python
import torch.nn as nn


class TrendCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(36, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

Param count check: Conv1d(1,24,k3) = 1·24·3+24 = 96. Conv1d(24,36,k3) = 24·36·3+36 = 2628. Linear(36,5) = 36·5+5 = 185. Total = 96+2628+185 = 2909 (5-class); with Linear(36,3) = 36·3+3 = 111, total = 2835 (3-class). Both land within ~2% of the ~2,850-param budget from the original brief.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
source venv/bin/activate
pytest tests/test_model.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/model.py tests/test_model.py
git commit -m "Add TrendCNN: small 1D-CNN sized to the ~2,850-param budget"
```

---

## Task 5: Training CLI + iglu smoke test

**Files:**
- Create: `training/train.py`

**Interfaces:**
- Consumes:
  - `data_formatter.base.DataFormatter(config) -> object` with `.train_data`/`.val_data`/`.test_data` pandas DataFrames (columns `id`, `time`, `gl`, `id_segment`) — from the cloned `GlucoBench/` (external, unmodified)
  - `training.dataset.GlucoseTrendDataset` (Task 3)
  - `training.model.TrendCNN` (Task 4)
  - `training.labeling.{FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP}` (Task 2)
- Produces: a runnable CLI (`python -m training.train --dataset {iglu,weinstock} --classes {5,3} --epochs N`) that writes `results/{dataset}_{classes}class_report.json`.

This task is an integration entry point, not pure logic — it is verified by actually running it against the small iglu dataset rather than by unit tests.

- [ ] **Step 1: Implement `training/train.py`**

```python
"""CLI entry point for training the glucose trend classifier."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
GLUCOBENCH_ROOT = GLUCOEDGE_ROOT / "GlucoBench"
sys.path.insert(0, str(GLUCOBENCH_ROOT))

from data_formatter.base import DataFormatter  # noqa: E402

from training.dataset import GlucoseTrendDataset  # noqa: E402
from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP  # noqa: E402
from training.model import TrendCNN  # noqa: E402


def load_formatter(dataset: str) -> DataFormatter:
    config_path = GLUCOBENCH_ROOT / "config" / f"{dataset}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["data_csv_path"] = str(GLUCOBENCH_ROOT / "raw_data" / f"{dataset}.csv")
    return DataFormatter(config)


def class_weights(train_ds: GlucoseTrendDataset, num_classes: int) -> torch.Tensor:
    labels = np.array(train_ds.labels)
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(train_ds, val_ds, num_classes, epochs, device):
    model = TrendCNN(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_ds, num_classes).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        denom = max(len(val_loader), 1)
        print(f"epoch {epoch + 1}/{epochs}  val_loss={val_loss / denom:.4f}")

    return model


def evaluate(model, test_ds, classes, device):
    loader = DataLoader(test_ds, batch_size=64)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    return classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        labels=list(range(len(classes))),
        output_dict=True,
        zero_division=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iglu", "weinstock"], required=True)
    parser.add_argument("--classes", choices=["5", "3"], default="5")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

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
    report = evaluate(model, test_ds, classes, device)

    results_dir = GLUCOEDGE_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
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

- [ ] **Step 2: Run the iglu smoke test**

```bash
source venv/bin/activate
python -m training.train --dataset iglu --classes 5 --epochs 2
```

Expected: prints non-zero train/val/test window counts, prints 2 epoch lines with decreasing or stable `val_loss`, ends with `wrote results/iglu_5class_report.json`. If train/val/test counts are 0, stop and re-check the `(id, id_segment)` grouping against the actual iglu data before proceeding — do not continue to Task 6 on top of a broken pipeline.

- [ ] **Step 3: Inspect the report structure**

```bash
python -c "import json; print(json.dumps(json.load(open('results/iglu_5class_report.json')), indent=2)[:1000])"
```

Expected: valid JSON with `dataset`, `num_classes`, `classes`, `train_label_counts`, and `classification_report` keys.

- [ ] **Step 4: Commit**

```bash
git add training/train.py
git commit -m "Add training CLI; verified end-to-end on iglu smoke test"
```

(`results/` stays untracked per `.gitignore` — it holds regenerable output, not source.)

---

## Task 6: Weinstock run and the 5-class vs 3-class decision

**Files:**
- Create: `docs/superpowers/specs/<today's-date>-weinstock-class-count-decision.md`

No new source code — this task runs the pipeline built in Tasks 1-5 against the full weinstock config and turns the result into a documented decision, per the spec's decision procedure.

- [ ] **Step 1: Run the weinstock 5-class training**

```bash
source venv/bin/activate
python -m training.train --dataset weinstock --classes 5 --epochs 20
```

Expected: completes (200 patients means more windows and longer runtime than the iglu smoke test — this can take several minutes, that is expected), ends with `wrote results/weinstock_5class_report.json`.

- [ ] **Step 2: Read the counts and recall for the two "fast" classes**

```bash
python -c "
import json
r = json.load(open('results/weinstock_5class_report.json'))
print('train_label_counts:', r['train_label_counts'])
cr = r['classification_report']
print('falling_fast recall:', cr['falling_fast']['recall'], 'support:', cr['falling_fast']['support'])
print('rising_fast recall:', cr['rising_fast']['recall'], 'support:', cr['rising_fast']['support'])
"
```

- [ ] **Step 3: Apply the decision procedure**

Using the numbers from Step 2, collapse to 3-class **only if**, for either `falling_fast` or `rising_fast`:
- `train_label_counts[class] < 200`, **or**
- the test-set recall for that class is `< 0.30` (not meaningfully above the ~20% random-guess baseline for 5 classes)

If neither condition fires for either class, keep the 5-class model and skip Step 4.

- [ ] **Step 4 (only if Step 3 triggered a collapse): Run the weinstock 3-class training**

```bash
source venv/bin/activate
python -m training.train --dataset weinstock --classes 3 --epochs 20
```

Expected: ends with `wrote results/weinstock_3class_report.json`.

- [ ] **Step 5: Write the decision record**

Create `docs/superpowers/specs/<today's-date>-weinstock-class-count-decision.md` (use the actual current date for `<today's-date>`, matching the `YYYY-MM-DD` convention used by the other files in this directory), filling in the real numbers observed in Steps 1-4 — do not leave any bracketed placeholder in the committed file:

```markdown
# Weinstock class-count decision

Ran 5-class training on the full weinstock config (200 patients).

Train window counts by class:
- falling_fast: <actual count from train_label_counts>
- falling: <actual count>
- stable: <actual count>
- rising: <actual count>
- rising_fast: <actual count>

Test-set per-class recall (5-class model):
- falling_fast: <actual recall>
- falling: <actual recall>
- stable: <actual recall>
- rising: <actual recall>
- rising_fast: <actual recall>

Decision: kept 5-class / collapsed to 3-class (state which), because <cite
the specific counts/recall numbers above against the <200-examples and
<30%-recall triggers from the spec's decision procedure>.

<If collapsed: also paste the 3-class classification_report numbers here,
per-class precision/recall/f1 for falling/stable/rising.>
```

- [ ] **Step 6: Commit the decision record**

```bash
git add docs/superpowers/specs/*-weinstock-class-count-decision.md
git commit -m "Document weinstock 5-class vs 3-class decision with observed numbers"
```

---

## Definition of done

- `pytest` passes with all tests from Tasks 2-4 green.
- `results/iglu_5class_report.json` and `results/weinstock_{5,3}class_report.json` (whichever the decision procedure produced) exist locally.
- A committed decision record states, with real numbers, whether GlucoEdge's classifier is 5-class or 3-class going forward — this determines the label set that the next sub-project (LiteRT conversion) and the Android app must match.
