# Accuracy Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Staged study (loss → capacity×window → pooling → ensemble) that either upgrades the bundled delta-anchor models or documents a null result, with one pre-registered held-out read.

**Architecture:** New losses in `training/losses.py`; `TrendCNN` gains a `channels` parameter; one resumable pipeline `experiments/accuracy_sweep.py` (run id `sweep-r1`) reusing `calibration_coverage`/`delta_inputs` helpers. Selection on weinstock val only; `stage_evaluate` refuses to run until `paper/results/calibration-coverage/protocol_sweep_r1.json` is committed and clean.

**Tech Stack:** PyTorch (CPU), LiteRT PT2E INT8 via `conversion/`, numpy bootstrap helpers in `experiments/analyze_calibration_coverage.py`.

**Spec:** `docs/superpowers/specs/2026-10-11-accuracy-sweep-design.md`

## Global Constraints

- Not a medical device: no output string implying clinical validity.
- GlucoBench read-only, never committed; participant ids/timestamps/window keys/per-window predictions stay in gitignored `results/`.
- Selection metric: mean val macro-F1 over seeds (0,1,2); eligible windows only (scored + no-gap-fill).
- held_out/temporal labels read ONLY by `stage_evaluate`, ONLY after protocol commit.
- Training recipe frozen: Adam lr=1e-3, batch 64, 20 epochs, CPU, `torch.manual_seed` + `np.random.seed`.
- Inputs always delta-anchor (`delta_windows(w, anchor=True)`, ANCHOR_SCALE 16).
- Ties within 0.1pp: cheaper arm wins (params, then shorter window, then unpooled, then simpler loss).
- Work on branch `feat/accuracy-sweep`; `pytest` must stay green without GlucoBench cloned.
- All commits end with the session's attribution lines.

## Review Focus

1. macro_f1 on a confusion matrix with an absent class (zero support and zero predictions) must count that class as 0, not NaN — Task 1 test.
2. ordinal_smooth target for boundary classes (0 and 4) must give the full ε to the single neighbor; every target row sums to 1 — Task 2 test.
3. `load_checkpoint` must still load every existing checkpoint (default channels) after channel inference lands — Task 3 test.
4. `pick_winner` on a tie must deterministically prefer the cheaper arm — Task 5 test.
5. `stage_evaluate` with an uncommitted or dirty protocol file must refuse before touching any label — Task 6 test.

---

### Task 1: macro_f1 metric

**Files:**
- Modify: `experiments/analyze_calibration_coverage.py` (add `macro_f1` next to `macro_recall`)
- Test: `tests/test_calibration_analysis.py`

**Interfaces:**
- Produces: `macro_f1(cm) -> float` over a (5,5) confusion-count matrix, sklearn `zero_division=0` semantics. Used by Tasks 5–6 for selection and bootstrap metric.

- [ ] **Step 0: Branch**

```bash
git checkout -b feat/accuracy-sweep
```

- [ ] **Step 1: Write the failing test** (append to `tests/test_calibration_analysis.py`)

```python
def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    y = np.array([0, 0, 1, 2, 2, 3, 4, 4, 4])
    p = np.array([0, 1, 1, 2, 3, 3, 4, 0, 4])
    assert macro_f1(confusion_counts(y, p)) == pytest.approx(
        f1_score(y, p, average="macro", labels=range(5), zero_division=0))


def test_macro_f1_absent_class_counts_zero():
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    y = np.array([0, 0, 1])   # classes 2,3,4 absent and never predicted
    p = np.array([0, 0, 1])
    assert macro_f1(confusion_counts(y, p)) == pytest.approx(2 / 5)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_calibration_analysis.py -k macro_f1 -v`
Expected: FAIL, `ImportError: cannot import name 'macro_f1'`

- [ ] **Step 3: Implement** (in `experiments/analyze_calibration_coverage.py`, after `minority_macro_recall`)

```python
def macro_f1(cm):
    """Macro-averaged F1 from confusion counts; a class with no support and
    no predictions contributes 0 (sklearn zero_division=0 semantics)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_calibration_analysis.py -v`
Expected: PASS (whole file)

- [ ] **Step 5: Commit**

```bash
git add experiments/analyze_calibration_coverage.py tests/test_calibration_analysis.py
git commit -m "feat(analysis): macro-F1 from confusion counts"
```

### Task 2: Loss arms

**Files:**
- Create: `training/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces: `LOSSES = ("ce", "ordinal_smooth", "distance_ce", "focal")`; `make_criterion(name: str, weights: torch.Tensor) -> callable(logits, targets) -> scalar tensor`; constants `EPSILON=0.15`, `LAMBDA=0.5`, `GAMMA=2.0`. Consumed by Task 4 (`train_model(criterion=...)`) and Task 5.

- [ ] **Step 1: Write the failing tests**

```python
"""Hand-checked cases for the sweep's loss arms (spec Stage A)."""
import pytest
import torch
import torch.nn.functional as F

from training.losses import (EPSILON, GAMMA, LAMBDA, LOSSES, _smeared_targets,
                             make_criterion)

W = torch.tensor([1.0, 2.0, 0.5, 1.0, 1.0])
LOGITS = torch.tensor([[2.0, 0.1, -1.0, 0.0, 0.5],
                       [0.0, 3.0, 0.2, -0.5, 1.0]])
Y = torch.tensor([0, 2])


def test_ce_matches_torch_weighted_ce():
    loss = make_criterion("ce", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(
        F.cross_entropy(LOGITS, Y, weight=W).item())


def test_smeared_targets_rows_sum_to_one_boundary_gets_full_epsilon():
    q = _smeared_targets(5, EPSILON)
    assert torch.allclose(q.sum(dim=1), torch.ones(5))
    assert q[0, 1].item() == pytest.approx(EPSILON)      # boundary: one neighbor
    assert q[2, 1].item() == pytest.approx(EPSILON / 2)  # interior: split
    assert q[2, 3].item() == pytest.approx(EPSILON / 2)
    assert q[4, 3].item() == pytest.approx(EPSILON)


def test_ordinal_smooth_hand_computed():
    q = _smeared_targets(5, EPSILON)
    logp = F.log_softmax(LOGITS, dim=1)
    per = -(q[Y] * logp).sum(dim=1)
    w = W[Y]
    expected = (per * w).sum() / w.sum()
    loss = make_criterion("ordinal_smooth", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(expected.item())


def test_distance_ce_adds_expected_ordinal_distance():
    p = F.softmax(LOGITS, dim=1)
    d = (torch.arange(5.0)[None, :] - Y[:, None].float()).abs()
    expected = (F.cross_entropy(LOGITS, Y, weight=W)
                + LAMBDA * (p * d).sum(dim=1).mean())
    loss = make_criterion("distance_ce", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(expected.item())


def test_focal_hand_computed_and_vanishes_at_confident_correct():
    logp = F.log_softmax(LOGITS, dim=1).gather(1, Y[:, None]).squeeze(1)
    pt = logp.exp()
    w = W[Y]
    expected = (-w * (1 - pt) ** GAMMA * logp).sum() / w.sum()
    loss = make_criterion("focal", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(expected.item())
    sure = torch.tensor([[50.0, 0.0, 0.0, 0.0, 0.0]])
    assert make_criterion("focal", W)(sure, torch.tensor([0])).item() == pytest.approx(0.0, abs=1e-6)


def test_unknown_loss_raises():
    with pytest.raises(ValueError):
        make_criterion("nope", W)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'training.losses'`

- [ ] **Step 3: Implement `training/losses.py`**

```python
"""Loss arms for the accuracy sweep (spec 2026-10-11, Stage A). Every arm
keeps the plain 5-logit softmax head; `weights` is the balanced class-weight
vector from training.train.class_weights. Weighted arms normalize by the
batch's summed true-class weights, matching torch CrossEntropyLoss."""
import torch
import torch.nn.functional as F

LOSSES = ("ce", "ordinal_smooth", "distance_ce", "focal")
EPSILON = 0.15  # ordinal_smooth mass moved to ordinal neighbors
LAMBDA = 0.5    # distance_ce penalty weight
GAMMA = 2.0     # focal focusing exponent


def _smeared_targets(num_classes, epsilon):
    """Row y: 1-epsilon on y, epsilon split over its ordinal neighbors
    (a boundary class gives the whole epsilon to its single neighbor)."""
    q = torch.zeros(num_classes, num_classes)
    for y in range(num_classes):
        neighbors = [j for j in (y - 1, y + 1) if 0 <= j < num_classes]
        q[y, y] = 1.0 - epsilon
        for j in neighbors:
            q[y, j] = epsilon / len(neighbors)
    return q


def make_criterion(name, weights):
    weights = weights.float()
    if name == "ce":
        return torch.nn.CrossEntropyLoss(weight=weights)
    if name == "ordinal_smooth":
        q = _smeared_targets(len(weights), EPSILON)

        def ordinal_smooth(logits, targets):
            logp = F.log_softmax(logits, dim=1)
            per = -(q.to(logits.device)[targets] * logp).sum(dim=1)
            w = weights.to(logits.device)[targets]
            return (per * w).sum() / w.sum()
        return ordinal_smooth
    if name == "distance_ce":
        ce = torch.nn.CrossEntropyLoss(weight=weights)
        dist = torch.arange(len(weights), dtype=torch.float32)

        def distance_ce(logits, targets):
            p = F.softmax(logits, dim=1)
            d = (dist.to(logits.device)[None, :] - targets[:, None].float()).abs()
            return ce(logits, targets) + LAMBDA * (p * d).sum(dim=1).mean()
        return distance_ce
    if name == "focal":
        def focal(logits, targets):
            logp = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
            pt = logp.exp()
            w = weights.to(logits.device)[targets]
            return (-w * (1.0 - pt) ** GAMMA * logp).sum() / w.sum()
        return focal
    raise ValueError(f"unknown loss arm: {name}")
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_losses.py -v`
Expected: PASS 6/6

- [ ] **Step 5: Commit**

```bash
git add training/losses.py tests/test_losses.py
git commit -m "feat(training): four loss arms for the accuracy sweep"
```

### Task 3: TrendCNN channels + checkpoint channel inference

**Files:**
- Modify: `training/model.py`
- Modify: `conversion/common.py` (`load_checkpoint`)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TrendCNN(num_classes=5, channels=(24, 36))` — default identical to current architecture; `load_checkpoint(path, num_classes)` infers channels from the state dict (`conv1.weight.shape[0]`, `conv2.weight.shape[0]`), so every existing caller keeps working unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_model.py`)

```python
def test_default_channels_match_current_architecture():
    model = TrendCNN(num_classes=5)
    assert sum(p.numel() for p in model.parameters()) == 2909


def test_wider_channels():
    import torch
    model = TrendCNN(num_classes=5, channels=(48, 72))
    assert model.conv1.out_channels == 48 and model.conv2.out_channels == 72
    assert model(torch.zeros(2, 1, 24)).shape == (2, 5)  # length-agnostic


def test_load_checkpoint_infers_channels(tmp_path):
    import torch
    from conversion.common import load_checkpoint
    for channels in ((24, 36), (96, 144)):
        src = TrendCNN(num_classes=5, channels=channels)
        p = tmp_path / f"{channels[0]}.pt"
        torch.save(src.state_dict(), p)
        loaded = load_checkpoint(p, 5)
        assert loaded.conv1.out_channels == channels[0]
        assert loaded.conv2.out_channels == channels[1]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_model.py -v`
Expected: FAIL, `TypeError: __init__() got an unexpected keyword argument 'channels'`

- [ ] **Step 3: Implement**

`training/model.py`:

```python
import torch.nn as nn


class TrendCNN(nn.Module):
    def __init__(self, num_classes: int = 5, channels=(24, 36)):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv1d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

`conversion/common.py` — replace the body of `load_checkpoint` so it infers channels before constructing (keep its existing eval/map_location behavior):

```python
def load_checkpoint(checkpoint_path: Path, num_classes: int) -> TrendCNN:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    channels = (state["conv1.weight"].shape[0], state["conv2.weight"].shape[0])
    model = TrendCNN(num_classes=num_classes, channels=channels)
    model.load_state_dict(state)
    model.eval()
    return model
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_model.py tests/test_train.py -v`
Expected: PASS (including pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add training/model.py conversion/common.py tests/test_model.py
git commit -m "feat(model): parameterize TrendCNN channels; infer channels on checkpoint load"
```

### Task 4: train_model injection

**Files:**
- Modify: `training/train.py:68` (`train_model`)
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `train_model(train_ds, val_ds, num_classes, epochs, device, model=None, criterion=None)` — both default to current behavior (fresh default TrendCNN, weighted CE), so `calibration_coverage`/`delta_inputs` callers are untouched.

- [ ] **Step 1: Write the failing test** (append to `tests/test_train.py`; reuse the file's existing tiny-dataset fixture if one exists, otherwise this self-contained one)

```python
def test_train_model_accepts_injected_model_and_criterion():
    import torch
    from training.model import TrendCNN
    from training.train import train_model

    class Tiny:
        labels = [0, 1, 2, 3, 4, 0, 1, 2]
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, i):
            torch.manual_seed(i)
            return torch.randn(1, 12), self.labels[i]

    calls = []
    def spy_criterion(logits, targets):
        calls.append(1)
        return torch.nn.functional.cross_entropy(logits, targets)

    injected = TrendCNN(num_classes=5, channels=(48, 72))
    out = train_model(Tiny(), Tiny(), 5, 1, torch.device("cpu"),
                      model=injected, criterion=spy_criterion)
    assert out is injected
    assert calls  # criterion actually used
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_train.py::test_train_model_accepts_injected_model_and_criterion -v`
Expected: FAIL, `TypeError: train_model() got an unexpected keyword argument 'model'`

- [ ] **Step 3: Implement** — change only the first two lines of `train_model`:

```python
def train_model(train_ds, val_ds, num_classes, epochs, device, model=None, criterion=None):
    model = (model if model is not None else TrendCNN(num_classes=num_classes)).to(device)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_ds, num_classes).to(device))
```

(rest of the function unchanged)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_train.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/train.py tests/test_train.py
git commit -m "feat(training): train_model accepts injected model and criterion"
```

### Task 5: Sweep pipeline — arms, views, precheck/train/select stages

**Files:**
- Create: `experiments/accuracy_sweep.py`
- Test: `tests/test_accuracy_sweep.py`

**Interfaces:**
- Consumes: `make_criterion`/`LOSSES` (Task 2), `TrendCNN(channels=...)` (Task 3), `train_model(model=, criterion=)` (Task 4), `macro_f1` (Task 1), `DeltaView` from `experiments.delta_inputs`, `_eligible`, `_run_dir`, `_write_json` from `experiments.calibration_coverage`, `class_weights` from `training.train`.
- Produces: pure helpers `arm(...) -> dict`, `stage_a_arms()`, `stage_b_arms(winner_loss, include_36)`, `param_count(channels)`, `cost_key(arm)`, `pick_winner(arms, means)`, `ConcatView`, `paired_rows(keys_a, keys_b)`; CLI stages `precheck`, `train --stage A|B|C|D`, `select --stage A|B|C|D`. Run layout: `results/calibration-coverage/sweep-r1/{precheck.json, checkpoints/<arm>_seed<k>.pt, selection/<stage>.json}`. `selection/<stage>.json` = `{"table": {arm_name: {"per_seed": {...}, "mean": float}}, "winner": <arm dict>}`; Stage D adds `{"carry_ensemble": bool, "best_seed": int, "ensemble_val_f1": float, "best_seed_val_f1": float}` and its winner dict gains `"ensemble": bool`.

- [ ] **Step 1: Write the failing tests for the pure parts**

```python
"""Pure-logic tests for the accuracy sweep (no GlucoBench needed)."""
import numpy as np
import pytest

from experiments.accuracy_sweep import (ConcatView, arm, cost_key,
                                        paired_rows, param_count,
                                        pick_winner, stage_a_arms,
                                        stage_b_arms)


def test_param_count_matches_default_architecture():
    assert param_count((24, 36)) == 2909


def test_stage_a_arms_cover_all_losses_at_baseline():
    arms = stage_a_arms()
    assert [a["loss"] for a in arms] == ["ce", "ordinal_smooth", "distance_ce", "focal"]
    assert all(a["channels"] == [24, 36] and a["length"] == 12 for a in arms)


def test_stage_b_arms_exclude_reused_baseline_cell():
    arms = stage_b_arms("focal", include_36=False)
    names = {a["name"] for a in arms}
    assert "B_c24x36_L12" not in names
    assert len(arms) == 5  # 3 channel grids x 2 lengths - reused cell
    arms36 = stage_b_arms("focal", include_36=True)
    assert len(arms36) == 8


def test_pick_winner_tie_prefers_cheaper_arm():
    a = arm("A_ce", "ce", (24, 36), 12)
    b = arm("B_big", "ce", (96, 144), 24)
    # b higher but within the 0.1pp tie band -> cheaper a wins
    assert pick_winner([a, b], {"A_ce": 0.5000, "B_big": 0.5009}) == a
    # outside the band -> b wins
    assert pick_winner([a, b], {"A_ce": 0.5000, "B_big": 0.5011}) == b


def test_cost_key_orders_params_length_pooled_loss():
    cheap = arm("x", "ce", (24, 36), 12)
    pooled = arm("y", "ce", (24, 36), 12, pooled=True)
    fancy_loss = arm("z", "focal", (24, 36), 12)
    assert cost_key(cheap) < cost_key(pooled)
    assert cost_key(cheap) < cost_key(fancy_loss)


def test_concat_view_concatenates_labels_and_items():
    class V:
        def __init__(self, labels):
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, k):
            return ("item", self.labels[k])
    cv = ConcatView([V([0, 1]), V([2])])
    assert cv.labels == [0, 1, 2]
    assert len(cv) == 3
    assert cv[2] == ("item", 2)


def test_paired_rows_intersects_on_keys_in_a_order():
    ka = [("p1", "t1"), ("p1", "t2"), ("p2", "t1")]
    kb = [("p2", "t1"), ("p1", "t1")]
    ia, ib = paired_rows(ka, kb)
    assert ia == [0, 2]
    assert ib == [1, 0]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_accuracy_sweep.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'experiments.accuracy_sweep'`

- [ ] **Step 3: Implement `experiments/accuracy_sweep.py`** (pure parts + precheck/train/select; evaluate/convert/analyze land in Task 6)

```python
"""Staged accuracy sweep (spec 2026-10-11). Run id sweep-r1.

Greedy stages A (loss) -> B (capacity x window) -> C (pooling) ->
D (ensemble), every selection on eligible weinstock val macro-F1 only.
held_out/temporal are touched by stage_evaluate alone, which refuses to
run until the sweep protocol file is committed and clean. All stages are
resumable: existing checkpoints/selections are skipped."""
import json
import subprocess
import time

import numpy as np

from experiments.calibration_coverage import (GLUCOEDGE_ROOT, _eligible,
                                              _run_dir, _write_json)
from training.deltas import delta_windows
from training.losses import LOSSES

ARM_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
POOL_COHORTS = ("weinstock", "dubosson", "hall")
CHANNEL_GRID = ((24, 36), (48, 72), (96, 144))
TIE_BAND = 0.001        # 0.1pp tie rule
ENSEMBLE_MARGIN = 0.005  # Stage D carry gate
EPOCHS = 20
PROTOCOL = GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage" / "protocol_sweep_r1.json"
CHAMPION = {  # bundled delta-anchor pair (D12 ship)
    "float": GLUCOEDGE_ROOT / "android/app/src/main/assets/trend_float.tflite",
    "int8": GLUCOEDGE_ROOT / "android/app/src/main/assets/trend_int8.tflite",
    "length": 12,
}


# --- arms -------------------------------------------------------------------


def arm(name, loss, channels, length, pooled=False):
    return {"name": name, "loss": loss, "channels": list(channels),
            "length": length, "pooled": pooled}


def stage_a_arms():
    return [arm(f"A_{l}", l, (24, 36), 12) for l in LOSSES]


def stage_b_arms(winner_loss, include_36):
    lengths = [12, 24] + ([36] if include_36 else [])
    return [arm(f"B_c{c1}x{c2}_L{n}", winner_loss, (c1, c2), n)
            for c1, c2 in CHANNEL_GRID for n in lengths
            if (c1, c2, n) != (24, 36, 12)]  # that cell is Stage A's winner, reused


def param_count(channels, num_classes=5):
    c1, c2 = channels
    return (3 * c1 + c1) + (3 * c1 * c2 + c2) + (c2 * num_classes + num_classes)


def cost_key(a):
    return (param_count(a["channels"]), a["length"], a["pooled"],
            LOSSES.index(a["loss"]), a.get("ensemble", False))


def pick_winner(arms, means):
    best = max(means[a["name"]] for a in arms)
    candidates = [a for a in arms if means[a["name"]] >= best - TIE_BAND]
    return min(candidates, key=cost_key)


# --- data views ---------------------------------------------------------------


class ConcatView:
    """Concatenation of DeltaView-like datasets; exposes .labels for
    class_weights."""

    def __init__(self, views):
        self.views = views
        self.labels = [l for v in views for l in v.labels]
        self._index = [(vi, k) for vi, v in enumerate(views)
                       for k in range(len(v))]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, k):
        vi, j = self._index[k]
        return self.views[vi][j]


_DATA_CACHE = {}


def _data(dataset, length):
    """(datasets dict, owner, no_gap_fill) for one cohort at one window
    length; cached, GlucoBench loaded lazily."""
    key = (dataset, length)
    if key not in _DATA_CACHE:
        from experiments.data_availability import load_cohort_context
        from training.dataset import GlucoseTrendDataset
        from training.labeling import FIVE_CLASSES
        formatter, owner, _, no_gap_fill = load_cohort_context(dataset)
        ds = {s: GlucoseTrendDataset(getattr(formatter, f"{s}_data"),
                                     classes=FIVE_CLASSES, input_length=length)
              for s in ("train", "val", "test")}
        _DATA_CACHE[key] = (ds, owner, no_gap_fill)
    return _DATA_CACHE[key]


def _train_views(a):
    from experiments.delta_inputs import DeltaView
    cohorts = POOL_COHORTS if a["pooled"] else ("weinstock",)
    views = []
    for c in cohorts:
        ds, owner, ngf = _data(c, a["length"])
        views.append(DeltaView(ds["train"], _eligible(ds["train"], owner, ngf, "train"), True))
    train_v = views[0] if len(views) == 1 else ConcatView(views)
    wds, wowner, wngf = _data("weinstock", a["length"])
    val_v = DeltaView(wds["val"], _eligible(wds["val"], wowner, wngf, "val"), True)
    return train_v, val_v


# --- pairing (used by analyze; pure) -----------------------------------------


def paired_rows(keys_a, keys_b):
    """Row indices pairing two window sets on (participant, anchor_time)
    keys, in keys_a order; windows only one side can score are dropped."""
    pos = {k: i for i, k in enumerate(keys_b)}
    ia = [i for i, k in enumerate(keys_a) if k in pos]
    ib = [pos[keys_a[i]] for i in ia]
    return ia, ib


# --- stage: precheck ----------------------------------------------------------


def stage_precheck(run_id):
    counts = {}
    for length in (12, 24, 36):
        ds, owner, ngf = _data("weinstock", length)
        counts[str(length)] = {s: len(_eligible(ds[s], owner, ngf, s))
                               for s in ("train", "val")}
    include_36 = counts["36"]["train"] >= 0.8 * counts["12"]["train"]
    out = {"counts": counts, "include_36": include_36,
           "rule": "length 36 joins Stage B iff its eligible train count is >= 80% of length 12's"}
    _write_json(_run_dir(run_id) / "precheck.json", out)
    print(json.dumps(out, indent=1))


# --- stage: train ---------------------------------------------------------------


def _read_selection(run, stage):
    return json.loads((run / "selection" / f"{stage}.json").read_text())


def _arms_for(run, stage):
    if stage == "A":
        return stage_a_arms(), ARM_SEEDS
    if stage == "B":
        winner = _read_selection(run, "A")["winner"]
        pre = json.loads((run / "precheck.json").read_text())
        return stage_b_arms(winner["loss"], pre["include_36"]), ARM_SEEDS
    if stage == "C":
        winner = _read_selection(run, "B")["winner"]
        return [dict(winner, name="C_pooled", pooled=True)], ARM_SEEDS
    if stage == "D":
        winner = _read_selection(run, "C")["winner"]
        return [winner], tuple(s for s in FINAL_SEEDS if s not in ARM_SEEDS)
    raise ValueError(stage)


def stage_train(run_id, stage):
    import torch

    from training.losses import make_criterion
    from training.model import TrendCNN
    from training.train import class_weights, train_model

    run = _run_dir(run_id)
    arms, seeds = _arms_for(run, stage)
    out = run / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    for a in arms:
        train_v, val_v = _train_views(a)
        for seed in seeds:
            ckpt = out / f"{a['name']}_seed{seed}.pt"
            if ckpt.exists():
                print(f"skip {ckpt.name}: exists")
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = TrendCNN(num_classes=5, channels=tuple(a["channels"]))
            criterion = make_criterion(a["loss"], class_weights(train_v, 5))
            model = train_model(train_v, val_v, 5, EPOCHS, torch.device("cpu"),
                                model=model, criterion=criterion)
            torch.save(model.state_dict(), ckpt)
            _write_json(out / f"{a['name']}_seed{seed}.json", {
                "arm": a, "seed": seed, "epochs": EPOCHS,
                "train_windows": len(train_v), "val_windows": len(val_v),
                "duration_s": round(time.time() - t0, 1)})
            print(f"{a['name']} seed {seed} done in {time.time() - t0:.0f}s")


# --- stage: select ---------------------------------------------------------------


def _val_logits(run, a, seed):
    import torch

    from conversion.common import load_checkpoint

    wds, owner, ngf = _data("weinstock", a["length"])
    idx = _eligible(wds["val"], owner, ngf, "val")
    windows = delta_windows(np.stack([wds["val"].windows[i] for i in idx]), True)
    model = load_checkpoint(run / "checkpoints" / f"{a['name']}_seed{seed}.pt", 5)
    with torch.no_grad():
        logits = model(torch.from_numpy(windows).float().unsqueeze(1)).numpy()
    y = np.array([wds["val"].labels[i] for i in idx])
    return logits, y


def _val_f1(run, a, seed):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    logits, y = _val_logits(run, a, seed)
    return macro_f1(confusion_counts(y, logits.argmax(axis=1)))


def stage_select(run_id, stage):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1

    run = _run_dir(run_id)
    (run / "selection").mkdir(exist_ok=True)
    arms, _ = _arms_for(run, stage)
    if stage == "B":  # Stage A's winner competes as the reused baseline cell
        arms = arms + [_read_selection(run, "A")["winner"]]
    if stage == "C":
        arms = arms + [_read_selection(run, "B")["winner"]]

    if stage in ("A", "B", "C"):
        table, means = {}, {}
        for a in arms:
            per_seed = {str(s): _val_f1(run, a, s) for s in ARM_SEEDS}
            means[a["name"]] = float(np.mean(list(per_seed.values())))
            table[a["name"]] = {"per_seed": per_seed, "mean": means[a["name"]]}
        winner = pick_winner(arms, means)
        out = {"stage": stage, "table": table, "winner": winner}
    else:  # D: ensemble vs best single seed, all on val
        a = arms[0]
        per_seed, logit_stack, y = {}, [], None
        for s in FINAL_SEEDS:
            logits, y = _val_logits(run, a, s)
            logit_stack.append(logits)
            per_seed[str(s)] = macro_f1(confusion_counts(y, logits.argmax(axis=1)))
        best_seed = max(per_seed, key=per_seed.get)
        ens_preds = np.mean(logit_stack, axis=0).argmax(axis=1)
        ens_f1 = macro_f1(confusion_counts(y, ens_preds))
        carry = ens_f1 - per_seed[best_seed] >= ENSEMBLE_MARGIN
        out = {"stage": "D", "table": {"per_seed": per_seed},
               "ensemble_val_f1": ens_f1, "best_seed_val_f1": per_seed[best_seed],
               "best_seed": int(best_seed), "carry_ensemble": bool(carry),
               "winner": dict(a, ensemble=bool(carry))}
    _write_json(run / "selection" / f"{stage}.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
```

(CLI `main()` is added in Task 6 with the remaining stages.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_accuracy_sweep.py -v`
Expected: PASS 7/7

- [ ] **Step 5: Run whole suite (no GlucoBench needed)**

Run: `pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add experiments/accuracy_sweep.py tests/test_accuracy_sweep.py
git commit -m "feat(experiments): accuracy sweep arms, views, precheck/train/select stages"
```

### Task 6: Sweep pipeline — calib/convert/evaluate/analyze + protocol guard

**Files:**
- Modify: `experiments/accuracy_sweep.py`
- Test: `tests/test_accuracy_sweep.py`

**Interfaces:**
- Consumes: Task 5's module; `build_pool`/`select` from `conversion.calibration`; `convert_float`/`convert_int8` from `conversion.convert`; `paired_cluster_bootstrap`, `confusion_counts`, `macro_f1` from `experiments.analyze_calibration_coverage`.
- Produces: `protocol_is_committed() -> bool`; `gate_results(y, pids, cand_int8, cand_float, champ_int8) -> dict`; CLI stages `calib`, `convert`, `evaluate`, `analyze`. Eval layout per component (`held_out`, `temporal`): `eval/<comp>/winner_meta.json` (labels, participants, keys), `eval/<comp>/champion_meta.json` (same, at length 12), `eval/<comp>/<arm>_seed<k>_{float,int8_<arm>}_logits.npy`, `eval/<comp>/champion_{float,int8}_preds.npy`. `analysis/sweep_aggregate.json` carries selection trail + confirmatory contrast + gates.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_protocol_guard_refuses_untracked_file(tmp_path, monkeypatch):
    import subprocess
    import experiments.accuracy_sweep as sweep
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    p = tmp_path / "protocol_sweep_r1.json"
    monkeypatch.setattr(sweep, "PROTOCOL", p)
    assert not sweep.protocol_is_committed()      # missing
    p.write_text("{}")
    assert not sweep.protocol_is_committed()      # untracked
    subprocess.run(["git", "-C", str(tmp_path), "add", p.name], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-qm", "freeze"], check=True)
    assert sweep.protocol_is_committed()          # tracked + clean
    p.write_text('{"dirty": true}')
    assert not sweep.protocol_is_committed()      # dirty


def test_gate_results_known_outcomes():
    import experiments.accuracy_sweep as sweep
    y = np.array([0, 1, 2, 3, 4] * 20)
    pids = np.array([f"p{i % 4}" for i in range(100)])
    perfect = y.copy()
    worse = y.copy(); worse[:30] = (worse[:30] + 1) % 5
    g = sweep.gate_results(y, pids, cand_int8=perfect, cand_float=perfect,
                           champ_int8=worse, n_boot=200)
    assert g["gate1_beats_champion"] is True
    assert g["gate2_int8_within_half_pp_of_float"] is True
    g2 = sweep.gate_results(y, pids, cand_int8=worse, cand_float=perfect,
                            champ_int8=perfect, n_boot=200)
    assert g2["gate1_beats_champion"] is False
    assert g2["gate2_int8_within_half_pp_of_float"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_accuracy_sweep.py -k "protocol or gate" -v`
Expected: FAIL, `AttributeError: ... has no attribute 'protocol_is_committed'`

- [ ] **Step 3: Implement** (append to `experiments/accuracy_sweep.py`)

```python
# --- stage: calib (selections for the winner's window length) -----------------


def stage_calib(run_id):
    from conversion.calibration import build_pool, select

    run = _run_dir(run_id)
    winner = _read_selection(run, "D")["winner"]
    wds, owner, ngf = _data("weinstock", winner["length"])
    val_idx = _eligible(wds["val"], owner, ngf, "val")
    pool = build_pool(wds["val"], val_idx)
    out = run / "selections"
    out.mkdir(exist_ok=True)
    for strategy, draw in (("sequential", 0), ("uniform_random", 101)):
        path = out / f"{strategy}_200_d{draw}.json"
        if path.exists():
            continue
        sel = select(pool, strategy, 200, seed=draw)
        sel["draw"] = draw
        _write_json(path, sel)
        print(f"selected {path.name}")


# --- stage: convert ---------------------------------------------------------------


def stage_convert(run_id):
    import torch

    from conversion.common import load_checkpoint
    from conversion.convert import convert_float, convert_int8

    run = _run_dir(run_id)
    winner = _read_selection(run, "D")["winner"]
    wds, _, _ = _data("weinstock", winner["length"])
    art = run / "artifacts"
    art.mkdir(exist_ok=True)
    sample = torch.zeros(1, 1, winner["length"], dtype=torch.float32)
    for seed in FINAL_SEEDS:
        model = load_checkpoint(run / "checkpoints" / f"{winner['name']}_seed{seed}.pt", 5)
        fpath = art / f"winner_seed{seed}_float.tflite"
        if not fpath.exists():
            convert_float(model, sample, fpath)
            print(f"converted {fpath.name}")
        for sel_path in sorted((run / "selections").glob("*_200_d*.json")):
            ipath = art / f"winner_seed{seed}_int8_{sel_path.stem}.tflite"
            if ipath.exists():
                continue
            sel = json.loads(sel_path.read_text())
            ws = np.stack([wds["val"].windows[i] for i in sel["indices"]])
            calib = [torch.from_numpy(w[None, None, :]) for w in delta_windows(ws, True)]
            timings = convert_int8(model, sample, calib, ipath)
            _write_json(ipath.with_suffix(".json"), {"selection": sel_path.stem, **timings})
            print(f"converted {ipath.name}")


# --- stage: evaluate (locked gate: the one held-out read) ----------------------


def protocol_is_committed():
    """True iff PROTOCOL is git-tracked and identical to its committed
    content."""
    if not PROTOCOL.exists():
        return False
    cwd = str(PROTOCOL.parent)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", PROTOCOL.name],
                             cwd=cwd, capture_output=True)
    if tracked.returncode != 0:
        return False
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", PROTOCOL.name],
                           cwd=cwd, capture_output=True)
    return dirty.returncode == 0


def _tflite_logits(path, windows):
    """Float logits for delta windows of any length; dequantizes INT8 IO."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    logits = np.empty((len(windows), 5), dtype=np.float32)
    is_int8 = inp["dtype"] == np.int8
    scale, zp = inp["quantization"]
    o_scale, o_zp = out["quantization"]
    for k, w in enumerate(windows):
        x = w.reshape(1, 1, -1).astype(np.float32)
        if is_int8:
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / scale) + zp, info.min, info.max).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] == np.int8:
            y = (y - o_zp) * o_scale
        logits[k] = y
    return logits


def _component_windows(length, split):
    ds, owner, ngf = _data("weinstock", length)
    idx = _eligible(ds["test"], owner, ngf, split)
    t = ds["test"]
    keys = [[str(t.participants[i]), str(t.anchor_times[i])] for i in idx]
    meta = {"labels": [int(t.labels[i]) for i in idx],
            "participants": [str(t.participants[i]) for i in idx],
            "keys": keys}
    windows = delta_windows(np.stack([t.windows[i] for i in idx]), True)
    return windows, meta


def stage_evaluate(run_id):
    if not protocol_is_committed():
        raise SystemExit("REFUSED: commit paper/results/calibration-coverage/"
                         "protocol_sweep_r1.json (clean) before evaluate")
    run = _run_dir(run_id)
    winner = _read_selection(run, "D")["winner"]
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        base = run / "eval" / comp
        base.mkdir(parents=True, exist_ok=True)
        w_windows, w_meta = _component_windows(winner["length"], split)
        if not (base / "winner_meta.json").exists():
            _write_json(base / "winner_meta.json", w_meta)
        c_windows, c_meta = ((w_windows, w_meta) if winner["length"] == CHAMPION["length"]
                             else _component_windows(CHAMPION["length"], split))
        if not (base / "champion_meta.json").exists():
            _write_json(base / "champion_meta.json", c_meta)
        for seed in FINAL_SEEDS:
            for art in sorted((run / "artifacts").glob(f"winner_seed{seed}_*.tflite")):
                p = base / f"{art.stem}_logits.npy"
                if not p.exists():
                    np.save(p, _tflite_logits(art, w_windows))
                    print(f"{comp}/{art.stem} evaluated")
        for kind in ("float", "int8"):
            p = base / f"champion_{kind}_preds.npy"
            if not p.exists():
                np.save(p, _tflite_logits(CHAMPION[kind], c_windows).argmax(axis=1))
                print(f"{comp}/champion_{kind} evaluated")


# --- stage: analyze ----------------------------------------------------------------


def gate_results(y, pids, cand_int8, cand_float, champ_int8, n_boot=2000):
    from experiments.analyze_calibration_coverage import (
        confusion_counts, macro_f1, paired_cluster_bootstrap)
    contrast = paired_cluster_bootstrap(y, pids, [(cand_int8, champ_int8)],
                                        metric=macro_f1, n_boot=n_boot)
    ci_low = contrast["ci95"][0] if contrast["ci95"] else None
    f_float = macro_f1(confusion_counts(y, cand_float))
    f_int8 = macro_f1(confusion_counts(y, cand_int8))
    return {
        "contrast_int8_vs_champion": contrast,
        "candidate_float_macro_f1": f_float,
        "candidate_int8_macro_f1": f_int8,
        "champion_int8_macro_f1": macro_f1(confusion_counts(y, champ_int8)),
        "gate1_beats_champion": bool(contrast["delta"] is not None
                                     and contrast["delta"] > 0
                                     and ci_low is not None and ci_low > -0.005),
        "gate2_int8_within_half_pp_of_float": bool(f_float - f_int8 <= 0.005),
    }


def stage_analyze(run_id, n_boot=2000):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1

    run = _run_dir(run_id)
    sel_d = _read_selection(run, "D")
    winner = sel_d["winner"]
    out = {"run_id": run_id,
           "selection_trail": {s: _read_selection(run, s) for s in "ABCD"},
           "components": {}}
    ship_arm = "int8_uniform_random_200_d101"
    for comp in ("held_out", "temporal"):
        base = run / "eval" / comp
        w_meta = json.loads((base / "winner_meta.json").read_text())
        c_meta = json.loads((base / "champion_meta.json").read_text())
        ia, ib = paired_rows([tuple(k) for k in w_meta["keys"]],
                             [tuple(k) for k in c_meta["keys"]])
        y = np.array(w_meta["labels"])[ia]
        pids = np.array(w_meta["participants"])[ia]
        champ = np.load(base / "champion_int8_preds.npy")[ib]

        def cand(kind):
            stack = [np.load(base / f"winner_seed{s}_{kind}_logits.npy")
                     for s in FINAL_SEEDS]
            if winner.get("ensemble"):
                return np.mean(stack, axis=0).argmax(axis=1)[ia]
            return stack[sel_d["best_seed"]].argmax(axis=1)[ia]

        per_seed = {str(s): {
            k: macro_f1(confusion_counts(
                np.array(w_meta["labels"]),
                np.load(base / f"winner_seed{s}_{k}_logits.npy").argmax(axis=1)))
            for k in ("float", ship_arm, "int8_sequential_200_d0")}
            for s in FINAL_SEEDS}
        out["components"][comp] = {
            "paired_windows": len(ia),
            "winner_only_windows": len(w_meta["keys"]) - len(ia),
            "per_seed_full_split": per_seed,
            "gates": gate_results(y, pids, cand(ship_arm), cand("float"),
                                  champ, n_boot=n_boot),
        }
    out["ship"] = {
        "candidate": "ensemble of 5 seeds" if winner.get("ensemble")
                     else f"seed {sel_d['best_seed']}",
        "gate3_note": ("ensemble ships only if gate1 point delta >= +1.0pp"
                       if winner.get("ensemble") else "single-model candidate"),
    }
    g = out["components"]["held_out"]["gates"]
    if winner.get("ensemble"):
        out["ship"]["gate3_point_delta_ok"] = bool(
            g["contrast_int8_vs_champion"]["delta"] is not None
            and g["contrast_int8_vs_champion"]["delta"] >= 0.01)
    out["ship"]["all_gates_pass"] = bool(
        g["gate1_beats_champion"] and g["gate2_int8_within_half_pp_of_float"]
        and out["ship"].get("gate3_point_delta_ok", True))
    (run / "analysis").mkdir(exist_ok=True)
    _write_json(run / "analysis" / "sweep_aggregate.json", out)
    print("wrote", run / "analysis" / "sweep_aggregate.json")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["precheck", "train", "select",
                                          "calib", "convert", "evaluate", "analyze"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage-name", choices=list("ABCD"),
                        help="required for train/select")
    args = parser.parse_args()
    if args.stage in ("train", "select"):
        if not args.stage_name:
            parser.error("--stage-name required")
        {"train": stage_train, "select": stage_select}[args.stage](args.run_id, args.stage_name)
    else:
        {"precheck": stage_precheck, "calib": stage_calib,
         "convert": stage_convert, "evaluate": stage_evaluate,
         "analyze": stage_analyze}[args.stage](args.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_accuracy_sweep.py -v && pytest -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep.py tests/test_accuracy_sweep.py
git commit -m "feat(experiments): sweep calib/convert/evaluate/analyze with protocol guard"
```

### Task 7: Run Stage A (loss)

**Files:** none created in-repo (gitignored `results/calibration-coverage/sweep-r1/`).

**Interfaces:**
- Consumes: CLI from Tasks 5–6.
- Produces: `selection/A.json` with 4-arm table + winner.

- [ ] **Step 1: Precheck** (also warms the data cache and records window counts)

Run: `source venv/bin/activate && python -m experiments.accuracy_sweep precheck --run-id sweep-r1`
Expected: prints counts for lengths 12/24/36 and `include_36` true/false; `results/calibration-coverage/sweep-r1/precheck.json` exists.

- [ ] **Step 2: Train Stage A in background** (12 trainings, ~2h)

Run (background): `python -m experiments.accuracy_sweep train --run-id sweep-r1 --stage-name A`
Expected: 12 checkpoints `A_<loss>_seed{0,1,2}.pt`; resumable if interrupted.

- [ ] **Step 3: Select**

Run: `python -m experiments.accuracy_sweep select --run-id sweep-r1 --stage-name A`
Expected: `selection/A.json` written; winner printed. Record winner + table in the ledger.

### Task 8: Run Stage B (capacity × window)

- [ ] **Step 1: Train** (15–24 trainings, several hours, background)

Run: `python -m experiments.accuracy_sweep train --run-id sweep-r1 --stage-name B`
Expected: checkpoints `B_c<..>x<..>_L<..>_seed{0,1,2}.pt` for every grid cell precheck allows.

- [ ] **Step 2: Select**

Run: `python -m experiments.accuracy_sweep select --run-id sweep-r1 --stage-name B`
Expected: `selection/B.json`; table includes the reused Stage A winner row. Ledger the winner.

### Task 9: Run Stage C (pooling)

- [ ] **Step 1: Train** (3 trainings; pooled data loads dubosson + hall contexts)

Run: `python -m experiments.accuracy_sweep train --run-id sweep-r1 --stage-name C`
Expected: `C_pooled_seed{0,1,2}.pt`.

- [ ] **Step 2: Select**

Run: `python -m experiments.accuracy_sweep select --run-id sweep-r1 --stage-name C`
Expected: `selection/C.json`; winner is pooled arm or Stage B winner, on weinstock val only. Ledger it.

### Task 10: Run Stage D (ensemble)

- [ ] **Step 1: Train seeds 3–4 of the winner**

Run: `python -m experiments.accuracy_sweep train --run-id sweep-r1 --stage-name D`
Expected: `<winner>_seed3.pt`, `<winner>_seed4.pt`.

- [ ] **Step 2: Select**

Run: `python -m experiments.accuracy_sweep select --run-id sweep-r1 --stage-name D`
Expected: `selection/D.json` with `carry_ensemble`, `best_seed`, both val F1s. Ledger the decision.

### Task 11: Protocol freeze + confirmatory read

**Files:**
- Create: `paper/results/calibration-coverage/protocol_sweep_r1.json`

- [ ] **Step 1: Calibration selections + artifacts**

Run: `python -m experiments.accuracy_sweep calib --run-id sweep-r1 && python -m experiments.accuracy_sweep convert --run-id sweep-r1`
Expected: 2 selection files; 15 artifacts (5 seeds × float + 2 INT8 arms).

- [ ] **Step 2: Verify evaluate refuses before freeze**

Run: `python -m experiments.accuracy_sweep evaluate --run-id sweep-r1`
Expected: exits with `REFUSED: commit ... protocol_sweep_r1.json`.

- [ ] **Step 3: Write and commit the protocol** — content assembled from the run directory (NO held-out/temporal label has been read yet):

```json
{
  "study": "accuracy-sweep",
  "run_id": "sweep-r1",
  "spec": "docs/superpowers/specs/2026-10-11-accuracy-sweep-design.md",
  "frozen_date_utc": "<today>",
  "glucoedge_commit_at_freeze": "<git rev-parse --short HEAD>",
  "selection_trail": "<the four selection/<stage>.json tables, verbatim>",
  "winner": "<selection/D.json winner dict>",
  "champion": {
    "float_sha256": "c30560f7283e4927a4b63ff0ffdd34f4108e035905b80c4ed4fd01b479dbb07a",
    "int8_sha256": "8590a3d25c409a46659cba3dab97f649c54b57061022e01cae0feec2f5366388"
  },
  "prediction": "winner INT8 (uniform_random_200_d101) >= champion INT8 on held-out macro-F1",
  "primary_endpoint": "held_out paired macro-F1 delta, participant-cluster bootstrap, 2000 draws, seed 20261010",
  "secondary_endpoint": "temporal, same recipe",
  "pairing": "intersection on (participant, anchor_time); winner-only windows reported, excluded from the contrast",
  "gates": {
    "1": "delta > 0 and CI lower bound > -0.5pp",
    "2": "winner INT8 within 0.5pp of winner's own float",
    "3": "ensemble ships only if Stage D margin passed and gate-1 point delta >= +1.0pp"
  }
}
```

```bash
git add paper/results/calibration-coverage/protocol_sweep_r1.json
git commit -m "docs(protocol): freeze accuracy-sweep confirmatory read before any held-out label"
```

- [ ] **Step 4: Evaluate + analyze**

Run: `python -m experiments.accuracy_sweep evaluate --run-id sweep-r1 && python -m experiments.accuracy_sweep analyze --run-id sweep-r1`
Expected: `analysis/sweep_aggregate.json` with gates. Ledger every gate outcome.

### Task 12: Publish results + paper

**Files:**
- Create: `paper/results/calibration-coverage/accuracy_sweep_r1.json`
- Modify: `paper/main.tex`

- [ ] **Step 1: Publish JSON** — copy `analysis/sweep_aggregate.json`, strip anything participant-level (keys/participants lists never leave `results/`; the aggregate only carries tables, contrasts, counts — verify by reading it), add `protocol` pointer field.

- [ ] **Step 2: Paper subsection** — after the delta-inputs subsection: stage winners' val table + confirmatory held-out/temporal contrast with CI + gate outcomes, one table (`tab:sweep`). State val-only selection and the single pre-registered read. Numbers copied from `accuracy_sweep_r1.json` only.

- [ ] **Step 3: Build + check**

Run: `cd paper && tectonic main.tex && pdftoppm -r 10 main.pdf /tmp/sweep_pp && ls /tmp/sweep_pp* | wc -l`
Expected: clean build; page count recorded (≤ 15; trim as in prior phases if it spills).

- [ ] **Step 4: Commit**

```bash
git add paper/results/calibration-coverage/accuracy_sweep_r1.json paper/main.tex paper/main.pdf
git commit -m "docs(paper): accuracy-sweep results - staged selection + pre-registered confirmatory read"
```

### Task 13: Ship decision

Two branches; the ledger records which ran.

**If `ship.all_gates_pass` is false:** add one sentence to MODELS.md ("accuracy sweep sweep-r1 did not beat the bundled pair; see paper") — commit, done. The research PR is the only PR.

**If true:** STOP and tell the human partner the gates passed before executing the ship (it changes bundled assets — the spec's ship PR is a separate, explicitly-surfaced step). The ship PR (separate branch `feat/ship-sweep-models` off main after the research PR merges) contains:
- regenerate goldens + assets together: export winner float + uniform d101 INT8 to `android/app/src/main/assets/`, rerun `conversion/export_golden_vectors.py`, update both apps' pinned sha256s;
- if winner length ≠ 12: replay buffer + `DeltaEncoding` length in `android/.../DeltaEncoding.kt` and `ios/.../DeltaEncoding.swift` + their tests (RED→GREEN);
- if ensemble: K assets + mean-logit averaging in both `TrendClassifier`s + parity pins for every artifact;
- MODELS.md provenance; run Android JVM tests + iOS simulator suite locally before pushing.
