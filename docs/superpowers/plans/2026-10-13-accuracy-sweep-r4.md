# Accuracy Sweep r4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run accuracy sweep r4 — a binding label-noise ceiling measurement (Experiment 0), then staged arms over objective heads (sqrt weights / rate regression / CORN), robustness (LSQ train labels / input noise / conditional mixup), and weight averaging (soup / SWA) — against the shipped r2 champion under the repo's frozen-protocol discipline.

**Architecture:** New pure helpers in `training/` (rate classification, LSQ slope, heads, augmentation, state-dict averaging) + one new stage runner `experiments/accuracy_sweep_r4.py` that imports r2/r3 machinery (TensorView, recalls, smoke, calib, protocol guard, bootstrap). All selection on weinstock val; the single held-out read happens in `stage evaluate` behind the committed-protocol guard; Experiment 0 gates whether any arm trains at all.

**Tech Stack:** Python 3.12, PyTorch (MPS), numpy, ai-edge-litert / litert_torch (conversion), pytest. No GlucoBench/DiaData needed by any unit test.

**Spec:** `docs/superpowers/specs/2026-10-13-accuracy-sweep-r4-design.md`

## Global Constraints

- Not a medical device: no code, comment, or output string implies clinical validity or treatment guidance.
- GlucoBench and DiaData are read-only externals; participant IDs, timestamps, window keys, per-window predictions stay in gitignored `results/`.
- held_out / temporal / PEDAP labels are read only by `stage evaluate`, only after `paper/results/calibration-coverage/protocol_sweep_r4.json` is committed clean; selection uses weinstock val only.
- Training arms run only after `paper/results/calibration-coverage/exp0_sweep_r4.json` is committed clean with `"abort": false`.
- Run id `sweep-r4`; branch `feat/accuracy-sweep-r4`; seeds (0,1,2,3,4); cap 15,000,000 windows ("near 15M" whole-subject overshoot); `SAMPLE_SEED` from r2 (20261012); 20 epochs, Adam lr 1e-3, MPS, frozen smoke batch; bootstrap seed **20261014** defined once as `BOOT_SEED_R4` with a test asserting its use.
- Clinical thresholds and the asymmetric boundary rules (`rate <= -2`, `<= -1`, `< 1`, `< 2`) are never changed; eval labels stay endpoint-based everywhere.
- Champion = currently bundled pair (`experiments.accuracy_sweep.CHAMPION`, read live from android assets; protocol pins float sha `96eef43c…`, INT8 `a887ea4e…`).
- Tie rule at every stage including seed selection: within 0.1pp mean val directional, `cost_key` then list order (for seeds: lowest seed index inside the band).
- `pytest` must pass with no CGM data present.
- Commit messages end with the session's attribution lines (see repo conversation conventions).

## Review Focus

1. Non-finite or absurd rate targets reaching the regression loss (a corrupted horizon row) must be impossible: tensor build asserts all rates finite — test in Task 7.
2. CORN decode on non-monotone logits (e.g. `[-1, +2, -3, +4]`) must count positives, not assume a monotone staircase — test in Task 3.
3. Experiment 0 abort at exactly 2.0pp headroom must PROCEED (abort is strict `<`) — test in Task 8.
4. Mixup collate with a batch of 1 (identity permutation) must return a valid (x, rate) pair, not crash — test in Task 4.
5. Soup over checkpoints with mismatched state-dict keys (stale cache, wrong arch) must refuse loudly, not average garbage — test in Task 6.

---

### Task 1: `classify_rate` + `lsq_slope` in training/labeling.py

**Files:**
- Modify: `training/labeling.py`
- Test: `tests/test_labeling.py` (append)

**Interfaces:**
- Produces: `classify_rate(rate: float) -> str` (one of FIVE_CLASSES; exact boundary rules of today's `label_trend`); `lsq_slope(points: np.ndarray, step_minutes: float = 5.0) -> np.ndarray` ((N, K) mg/dL values at fixed spacing → (N,) least-squares slope in mg/dL/min).
- Consumes: nothing new.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_labeling.py`:

```python
import numpy as np

from training.labeling import classify_rate, label_trend, lsq_slope


def test_classify_rate_boundaries():
    # asymmetric rules: <= on the falling side, < on the rising side
    assert classify_rate(-2.0) == "falling_fast"
    assert classify_rate(-1.0) == "falling"
    assert classify_rate(-0.999) == "stable"
    assert classify_rate(0.999) == "stable"
    assert classify_rate(1.0) == "rising"
    assert classify_rate(1.999) == "rising"
    assert classify_rate(2.0) == "rising_fast"


def test_classify_rate_matches_label_trend():
    for last, future in [(100, 130), (100, 70), (100, 101), (100, 85), (100, 115)]:
        rate = (future - last) / 15.0
        assert classify_rate(rate) == label_trend(last, future)


def test_lsq_slope_hand_computed():
    # exact line: 100, 105, 110, 115 over 0/5/10/15 min -> slope 1.0 mg/dL/min
    pts = np.array([[100.0, 105.0, 110.0, 115.0],
                    [100.0, 100.0, 100.0, 100.0],
                    [120.0, 110.0, 100.0, 90.0]], dtype=np.float32)
    s = lsq_slope(pts)
    assert np.allclose(s, [1.0, 0.0, -2.0], atol=1e-6)


def test_lsq_slope_noisy_vs_endpoint():
    # noise on an interior point moves the endpoint rate, not the LSQ rate sign
    pts = np.array([[100.0, 112.0, 101.0, 102.0]], dtype=np.float32)
    endpoint = (pts[0, -1] - pts[0, 0]) / 15.0
    assert abs(lsq_slope(pts)[0] - endpoint) > 0.0  # estimators differ
```

- [ ] **Step 2: Run tests, verify the new ones fail**

Run: `venv/bin/python -m pytest tests/test_labeling.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'classify_rate'`.

- [ ] **Step 3: Implement**

In `training/labeling.py`, refactor `label_trend` to delegate (behavior byte-identical) and add `lsq_slope`:

```python
import numpy as np


def classify_rate(rate: float) -> str:
    """Rate in mg/dL/min -> one of FIVE_CLASSES. Boundary rules are the
    clinical anchors: <= on the falling side, < on the rising side."""
    if rate <= -2:
        return "falling_fast"
    elif rate <= -1:
        return "falling"
    elif rate < 1:
        return "stable"
    elif rate < 2:
        return "rising"
    return "rising_fast"


def label_trend(last_value: float, future_value: float, horizon_minutes: float = 15.0) -> str:
    # last_value/future_value are mg/dL (GlucoBench leaves `gl` unscaled), so
    # rate is mg/dL per minute - the thresholds assume that unit.
    return classify_rate((future_value - last_value) / horizon_minutes)


def lsq_slope(points, step_minutes: float = 5.0):
    """(N, K) equally-spaced mg/dL readings -> (N,) least-squares slope in
    mg/dL/min. Closed form: sum((t-tbar)(g-gbar)) / sum((t-tbar)^2)."""
    g = np.asarray(points, dtype=np.float64)
    t = np.arange(g.shape[1], dtype=np.float64) * step_minutes
    tc = t - t.mean()
    return ((g - g.mean(axis=1, keepdims=True)) * tc).sum(axis=1) / (tc ** 2).sum()
```

(The module keeps `FIVE_CLASSES` / `THREE_CLASSES` / `THREE_CLASS_MAP` unchanged; numpy import is new.)

- [ ] **Step 4: Run the full labeling suite**

Run: `venv/bin/python -m pytest tests/test_labeling.py -v`
Expected: ALL PASS (old `label_trend` tests prove the refactor preserved behavior).

- [ ] **Step 5: Commit**

```bash
git add training/labeling.py tests/test_labeling.py
git commit -m "feat(r4): classify_rate extraction + lsq_slope (spec section 2)"
```

### Task 2: dataset retains horizon readings and endpoint rates

**Files:**
- Modify: `training/dataset.py`
- Test: `tests/test_dataset.py` (append)

**Interfaces:**
- Produces: `GlucoseTrendDataset.horizons` (list of np arrays, each shape (horizon,), the readings after the window) and `.rates` (list of float endpoint rates, `(future - last)/horizon_minutes`), aligned with `windows`/`labels`.
- Consumes: Task 1's `classify_rate` indirectly (labels unchanged).

- [ ] **Step 1: Write failing test**

Append to `tests/test_dataset.py` (reuse the file's existing frame-building helper style; it already constructs small DataFrames with id/id_segment/time/gl):

```python
def test_horizons_and_rates_aligned(simple_frame):  # adapt fixture name to the file's existing helper
    ds = GlucoseTrendDataset(simple_frame, classes=FIVE_CLASSES)
    assert len(ds.horizons) == len(ds) == len(ds.rates)
    for k in range(len(ds)):
        h = ds.horizons[k]
        assert h.shape == (ds.horizon,)
        expected_rate = (h[-1] - ds.windows[k][-1]) / ds.horizon_minutes
        assert abs(ds.rates[k] - expected_rate) < 1e-6
        # label still derives from the endpoint rate
        assert ds.labels[k] == ds.class_to_idx[classify_rate(expected_rate)]
```

(At execution: open `tests/test_dataset.py` first, copy its existing frame fixture/helper, keep window count > 0.)

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `AttributeError: ... 'horizons'`.

- [ ] **Step 3: Implement**

In `training/dataset.py.__init__`, add beside the existing per-window lists:

```python
        self.horizons = []   # readings after the window, shape (horizon,)
        self.rates = []      # endpoint rate in mg/dL/min, label_trend's input
```

and inside the window loop, after `future_value = ...`:

```python
                horizon_vals = values[i + input_length:i + span]
                self.horizons.append(horizon_vals)
                self.rates.append(float(future_value - last_value) / horizon_minutes)
```

- [ ] **Step 4: Run full dataset + train suites**

Run: `venv/bin/python -m pytest tests/test_dataset.py tests/test_train.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py tests/test_dataset.py
git commit -m "feat(r4): dataset retains horizon readings + endpoint rates"
```

### Task 3: heads module — sqrt weights, CORN loss/decode, output decoding

**Files:**
- Create: `training/heads.py`
- Test: `tests/test_heads.py`

**Interfaces:**
- Produces:
  - `HEADS = ("ce", "wtemp", "reg", "corn")` and `head_outputs(head) -> int` (5, 5, 1, 4);
  - `sqrt_class_weights(train_ds, num_classes) -> torch.Tensor` (= `class_weights(...).sqrt()`);
  - `corn_loss(logits, targets, weights) -> torch.Tensor` (logits (B,4), targets (B,) long 0–4, weights (5,) per-class sample weights);
  - `decode_outputs(head, outputs: np.ndarray) -> np.ndarray` ((B, width) raw model outputs → (B,) class indices: ce/wtemp argmax; reg thresholds via `classify_rate`; corn counts `> 0` logits);
  - `make_r4_criterion(head, weights)` returning the training criterion for each head (`ce`→weighted CE; `wtemp`→CE with sqrt weights already passed in; `reg`→Huber(δ=1.0) on `outputs.squeeze(-1)` vs float targets; `corn`→`corn_loss` closure).
- Consumes: `training.train.class_weights`, `training.labeling.classify_rate`, `training.labeling.FIVE_CLASSES`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_heads.py`:

```python
import numpy as np
import torch

from training.heads import (HEADS, corn_loss, decode_outputs, head_outputs,
                            make_r4_criterion, sqrt_class_weights)
from training.labeling import FIVE_CLASSES


class _Tiny:
    labels = [0, 2, 2, 2, 4]  # imbalanced: stable 3x, fasts 1x each


def test_head_widths():
    assert [head_outputs(h) for h in HEADS] == [5, 5, 1, 4]


def test_sqrt_weights_are_sqrt_of_balanced():
    from training.train import class_weights
    w = class_weights(_Tiny(), 5)
    assert torch.allclose(sqrt_class_weights(_Tiny(), 5), w.sqrt())


def test_decode_ce_argmax():
    out = np.array([[0.1, 0.2, 3.0, 0.1, 0.1], [5.0, 0, 0, 0, 0]], dtype=np.float32)
    assert decode_outputs("ce", out).tolist() == [2, 0]


def test_decode_reg_thresholds_match_labeler():
    rates = np.array([[-2.0], [-1.0], [-0.5], [0.99], [1.0], [2.0]], dtype=np.float32)
    idx = decode_outputs("reg", rates)
    names = [FIVE_CLASSES[i] for i in idx]
    assert names == ["falling_fast", "falling", "stable", "stable", "rising", "rising_fast"]


def test_decode_corn_counts_positives_even_non_monotone():
    out = np.array([[-1.0, 2.0, -3.0, 4.0],   # 2 positives -> class 2
                    [1.0, 1.0, 1.0, 1.0],      # 4 -> class 4
                    [-1.0, -1.0, -1.0, -1.0]], # 0 -> class 0
                   dtype=np.float32)
    assert decode_outputs("corn", out).tolist() == [2, 4, 0]


def test_corn_loss_hand_case():
    # One sample, y=2: outputs 0 and 1 are conditioned on (targets 1,1),
    # output 2 is conditioned on y>=2 (target 0), output 3 is NOT trained
    # (y >= 3 subset empty for this sample).
    logits = torch.zeros(1, 4)  # sigmoid = 0.5 everywhere -> BCE = ln 2 per term
    w = torch.ones(5)
    loss = corn_loss(logits, torch.tensor([2]), w)
    assert torch.isclose(loss, torch.tensor(float(np.log(2.0))), atol=1e-6)


def test_make_r4_criterion_reg_is_huber():
    crit = make_r4_criterion("reg", torch.ones(5))
    out = torch.tensor([[1.5], [0.0]])
    tgt = torch.tensor([1.5, 2.0])
    # |err| = 0 and 2.0 -> huber(delta=1): 0 and (2 - 0.5) = 1.5; mean = 0.75
    assert torch.isclose(crit(out, tgt), torch.tensor(0.75), atol=1e-6)
```

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_heads.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'training.heads'`.

- [ ] **Step 3: Implement `training/heads.py`**

```python
"""r4 objective heads (spec 2026-10-13 section 3, Stage H): output widths,
training criteria, and the decode back to the 5 trend classes. Decode rules
mirror training.labeling exactly; eval labels never change."""
import numpy as np
import torch
import torch.nn.functional as F

from training.labeling import FIVE_CLASSES, classify_rate

HEADS = ("ce", "wtemp", "reg", "corn")
_WIDTHS = {"ce": 5, "wtemp": 5, "reg": 1, "corn": 4}
HUBER_DELTA = 1.0


def head_outputs(head):
    return _WIDTHS[head]


def sqrt_class_weights(train_ds, num_classes):
    from training.train import class_weights
    return class_weights(train_ds, num_classes).sqrt()


def corn_loss(logits, targets, weights):
    """CORN conditional loss. Output k is trained only on samples with
    y >= k, target 1[y > k]; per-sample class weight w[y]; normalized by
    the summed weights of all trained (sample, output) terms."""
    total, denom = logits.sum() * 0.0, logits.sum() * 0.0
    w = weights.to(logits.device)[targets]
    for k in range(logits.shape[1]):
        mask = targets >= k
        if not bool(mask.any()):
            continue
        t = (targets[mask] > k).float()
        per = F.binary_cross_entropy_with_logits(logits[mask, k], t, reduction="none")
        total = total + (per * w[mask]).sum()
        denom = denom + w[mask].sum()
    return total / denom


def make_r4_criterion(head, weights):
    weights = weights.float()
    if head in ("ce", "wtemp"):  # wtemp differs only in the weights passed in
        return torch.nn.CrossEntropyLoss(weight=weights)
    if head == "reg":
        huber = torch.nn.HuberLoss(delta=HUBER_DELTA)

        def reg(outputs, targets):
            return huber(outputs.squeeze(-1), targets.float())
        return reg
    if head == "corn":
        def corn(outputs, targets):
            return corn_loss(outputs, targets, weights)
        return corn
    raise ValueError(f"unknown head {head!r}")


def decode_outputs(head, outputs):
    """(B, width) raw model outputs -> (B,) FIVE_CLASSES indices."""
    out = np.asarray(outputs)
    if head in ("ce", "wtemp"):
        return out.argmax(axis=1)
    if head == "reg":
        idx = {c: i for i, c in enumerate(FIVE_CLASSES)}
        return np.array([idx[classify_rate(float(r))] for r in out[:, 0]], dtype=np.int64)
    if head == "corn":
        return (out > 0).sum(axis=1).astype(np.int64)
    raise ValueError(f"unknown head {head!r}")
```

- [ ] **Step 4: Run, verify pass**

Run: `venv/bin/python -m pytest tests/test_heads.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add training/heads.py tests/test_heads.py
git commit -m "feat(r4): objective heads - sqrt weights, CORN, regression decode"
```

### Task 4: augmentation module — level noise and mixup

**Files:**
- Create: `training/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Produces:
  - `noisy_delta_windows(levels, sigma, rng) -> np.ndarray` ((N, 12) raw mg/dL levels → (N, 12) delta-anchor encodings of `levels + N(0, sigma)`, float32);
  - `mixup_collate(alpha, seed)` → a torch DataLoader `collate_fn` for batches of `(x: (1,12) float tensor, rate: float)` pairs returning mixed `(x: (B,1,12), rate: (B,))` with one Beta(alpha, alpha) λ per batch, partner = seeded random permutation.
- Consumes: `training.deltas.delta_windows`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_augment.py`:

```python
import numpy as np
import torch

from training.augment import mixup_collate, noisy_delta_windows
from training.deltas import delta_windows


def test_noisy_deltas_sigma_zero_is_plain_encoding():
    levels = np.linspace(100, 155, 12, dtype=np.float32)[None]
    rng = np.random.default_rng(0)
    assert np.allclose(noisy_delta_windows(levels, 0.0, rng),
                       delta_windows(levels, True))


def test_noisy_deltas_deterministic_under_seed():
    levels = np.full((4, 12), 120.0, dtype=np.float32)
    a = noisy_delta_windows(levels, 2.0, np.random.default_rng(7))
    b = noisy_delta_windows(levels, 2.0, np.random.default_rng(7))
    c = noisy_delta_windows(levels, 2.0, np.random.default_rng(8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def _items(n):
    return [(torch.full((1, 12), float(i)), float(i)) for i in range(n)]


def test_mixup_convex_combination():
    fn = mixup_collate(alpha=0.4, seed=3)
    x, r = fn(_items(8))
    assert x.shape == (8, 1, 12) and r.shape == (8,)
    # every mixed rate lies between the min and max source rates
    assert float(r.min()) >= 0.0 and float(r.max()) <= 7.0
    # x and rate are mixed with the SAME lambda: x rows are constant-valued
    assert torch.allclose(x[:, 0, 0], r)


def test_mixup_batch_of_one_is_identity():
    fn = mixup_collate(alpha=0.4, seed=0)
    x, r = fn(_items(1))
    assert x.shape == (1, 1, 12) and torch.isclose(r[0], torch.tensor(0.0))
```

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_augment.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/augment.py`**

```python
"""r4 robustness-arm transforms (spec section 3, Stage R). Train-time only;
eval inputs are never touched."""
import numpy as np
import torch

from training.deltas import delta_windows


def noisy_delta_windows(levels, sigma, rng):
    """Delta-anchor encode levels + N(0, sigma). sigma in mg/dL; fresh draw
    per call (the caller re-encodes each epoch)."""
    levels = np.asarray(levels, dtype=np.float32)
    if sigma > 0:
        levels = levels + rng.normal(0.0, sigma, size=levels.shape).astype(np.float32)
    return delta_windows(levels, True).astype(np.float32)


def mixup_collate(alpha, seed):
    """collate_fn mixing (x, rate) batches: one Beta(alpha, alpha) lambda per
    batch, partner = seeded permutation. Linear in the regression target -
    only valid for the reg head (spec: R_mixup exists only if H_reg won)."""
    rng = np.random.default_rng(seed)

    def collate(items):
        x = torch.stack([it[0] for it in items])
        r = torch.tensor([it[1] for it in items], dtype=torch.float32)
        lam = float(rng.beta(alpha, alpha))
        perm = torch.from_numpy(rng.permutation(len(items)))
        return lam * x + (1 - lam) * x[perm], lam * r + (1 - lam) * r[perm]

    return collate
```

- [ ] **Step 4: Run, verify pass**

Run: `venv/bin/python -m pytest tests/test_augment.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add training/augment.py tests/test_augment.py
git commit -m "feat(r4): level-noise and mixup transforms for Stage R"
```

### Task 5: train_model gains collate_fn and per-epoch checkpoints

**Files:**
- Modify: `training/train.py:74-100` (`train_model`)
- Test: `tests/test_train.py` (append)

**Interfaces:**
- Produces: `train_model(train_ds, val_ds, num_classes, epochs, device, model=None, criterion=None, batch_size=64, collate_fn=None, epoch_dir=None)` — `collate_fn` is passed to the train DataLoader only; `epoch_dir` (a Path), when set, writes `epoch{e:02d}.pt` (CPU state_dict) after each epoch.
- Consumes: nothing new. All existing callers keep working (both new params default off).

- [ ] **Step 1: Write failing test**

Append to `tests/test_train.py` (reuse its existing tiny-dataset fixture style):

```python
def test_train_model_epoch_dir_and_collate(tmp_path):
    import torch

    from training.train import train_model

    xs = torch.randn(8, 1, 12)
    ys = torch.randint(0, 5, (8,))
    ds = list(zip(xs, [int(y) for y in ys]))
    calls = []

    def collate(items):
        calls.append(len(items))
        return (torch.stack([i[0] for i in items]),
                torch.tensor([i[1] for i in items]))

    train_model(ds, ds, 5, 2, torch.device("cpu"), batch_size=4,
                collate_fn=collate, epoch_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.glob("epoch*.pt")) == ["epoch00.pt", "epoch01.pt"]
    assert calls  # collate_fn actually used
    sd = torch.load(tmp_path / "epoch01.pt", map_location="cpu", weights_only=True)
    assert all(v.device.type == "cpu" for v in sd.values())
```

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_train.py -v -k epoch_dir`
Expected: FAIL `TypeError: train_model() got an unexpected keyword argument`.

- [ ] **Step 3: Implement**

In `training/train.py`, change the signature and loader, and add the per-epoch save at the end of the epoch loop:

```python
def train_model(train_ds, val_ds, num_classes, epochs, device, model=None,
                criterion=None, batch_size=64, collate_fn=None, epoch_dir=None):
    ...
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
```

and after the `print(f"epoch ...")` line, still inside the loop:

```python
        if epoch_dir is not None:
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       epoch_dir / f"epoch{epoch:02d}.pt")
```

- [ ] **Step 4: Run the train suite**

Run: `venv/bin/python -m pytest tests/test_train.py -v`
Expected: ALL PASS (existing callers untouched).

- [ ] **Step 5: Commit**

```bash
git add training/train.py tests/test_train.py
git commit -m "feat(r4): train_model collate_fn + per-epoch checkpoint retention"
```

### Task 6: state-dict averaging (soup / SWA)

**Files:**
- Create: `training/averaging.py`
- Test: `tests/test_averaging.py`

**Interfaces:**
- Produces: `average_state_dicts(dicts: list[dict]) -> dict` — uniform mean of float tensors; refuses (ValueError) on empty input or mismatched key sets/shapes.
- Consumes: nothing.

- [ ] **Step 1: Write failing tests**

Create `tests/test_averaging.py`:

```python
import pytest
import torch

from training.averaging import average_state_dicts


def _sd(v):
    return {"w": torch.full((2, 2), float(v)), "b": torch.tensor([float(v)])}


def test_average_is_mean():
    out = average_state_dicts([_sd(0), _sd(2), _sd(4)])
    assert torch.allclose(out["w"], torch.full((2, 2), 2.0))
    assert torch.allclose(out["b"], torch.tensor([2.0]))


def test_average_of_identical_is_identity():
    out = average_state_dicts([_sd(3), _sd(3)])
    assert torch.allclose(out["w"], _sd(3)["w"])


def test_mismatched_keys_refused():
    bad = {"w": torch.zeros(2, 2)}  # missing "b"
    with pytest.raises(ValueError):
        average_state_dicts([_sd(1), bad])
    with pytest.raises(ValueError):
        average_state_dicts([])
```

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_averaging.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/averaging.py`**

```python
"""Uniform state-dict averaging for Stage W (spec section 3): model soup
across seeds, SWA across epoch checkpoints. TrendCNN has no BatchNorm, so a
plain parameter mean is exact - no statistics recompute."""
import torch


def average_state_dicts(dicts):
    if not dicts:
        raise ValueError("average_state_dicts: empty input")
    keys = set(dicts[0])
    for d in dicts[1:]:
        if set(d) != keys or any(d[k].shape != dicts[0][k].shape for k in keys):
            raise ValueError("average_state_dicts: mismatched state dicts")
    return {k: torch.stack([d[k].float() for d in dicts]).mean(dim=0) for k in keys}
```

- [ ] **Step 4: Run, verify pass**

Run: `venv/bin/python -m pytest tests/test_averaging.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add training/averaging.py tests/test_averaging.py
git commit -m "feat(r4): uniform state-dict averaging for soup/SWA"
```

### Task 7: r4 runner — constants, arms, cohort cache, tensors, views

**Files:**
- Create: `experiments/accuracy_sweep_r4.py`
- Test: `tests/test_accuracy_sweep_r4.py`

**Interfaces:**
- Produces (consumed by Tasks 8–12; all in `experiments/accuracy_sweep_r4.py`):
  - constants `ARM_SEEDS_R4 = (0, 1, 2, 3, 4)`, `BOOT_SEED_R4 = 20261014`, `HEADROOM_MIN = 0.02`, `MIXTURE_R4 = "D5"`, `EXP0_PATH`, `PROTOCOL_R4` (both under `paper/results/calibration-coverage/`), `NOISE_SIGMA = 2.0`, `MIXUP_ALPHA = 0.4`, `SWA_LAST_K = 5`;
  - `arm_r4(name, head, labels="endpoint", aug="none") -> dict` with keys `name, head, labels, aug, mixture, channels=[96,144], length=12, in_channels=1, arch="cnn"`;
  - `stage_h_arms()` → `[H_ce, H_wtemp, H_reg, H_corn]`; `stage_r_arms(winner) -> [R_base, R_labels, R_noise(, R_mixup)]` (R_mixup present iff `winner["head"] == "reg"`);
  - `_cohort_cache_r4(cohort) -> (levels, horizons, y, pids)` (npz cache `results/diadata_cache/r4/{cohort}.npz`);
  - `build_arm_tensors_r4(run, a, sample_seed) -> Path` — npz with `x` (N,1,12 encoded), `levels` (N,12), `y` (N,), `r` (N, endpoint rates), `lsq` (N, LSQ rates over window[-1]+horizons); asserts all rates finite; path `tensors/{mixture}_{labels}_{aug}_s{seed}.npz` (head is NOT in the key — all heads share tensors);
  - views: `R4View(x, y)` (int targets), `RateView(x, r)` (float targets), `NoisyLevelView(levels, y_or_r, sigma, seed, float_targets)` (re-encodes with fresh noise per `__getitem__`);
  - `_train_targets(a, z) -> np.ndarray` — labels arm + head → int labels (endpoint or LSQ-classified) or float rates (endpoint or LSQ) for reg.
- Consumes: Tasks 1–4; r2/r3 helpers (`TensorView`, `SAMPLE_SEED`, `sample_subjects`, `CAP`, `TRAIN_COHORTS`, `_run_dir`, `_write_json`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_accuracy_sweep_r4.py`:

```python
import numpy as np
import pytest

from experiments.accuracy_sweep_r4 import (ARM_SEEDS_R4, BOOT_SEED_R4,
                                           HEADROOM_MIN, NoisyLevelView,
                                           RateView, _tensor_path_r4,
                                           _train_targets, arm_r4,
                                           stage_h_arms, stage_r_arms)


def test_constants():
    assert ARM_SEEDS_R4 == (0, 1, 2, 3, 4)
    assert BOOT_SEED_R4 == 20261014
    assert HEADROOM_MIN == 0.02


def test_stage_h_arms():
    arms = stage_h_arms()
    assert [a["name"] for a in arms] == ["H_ce", "H_wtemp", "H_reg", "H_corn"]
    assert [a["head"] for a in arms] == ["ce", "wtemp", "reg", "corn"]
    assert all(a["mixture"] == "D5" and a["in_channels"] == 1 for a in arms)


def test_stage_r_arms_mixup_gated_on_reg():
    reg_winner = arm_r4("H_reg", "reg")
    ce_winner = arm_r4("H_ce", "ce")
    assert [a["name"] for a in stage_r_arms(reg_winner)] == [
        "R_base", "R_labels", "R_noise", "R_mixup"]
    assert [a["name"] for a in stage_r_arms(ce_winner)] == [
        "R_base", "R_labels", "R_noise"]
    # R arms inherit the winning head
    assert all(a["head"] == "ce" for a in stage_r_arms(ce_winner))


def test_tensor_key_ignores_head_but_not_labels_or_aug(tmp_path):
    a1 = arm_r4("H_ce", "ce")
    a2 = arm_r4("H_corn", "corn")
    a3 = arm_r4("R_labels", "ce", labels="lsq")
    a4 = arm_r4("R_noise", "ce", aug="noise")
    p = {k: _tensor_path_r4(tmp_path, a, 1) for k, a in
         {"a1": a1, "a2": a2, "a3": a3, "a4": a4}.items()}
    assert p["a1"] == p["a2"]                 # heads share tensors
    assert len({p["a1"], p["a3"], p["a4"]}) == 3  # labels/aug do not


def test_train_targets_routing():
    z = {"y": np.array([2, 3]), "r": np.array([0.5, 1.2]),
         "lsq": np.array([0.4, 1.1]),
         "levels": np.zeros((2, 12), dtype=np.float32)}
    assert _train_targets(arm_r4("H_ce", "ce"), z).tolist() == [2, 3]
    assert _train_targets(arm_r4("H_reg", "reg"), z).tolist() == [0.5, 1.2]
    lsq_reg = _train_targets(arm_r4("R_labels", "reg", labels="lsq"), z)
    assert lsq_reg.tolist() == [0.4, 1.1]
    lsq_cls = _train_targets(arm_r4("R_labels", "ce", labels="lsq"), z)
    assert lsq_cls.tolist() == [2, 3]  # 0.4 -> stable(2), 1.1 -> rising(3)


def test_rate_view_yields_float_targets():
    v = RateView(np.zeros((3, 1, 12), dtype=np.float32),
                 np.array([0.5, -1.5, 2.5], dtype=np.float32))
    x, r = v[1]
    assert x.shape == (1, 12) and isinstance(r, float) and r == -1.5


def test_noisy_view_reencodes_per_access():
    levels = np.full((2, 12), 120.0, dtype=np.float32)
    v = NoisyLevelView(levels, np.array([2, 2]), sigma=2.0, seed=0,
                       float_targets=False)
    x1, _ = v[0]
    x2, _ = v[0]
    assert x1.shape == (1, 12) and not np.array_equal(x1.numpy(), x2.numpy())


def test_build_tensors_refuses_non_finite_rates(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r4 as r4

    def fake_cohort(cohort):
        levels = np.full((3, 12), 100.0, dtype=np.float32)
        horizons = np.full((3, 3), np.nan, dtype=np.float32)  # corrupted
        y = np.array([2, 2, 2], dtype=np.int64)
        pids = np.array(["p1", "p2", "p3"])
        return levels, horizons, y, pids

    monkeypatch.setattr(r4, "_cohort_cache_r4", fake_cohort)
    with pytest.raises(SystemExit, match="non-finite"):
        r4.build_arm_tensors_r4(tmp_path, r4.arm_r4("H_ce", "ce"), sample_seed=1)
```

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement the runner skeleton**

Create `experiments/accuracy_sweep_r4.py` (first slice — later tasks extend this file):

```python
"""Accuracy sweep r4: noise-ceiling gate + objective heads + robustness +
weight averaging (spec 2026-10-13-accuracy-sweep-r4-design.md).

Selection stays on weinstock val; Experiment 0 gates training; champion is
the bundled (shipped r2) pair read live from android assets."""
import json
from pathlib import Path

import numpy as np

from experiments.accuracy_sweep_r2 import (SAMPLE_SEED, TensorView,
                                           directional_recall, stable_recall)
from experiments.calibration_coverage import GLUCOEDGE_ROOT, _run_dir, _write_json

ARM_SEEDS_R4 = (0, 1, 2, 3, 4)
BOOT_SEED_R4 = 20261014  # pre-registered (protocol_sweep_r4.json); single source
HEADROOM_MIN = 0.02      # Experiment 0 abort: headroom < 2pp (strict)
MIXTURE_R4 = "D5"        # shipped r2 champion mixture; RT-CGM not used (r3 null)
NOISE_SIGMA = 2.0        # mg/dL, R_noise arm
MIXUP_ALPHA = 0.4        # R_mixup arm (reg head only)
SWA_LAST_K = 5           # epochs averaged per seed in W_swa
EXP0_PATH = (GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage"
             / "exp0_sweep_r4.json")
PROTOCOL_R4 = (GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage"
               / "protocol_sweep_r4.json")


def arm_r4(name, head, labels="endpoint", aug="none"):
    return {"name": name, "head": head, "labels": labels, "aug": aug,
            "mixture": MIXTURE_R4, "arch": "cnn", "in_channels": 1,
            "channels": [96, 144], "length": 12}


def stage_h_arms():
    return [arm_r4("H_ce", "ce"), arm_r4("H_wtemp", "wtemp"),
            arm_r4("H_reg", "reg"), arm_r4("H_corn", "corn")]


def stage_r_arms(winner):
    h = winner["head"]
    arms = [arm_r4("R_base", h),
            arm_r4("R_labels", h, labels="lsq"),
            arm_r4("R_noise", h, aug="noise")]
    if h == "reg":
        arms.append(arm_r4("R_mixup", h, aug="mixup"))
    return arms


def _tensor_path_r4(run, a, sample_seed):
    return (Path(run) / "tensors"
            / f"{a['mixture']}_{a['labels']}_{a['aug']}_s{sample_seed}.npz")


def _cohort_cache_r4(cohort):
    """Raw level windows + horizon readings per DiaData training cohort;
    own npz cache (r3's hour cache lacks horizons)."""
    safe = cohort.replace("-", "_")
    cache = GLUCOEDGE_ROOT / "results" / "diadata_cache" / "r4" / f"{safe}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["levels"], z["horizons"], z["y"], z["pids"]
    from training.dataset import GlucoseTrendDataset
    from training.diadata import load_diadata
    from training.labeling import FIVE_CLASSES
    frame = load_diadata(GLUCOEDGE_ROOT.parent / "DiaData", [cohort],
                         cache_dir=GLUCOEDGE_ROOT / "results" / "diadata_cache")
    ds = GlucoseTrendDataset(frame, classes=FIVE_CLASSES, input_length=12)
    out = (np.stack(ds.windows).astype(np.float32),
           np.stack(ds.horizons).astype(np.float32),
           np.asarray(ds.labels, dtype=np.int64),
           np.asarray([str(p) for p in ds.participants]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, levels=out[0], horizons=out[1], y=out[2], pids=out[3])
    return out


def build_arm_tensors_r4(run, a, sample_seed=SAMPLE_SEED):
    from training.deltas import delta_windows
    from training.labeling import lsq_slope
    from training.mixtures import CAP, TRAIN_COHORTS, sample_subjects
    path = _tensor_path_r4(run, a, sample_seed)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {c: _cohort_cache_r4(c) for c in TRAIN_COHORTS}
    sizes = {c: len(d[2]) for c, d in data.items()}
    total = sum(sizes.values())
    lv, hz, ys = [], [], []
    for c in TRAIN_COHORTS:
        levels, horizons, y, pids = data[c]
        counts = {}
        for p in pids:
            counts[p] = counts.get(p, 0) + 1
        chosen, _ = sample_subjects(counts, int(CAP * sizes[c] / total), sample_seed)
        mask = np.isin(pids, np.asarray(chosen))
        lv.append(levels[mask]); hz.append(horizons[mask]); ys.append(y[mask])
    levels = np.concatenate(lv); horizons = np.concatenate(hz)
    y = np.concatenate(ys)
    r = (horizons[:, -1] - levels[:, -1]) / 15.0
    lsq = lsq_slope(np.concatenate([levels[:, -1:], horizons], axis=1)).astype(np.float32)
    if not (np.isfinite(r).all() and np.isfinite(lsq).all()):
        raise SystemExit("REFUSED: non-finite rate targets in arm tensors")
    x = delta_windows(levels, True).astype(np.float32)[:, None, :]
    np.savez_compressed(path, x=x, levels=levels, y=y,
                        r=r.astype(np.float32), lsq=lsq)
    return path


def _train_targets(a, z):
    """Arm's labels/head -> training target array from a tensor npz dict."""
    rates = z["lsq"] if a["labels"] == "lsq" else z["r"]
    if a["head"] == "reg":
        return np.asarray(rates, dtype=np.float32)
    if a["labels"] == "lsq":
        from training.labeling import FIVE_CLASSES, classify_rate
        idx = {c: i for i, c in enumerate(FIVE_CLASSES)}
        return np.array([idx[classify_rate(float(v))] for v in rates], dtype=np.int64)
    return np.asarray(z["y"], dtype=np.int64)


class R4View(TensorView):
    """Rows already (1, 12); int targets."""

    def __getitem__(self, k):
        import torch
        return torch.from_numpy(self.x[k]), int(self.y[k])


class RateView:
    def __init__(self, x, r):
        self.x, self.y = x, np.asarray(r, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, k):
        import torch
        return torch.from_numpy(self.x[k]), float(self.y[k])


class NoisyLevelView:
    """Re-encodes levels with fresh Gaussian noise on every access (R_noise)."""

    def __init__(self, levels, targets, sigma, seed, float_targets):
        self.levels = levels
        self.y = targets
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.float_targets = float_targets

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, k):
        import torch
        from training.augment import noisy_delta_windows
        x = noisy_delta_windows(self.levels[k:k + 1], self.sigma, self.rng)[0][None]
        t = float(self.y[k]) if self.float_targets else int(self.y[k])
        return torch.from_numpy(x), t
```

Note: `R4View.__init__` is inherited from r2's `TensorView(x, y)`; at execution confirm its constructor stores `self.x`/`self.y` (it does — `TensorView3` in r3 relies on the same) and that `class_weights`' `_dataset_labels` path is never called with float-target views (reg arms pass weights built from `z["y"]` — see Task 9).

- [ ] **Step 4: Run, verify pass**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r4.py tests/test_accuracy_sweep_r4.py
git commit -m "feat(r4): runner skeleton - arms, cohort cache with horizons, tensors, views"
```

### Task 8: Experiment 0 — ceiling measurement and binding abort guard

**Files:**
- Modify: `experiments/accuracy_sweep_r4.py` (append)
- Test: `tests/test_accuracy_sweep_r4.py` (append)

**Interfaces:**
- Produces:
  - `exp0_agreement(endpoint_idx, smoothed_idx) -> dict` — pure: `{"ceiling_directional", "ceiling_stable", "per_class_recall"}` treating smoothed as predictor of endpoint on 5-class indices;
  - `exp0_verdict(ceiling_directional, champion_val_directional) -> dict` — `{"headroom", "abort"}`, abort iff `headroom < HEADROOM_MIN` (strict);
  - `stage_exp0(run_id)` — computes labels on eligible weinstock val (endpoint + LSQ + σ∈{1,2} perturbed companions), champion val directional via `champion_val_stats`, writes run-dir copy AND `EXP0_PATH` (aggregates only: counts, recalls, verdict, estimator description verbatim);
  - `_exp0_gate()` — raises `SystemExit("REFUSED: ...")` unless `EXP0_PATH` exists, is git-committed clean (`protocol_is_committed(EXP0_PATH)`), and has `"abort": false`. Called at the top of `stage_train`.
- Consumes: Task 1 `lsq_slope`/`classify_rate`; r2 `champion_val_stats`; r1 `protocol_is_committed`, `_data`, `_eligible`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_accuracy_sweep_r4.py`:

```python
def test_exp0_agreement_hand_case():
    from experiments.accuracy_sweep_r4 import exp0_agreement
    endpoint = np.array([0, 1, 2, 3, 4, 2, 2, 1])
    smoothed = np.array([0, 2, 2, 3, 4, 2, 3, 1])  # misses one falling, one stable
    out = exp0_agreement(endpoint, smoothed)
    # directional = mean recall over classes 0,1,3,4 = (1 + 0.5 + 1 + 1)/4
    assert abs(out["ceiling_directional"] - 0.875) < 1e-9
    assert abs(out["ceiling_stable"] - 2 / 3) < 1e-9


def test_exp0_verdict_boundary_exact_2pp_proceeds():
    from experiments.accuracy_sweep_r4 import exp0_verdict
    assert exp0_verdict(0.5530, 0.5330)["abort"] is False      # exactly 2.0pp
    assert exp0_verdict(0.55299, 0.5330)["abort"] is True      # just under
    assert abs(exp0_verdict(0.60, 0.5330)["headroom"] - 0.067) < 1e-9


def test_exp0_gate_refuses_without_file(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r4 as r4
    monkeypatch.setattr(r4, "EXP0_PATH", tmp_path / "exp0_sweep_r4.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        r4._exp0_gate()


def test_exp0_gate_refuses_on_abort(tmp_path, monkeypatch):
    import json as _json

    import experiments.accuracy_sweep_r4 as r4
    p = tmp_path / "exp0_sweep_r4.json"
    p.write_text(_json.dumps({"abort": True}))
    monkeypatch.setattr(r4, "EXP0_PATH", p)
    monkeypatch.setattr(r4, "protocol_is_committed", lambda path: True)
    with pytest.raises(SystemExit, match="REFUSED"):
        r4._exp0_gate()
```

(Import `protocol_is_committed` into the r4 module namespace so the monkeypatch works: `from experiments.accuracy_sweep import protocol_is_committed`.)

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v -k exp0`
Expected: FAIL (names missing).

- [ ] **Step 3: Implement (append to the runner)**

```python
from experiments.accuracy_sweep import protocol_is_committed  # top of file

EXP0_ESTIMATOR = ("smoothed label = least-squares slope over the four horizon "
                  "readings g[t..t+3] at 0/5/10/15 min, classified with the "
                  "unchanged clinical thresholds; endpoint label = "
                  "(g[t+3]-g[t])/15 classified identically")


def exp0_agreement(endpoint_idx, smoothed_idx):
    from experiments.analyze_calibration_coverage import confusion_counts
    from training.labeling import FIVE_CLASSES
    cm = confusion_counts(np.asarray(endpoint_idx), np.asarray(smoothed_idx))
    return {"ceiling_directional": directional_recall(cm),
            "ceiling_stable": stable_recall(cm),
            "per_class_recall": {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
                                 for i, c in enumerate(FIVE_CLASSES)}}


def exp0_verdict(ceiling_directional, champion_val_directional):
    # round to 1e-6 (0.0001pp) so the binary-float residue of a subtraction
    # like 0.553 - 0.533 cannot flip the pre-registered strict < at exactly 2pp
    headroom = round(float(ceiling_directional - champion_val_directional), 6)
    return {"headroom": headroom, "abort": bool(headroom < HEADROOM_MIN)}


def _val_label_views():
    """Eligible weinstock val: (levels (N,12), horizons (N,3), endpoint idx)."""
    from experiments.accuracy_sweep import _data
    from experiments.calibration_coverage import _eligible
    ds, owner, ngf = _data("weinstock", 12)
    idx = _eligible(ds["val"], owner, ngf, "val")
    t = ds["val"]
    levels = np.stack([t.windows[i] for i in idx]).astype(np.float32)
    horizons = np.stack([t.horizons[i] for i in idx]).astype(np.float32)
    y = np.asarray([t.labels[i] for i in idx], dtype=np.int64)
    return levels, horizons, y


def _classify_rates(rates):
    from training.labeling import FIVE_CLASSES, classify_rate
    idx = {c: i for i, c in enumerate(FIVE_CLASSES)}
    return np.array([idx[classify_rate(float(v))] for v in rates], dtype=np.int64)


def stage_exp0(run_id):
    from experiments.accuracy_sweep_r2 import champion_val_stats
    from training.labeling import lsq_slope
    run = _run_dir(run_id)
    levels, horizons, y = _val_label_views()
    pts = np.concatenate([levels[:, -1:], horizons], axis=1)
    smoothed = _classify_rates(lsq_slope(pts))
    agree = exp0_agreement(y, smoothed)
    champ = champion_val_stats(run)
    verdict = exp0_verdict(agree["ceiling_directional"], champ["directional"])
    noise = {}
    for sigma in (1.0, 2.0):
        rng = np.random.default_rng(20261013)
        noisy = pts + rng.normal(0.0, sigma, size=pts.shape)
        rate = (noisy[:, -1] - noisy[:, 0]) / 15.0
        cm_in = exp0_agreement(y, _classify_rates(rate))
        noise[f"sigma_{sigma:g}"] = cm_in["ceiling_directional"]
    out = {"run_id": run_id, "estimator": EXP0_ESTIMATOR,
           "val_windows": int(len(y)),
           **agree,
           "champion_val_directional": champ["directional"],
           **verdict,
           "noise_companions_directional": noise,
           "note": "descriptive label-noise measurement; abort rule "
                   f"pre-registered: headroom < {HEADROOM_MIN} aborts r4"}
    _write_json(Path(run) / "exp0.json", out)
    _write_json(EXP0_PATH, out)
    print(json.dumps(verdict, indent=1))
    return out


def _exp0_gate():
    if not EXP0_PATH.exists() or not protocol_is_committed(EXP0_PATH):
        raise SystemExit("REFUSED: commit paper/results/calibration-coverage/"
                         "exp0_sweep_r4.json (clean) before training")
    if json.loads(EXP0_PATH.read_text()).get("abort", True):
        raise SystemExit("REFUSED: Experiment 0 aborted r4 - headroom below "
                         "the pre-registered 2pp; no arm trains")
```

- [ ] **Step 4: Run, verify pass**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r4.py tests/test_accuracy_sweep_r4.py
git commit -m "feat(r4): Experiment 0 ceiling measurement + binding abort gate"
```

### Task 9: stages H and R — head-aware training and selection with the tie rule

**Files:**
- Modify: `experiments/accuracy_sweep_r4.py` (append)
- Test: `tests/test_accuracy_sweep_r4.py` (append)

**Interfaces:**
- Produces:
  - `_train_view(run, a, z)` → the right view for an arm (R4View / RateView / NoisyLevelView) + optional `collate_fn` (mixup);
  - `_train_one_r4(run, a, seed, z, batch)` — builds model `TrendCNN(num_classes=head_outputs(a["head"]), channels=(96,144))`, criterion via `make_r4_criterion` (weights from `z["y"]` — ALWAYS the hard endpoint labels, sqrt'd for `wtemp`), trains with `epoch_dir=run/checkpoints/epochs/{arm}_seed{seed}/`, saves checkpoint dict `{"state_dict", "arch": "cnn", "in_channels": 1, "channels": [96,144], "head": a["head"]}` + provenance sidecar;
  - `_val_stats_r4(run, ckpt_path, head)` → `(directional, stable)` on eligible val via `decode_outputs`;
  - `stage_train(run_id, stage)` for `H`/`R` (calls `_exp0_gate()` first; skip-if-exists per checkpoint); `stage_select(run_id, stage)` for `H`/`R` writing `selection/{stage}.json` with the r3 table shape;
  - `pick_winner_r4(arms, means)` and `best_seed_r4(per_seed, floor)` — 0.1pp tie band, `cost_key` then list order (seeds: lowest index in band);
  - `_ckpt_r4(run, a, seed) -> Path` (`checkpoints/{name}_seed{seed}.pt`).
- Consumes: Tasks 3, 4, 5, 7, 8; r2 `_frozen_batch`, `champion_val_stats`; r1 `pick_winner` (reused inside `pick_winner_r4` where applicable).

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_best_seed_tie_rule_lowest_in_band():
    from experiments.accuracy_sweep_r4 import best_seed_r4
    per_seed = {"0": (0.5300, 0.55), "1": (0.5308, 0.55), "2": (0.5309, 0.55),
                "3": (0.5200, 0.55), "4": (0.5100, 0.40)}
    # floor 0.50: seed 4 disqualified; top = 0.5309, band includes 1 and 2 and 0
    # (0.5300 is 0.09pp below top -> inside 0.1pp band) -> lowest index: 0
    assert best_seed_r4(per_seed, floor=0.50) == 0


def test_best_seed_all_below_floor_refused():
    from experiments.accuracy_sweep_r4 import best_seed_r4
    with pytest.raises(SystemExit, match="stable floor"):
        best_seed_r4({"0": (0.53, 0.30)}, floor=0.50)


def test_train_view_routing(tmp_path):
    from experiments.accuracy_sweep_r4 import _train_view, arm_r4
    z = {"x": np.zeros((4, 1, 12), dtype=np.float32),
         "levels": np.full((4, 12), 100.0, dtype=np.float32),
         "y": np.array([2, 2, 3, 1]), "r": np.array([0.1, 0.2, 1.1, -1.2]),
         "lsq": np.array([0.1, 0.2, 1.1, -1.2])}
    v, collate = _train_view(tmp_path, arm_r4("H_ce", "ce"), z)
    assert type(v).__name__ == "R4View" and collate is None
    v, collate = _train_view(tmp_path, arm_r4("H_reg", "reg"), z)
    assert type(v).__name__ == "RateView" and collate is None
    v, collate = _train_view(tmp_path, arm_r4("R_noise", "corn", aug="noise"), z)
    assert type(v).__name__ == "NoisyLevelView"
    v, collate = _train_view(tmp_path, arm_r4("R_mixup", "reg", aug="mixup"), z)
    assert type(v).__name__ == "RateView" and collate is not None


def test_val_stats_decode_plumbing(tmp_path, monkeypatch):
    # a fake 1-output checkpoint decoding through thresholds end to end
    import torch

    import experiments.accuracy_sweep_r4 as r4
    from training.model import TrendCNN

    m = TrendCNN(num_classes=1, channels=(4, 6))
    torch.save({"state_dict": m.state_dict(), "arch": "cnn", "in_channels": 1,
                "channels": [4, 6], "head": "reg"}, tmp_path / "w.pt")
    fake_val = r4.R4View(np.zeros((6, 1, 12), dtype=np.float32),
                         np.array([2, 2, 2, 2, 2, 2]))
    monkeypatch.setattr(r4, "_val_view_r4", lambda: fake_val)
    d, s = r4._val_stats_r4(tmp_path, tmp_path / "w.pt", "reg")
    assert 0.0 <= d <= 1.0 and 0.0 <= s <= 1.0
```

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v -k "tie or view or val_stats"`
Expected: FAIL (names missing).

- [ ] **Step 3: Implement (append)**

```python
def _ckpt_r4(run, a, seed):
    return Path(run) / "checkpoints" / f"{a['name']}_seed{seed}.pt"


def _val_view_r4():
    levels, horizons, y = _val_label_views()
    from training.deltas import delta_windows
    x = delta_windows(levels, True).astype(np.float32)[:, None, :]
    return R4View(x, y)


_VAL_R4 = []


def _val_r4():
    if not _VAL_R4:
        _VAL_R4.append(_val_view_r4())
    return _VAL_R4[0]


def _train_view(run, a, z):
    """Arm -> (train view, collate_fn or None). Weights/criteria elsewhere."""
    targets = _train_targets(a, z)
    float_t = a["head"] == "reg"
    if a["aug"] == "noise":
        return NoisyLevelView(z["levels"], targets, NOISE_SIGMA,
                              seed=SAMPLE_SEED, float_targets=float_t), None
    if a["aug"] == "mixup":
        from training.augment import mixup_collate
        return RateView(z["x"], targets), mixup_collate(MIXUP_ALPHA, SAMPLE_SEED)
    if float_t:
        return RateView(z["x"], targets), None
    return R4View(z["x"], targets), None


def _train_one_r4(run, a, seed, z, batch):
    import time as _t

    import torch
    from training.heads import head_outputs, make_r4_criterion, sqrt_class_weights
    from training.model import TrendCNN
    from training.train import class_weights, resolve_device, train_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device("mps")

    class _Y:  # weights always come from the hard endpoint labels
        labels = z["y"]
    weights = (sqrt_class_weights(_Y(), 5) if a["head"] == "wtemp"
               else class_weights(_Y(), 5))
    model = TrendCNN(num_classes=head_outputs(a["head"]),
                     channels=tuple(a["channels"]))
    criterion = make_r4_criterion(a["head"], weights.to(device))
    view, collate = _train_view(run, a, z)
    epoch_dir = Path(run) / "checkpoints" / "epochs" / f"{a['name']}_seed{seed}"
    t0 = _t.time()
    model = train_model(view, _val_r4(), head_outputs(a["head"]), 20, device,
                        model=model, criterion=criterion, batch_size=batch,
                        collate_fn=collate, epoch_dir=epoch_dir)
    ckpt = _ckpt_r4(run, a, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "arch": "cnn", "in_channels": 1, "channels": a["channels"],
                "head": a["head"]}, ckpt)
    ckpt.with_suffix(".json").write_text(json.dumps({
        "arm": a["name"], "head": a["head"], "labels": a["labels"],
        "aug": a["aug"], "seed": seed, "device": device.type, "batch": batch,
        "train_windows": len(view), "epochs": 20,
        "duration_s": round(_t.time() - t0, 1)}, indent=1))
    return ckpt


def _val_stats_r4(run, ckpt_path, head):
    import torch
    from experiments.analyze_calibration_coverage import confusion_counts
    from training.heads import decode_outputs, head_outputs
    from conversion.common import load_checkpoint
    val = _val_r4()
    model = load_checkpoint(ckpt_path, head_outputs(head))
    outs = []
    with torch.no_grad():
        for i in range(0, len(val), 8192):
            outs.append(model(torch.from_numpy(val.x[i:i + 8192])).numpy())
    preds = decode_outputs(head, np.concatenate(outs))
    cm = confusion_counts(val.y, preds)
    return directional_recall(cm), stable_recall(cm)


def _read_sel(run, stage):
    return json.loads((Path(run) / "selection" / f"{stage}.json").read_text())


def _arms_for(run, stage):
    if stage == "H":
        return stage_h_arms()
    if stage == "R":
        return stage_r_arms(_read_sel(run, "H")["winner"])
    raise SystemExit(f"unknown stage {stage}")


def stage_train(run_id, stage):
    import time as _t
    from experiments.accuracy_sweep_r2 import _frozen_batch
    _exp0_gate()
    run = _run_dir(run_id)
    batch = _frozen_batch(run)
    for a in _arms_for(run, stage):
        z = None
        for seed in ARM_SEEDS_R4:
            ckpt = _ckpt_r4(run, a, seed)
            if ckpt.exists():
                print(f"skip {ckpt.name}: exists")
                continue
            if a["name"] in ("H_ce", "R_base"):
                raise SystemExit(f"REFUSED: {a['name']} is a reuse cell - copy "
                                 "its checkpoints (see spec section 3), do not train")
            if z is None:
                z = dict(np.load(build_arm_tensors_r4(run, a)))
            t0 = _t.time()
            _train_one_r4(run, a, seed, z, batch)
            print(f"{a['name']} seed {seed} done in {_t.time()-t0:.0f}s")


COST_ORDER = ("ce", "wtemp", "reg", "corn")  # cheaper decode first (tie rule)


def pick_winner_r4(arms, means):
    best = max(means.values())
    band = [a for a in arms if a["name"] in means
            and best - means[a["name"]] <= 0.001]
    band.sort(key=lambda a: (COST_ORDER.index(a["head"]),
                             [x["name"] for x in arms].index(a["name"])))
    return band[0]


def best_seed_r4(per_seed, floor):
    ok = {s: v for s, v in per_seed.items() if v[1] >= floor}
    if not ok:
        raise SystemExit("REFUSED: every seed failed the stable floor")
    best = max(v[0] for v in ok.values())
    band = sorted(int(s) for s, v in ok.items() if best - v[0] <= 0.001)
    return band[0]


def _select(run, stage, arms):
    from experiments.accuracy_sweep_r2 import champion_val_stats
    champ = champion_val_stats(run)
    floor = champ["stable"] - 0.05
    table, means = {}, {}
    for a in arms:
        per_seed = {str(s): _val_stats_r4(run, _ckpt_r4(run, a, s), a["head"])
                    for s in ARM_SEEDS_R4}
        dir_mean = float(np.mean([v[0] for v in per_seed.values()]))
        stab_mean = float(np.mean([v[1] for v in per_seed.values()]))
        dq = stab_mean < floor
        table[a["name"]] = {
            "per_seed": {s: {"directional": v[0], "stable": v[1]}
                         for s, v in per_seed.items()},
            "mean_directional": dir_mean, "mean_stable": stab_mean,
            "disqualified": bool(dq)}
        if not dq:
            means[a["name"]] = dir_mean
    if not means:
        raise SystemExit("REFUSED: every arm failed the stable-recall "
                         "disqualifier; no winner can be selected")
    winner = pick_winner_r4(arms, means)
    out = {"stage": stage, "champion_val": champ, "stable_floor": floor,
           "table": table, "winner": winner}
    sel = Path(run) / "selection"
    sel.mkdir(parents=True, exist_ok=True)
    _write_json(sel / f"{stage}.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
    return out


def stage_select(run_id, stage):
    run = _run_dir(run_id)
    if stage in ("H", "R"):
        return _select(run, stage, _arms_for(run, stage))
    raise SystemExit(f"unknown stage {stage}")
```

Reuse-cell note (spec section 3): `H_ce` copies r3 `D_base` (= r2 M3) seed checkpoints into `checkpoints/H_ce_seed{s}.pt` with the `"head": "ce"` key added, after a bit-identical smoke reproduction, exactly as r3 did; `R_base` copies the H winner's five checkpoints under the `R_base` name. The runner refuses to *train* those cells; the copy is a documented manual step in the run log, recorded in the deviations block if provenance sidecars differ.

- [ ] **Step 4: Run, verify pass**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r4.py tests/test_accuracy_sweep_r4.py
git commit -m "feat(r4): stages H/R - head-aware training, selection, tie rule"
```

### Task 10: Stage W — soup, SWA, best-seed cells

**Files:**
- Modify: `experiments/accuracy_sweep_r4.py` (append)
- Test: `tests/test_accuracy_sweep_r4.py` (append)

**Interfaces:**
- Produces: `stage_select(run_id, "W")` → builds the three cells from the R winner:
  - `W_best`: `best_seed_r4` over the R winner's per-seed val stats;
  - `W_soup`: `average_state_dicts` over the 5 seed checkpoints → saved as `checkpoints/W_soup.pt` (same meta dict, `"seed": "soup"` in the sidecar);
  - `W_swa`: per seed, average `epoch15..epoch19` state dicts from `checkpoints/epochs/{arm}_seed{s}/` → `checkpoints/W_swa_seed{s}.pt`, then best seed among them; **skipped with `{"skipped": "no epoch history"}`** when the winner is a reuse lineage without epoch dirs;
  - writes `selection/W.json`: per-cell val `(directional, stable)`, the winning cell name, `winner` (arm dict + `"cell"` + `"best_seed"` or `"soup"`), stable floor applied per cell.
- Consumes: Tasks 6, 9.

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_stage_w_cells(tmp_path, monkeypatch):
    import torch

    import experiments.accuracy_sweep_r4 as r4
    from training.model import TrendCNN

    run = tmp_path
    a = r4.arm_r4("R_noise", "ce", aug="noise")
    (run / "selection").mkdir()
    (run / "selection" / "R.json").write_text(json.dumps(
        {"winner": a, "champion_val": {"directional": 0.5, "stable": 0.5}}))
    for s in r4.ARM_SEEDS_R4:
        m = TrendCNN(num_classes=5, channels=(4, 6))
        torch.save({"state_dict": m.state_dict(), "arch": "cnn",
                    "in_channels": 1, "channels": [4, 6], "head": "ce"},
                   r4._ckpt_r4(run, a, s).parent.mkdir(parents=True, exist_ok=True)
                   or r4._ckpt_r4(run, a, s))
        ed = run / "checkpoints" / "epochs" / f"R_noise_seed{s}"
        ed.mkdir(parents=True)
        for e in range(20):
            # distinct constant per epoch so the SWA slice is verifiable
            torch.save({k: torch.full_like(v, float(e))
                        for k, v in m.state_dict().items()},
                       ed / f"epoch{e:02d}.pt")
    fake_val = r4.R4View(np.zeros((4, 1, 12), dtype=np.float32),
                         np.array([2, 2, 2, 2]))
    monkeypatch.setattr(r4, "_val_r4", lambda: fake_val)
    monkeypatch.setattr(r4, "champion_val_stats",
                        lambda run: {"directional": 0.0, "stable": 0.0})
    out = r4.stage_select_w(run)
    assert set(out["table"]) == {"W_best", "W_soup", "W_swa"}
    assert (run / "checkpoints" / "W_soup.pt").exists()
    assert out["winner"]["cell"] in {"W_best", "W_soup", "W_swa"}
    # SWA window off-by-one guard: epochs hold constants 0..19, so the
    # last-5 average must be exactly mean(15..19) = 17.0
    swa0 = torch.load(run / "checkpoints" / "W_swa_seed0.pt",
                      map_location="cpu", weights_only=True)["state_dict"]
    assert all(torch.allclose(v, torch.full_like(v, 17.0)) for v in swa0.values())


def test_stage_w_swa_skipped_without_epochs(tmp_path, monkeypatch):
    import torch

    import experiments.accuracy_sweep_r4 as r4
    from training.model import TrendCNN

    run = tmp_path
    a = r4.arm_r4("R_base", "ce")
    (run / "selection").mkdir()
    (run / "selection" / "R.json").write_text(json.dumps(
        {"winner": a, "champion_val": {"directional": 0.5, "stable": 0.5}}))
    for s in r4.ARM_SEEDS_R4:
        m = TrendCNN(num_classes=5, channels=(4, 6))
        p = r4._ckpt_r4(run, a, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": m.state_dict(), "arch": "cnn",
                    "in_channels": 1, "channels": [4, 6], "head": "ce"}, p)
    fake_val = r4.R4View(np.zeros((4, 1, 12), dtype=np.float32),
                         np.array([2, 2, 2, 2]))
    monkeypatch.setattr(r4, "_val_r4", lambda: fake_val)
    monkeypatch.setattr(r4, "champion_val_stats",
                        lambda run: {"directional": 0.0, "stable": 0.0})
    out = r4.stage_select_w(run)
    assert out["table"]["W_swa"] == {"skipped": "no epoch history"}
```

(Fix the first test's ckpt-save line at execution if the mkdir-or trick reads badly — expand to two statements.)

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v -k stage_w`
Expected: FAIL.

- [ ] **Step 3: Implement (append)**

```python
from experiments.accuracy_sweep_r2 import champion_val_stats  # top of file


def _save_averaged(run, name, dicts, meta_from):
    import torch
    from training.averaging import average_state_dicts
    avg = average_state_dicts(dicts)
    out = Path(run) / "checkpoints" / f"{name}.pt"
    base = torch.load(meta_from, map_location="cpu", weights_only=True)
    torch.save({"state_dict": avg, "arch": base["arch"],
                "in_channels": base["in_channels"],
                "channels": base["channels"], "head": base["head"]}, out)
    return out


def stage_select_w(run):
    import torch
    a = _read_sel(run, "R")["winner"]
    head = a["head"]
    champ = champion_val_stats(run)
    floor = champ["stable"] - 0.05
    table = {}

    per_seed = {str(s): _val_stats_r4(run, _ckpt_r4(run, a, s), head)
                for s in ARM_SEEDS_R4}
    best = best_seed_r4(per_seed, floor)
    table["W_best"] = {"per_seed": {s: {"directional": v[0], "stable": v[1]}
                                    for s, v in per_seed.items()},
                       "best_seed": best}

    seed_dicts = [torch.load(_ckpt_r4(run, a, s), map_location="cpu",
                             weights_only=True)["state_dict"]
                  for s in ARM_SEEDS_R4]
    soup_path = _save_averaged(run, "W_soup", seed_dicts, _ckpt_r4(run, a, 0))
    d, s_ = _val_stats_r4(run, soup_path, head)
    table["W_soup"] = {"directional": d, "stable": s_}

    epoch_dirs = {s: Path(run) / "checkpoints" / "epochs" / f"{a['name']}_seed{s}"
                  for s in ARM_SEEDS_R4}
    if all(ed.is_dir() and len(sorted(ed.glob("epoch*.pt"))) >= SWA_LAST_K
           for ed in epoch_dirs.values()):
        swa_stats = {}
        for s, ed in epoch_dirs.items():
            last = sorted(ed.glob("epoch*.pt"))[-SWA_LAST_K:]
            dicts = [torch.load(p, map_location="cpu", weights_only=True)
                     for p in last]
            p = _save_averaged(run, f"W_swa_seed{s}", dicts, _ckpt_r4(run, a, 0))
            swa_stats[str(s)] = _val_stats_r4(run, p, head)
        swa_best = best_seed_r4(swa_stats, floor)
        table["W_swa"] = {"per_seed": {s: {"directional": v[0], "stable": v[1]}
                                       for s, v in swa_stats.items()},
                          "best_seed": swa_best}
    else:
        table["W_swa"] = {"skipped": "no epoch history"}
        swa_stats, swa_best = None, None

    cells = {"W_best": per_seed[str(best)],
             "W_soup": (table["W_soup"]["directional"], table["W_soup"]["stable"])}
    if swa_stats:
        cells["W_swa"] = swa_stats[str(swa_best)]
    eligible = {k: v for k, v in cells.items() if v[1] >= floor}
    if not eligible:
        raise SystemExit("REFUSED: every W cell failed the stable floor")
    top = max(v[0] for v in eligible.values())
    order = ["W_best", "W_soup", "W_swa"]  # cost: no new artifact < soup < swa
    cell = next(k for k in order if k in eligible and top - eligible[k][0] <= 0.001)
    winner = dict(a, cell=cell,
                  best_seed=(best if cell == "W_best"
                             else swa_best if cell == "W_swa" else "soup"))
    out = {"stage": "W", "champion_val": champ, "stable_floor": floor,
           "table": table, "winner": winner}
    (Path(run) / "selection").mkdir(exist_ok=True)
    _write_json(Path(run) / "selection" / "W.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
    return out
```

and route it in `stage_select`:

```python
def stage_select(run_id, stage):
    run = _run_dir(run_id)
    if stage in ("H", "R"):
        return _select(run, stage, _arms_for(run, stage))
    if stage == "W":
        return stage_select_w(run)
    raise SystemExit(f"unknown stage {stage}")
```

- [ ] **Step 4: Run, verify pass**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r4.py tests/test_accuracy_sweep_r4.py
git commit -m "feat(r4): Stage W - soup/SWA cells with epoch-history skip rule"
```

### Task 11: convert, evaluate, analyze — head-aware widths, gates, protocol guard

**Files:**
- Modify: `experiments/accuracy_sweep_r4.py` (append; add `main()`)
- Test: `tests/test_accuracy_sweep_r4.py` (append)

**Interfaces:**
- Produces:
  - `_winner_ckpts(run) -> list[(label, Path)]` — the W winner's artifact set (one path for soup; five seed paths otherwise, confirmatory candidate = winning seed);
  - `_tflite_outputs_r4(path, x3, width)` — generalizes r3's `_tflite_logits_r3` to any output width (same INT8 dequant);
  - `stage_convert(run_id)` — converts the W winner (float + both calib INT8s) with `sample = torch.zeros(1, 1, 12)`;
  - `stage_evaluate(run_id)` — REFUSES unless `protocol_is_committed(PROTOCOL_R4)`; writes held_out/temporal winner outputs + champion preds + meta, and PEDAP preds (decode via `decode_outputs`);
  - `gate_results_r4(...)` — identical to r3's gates but `BOOT_SEED_R4`; candidate preds come pre-decoded (class indices), champion preds argmaxed as today;
  - `stage_analyze(run_id)` — aggregate JSON `analysis/sweep_r4_aggregate.json` with `exp0`, selection trail (H, R, W), smoke, per-arm train windows, components (held_out/temporal: pairing block, macro-F1, gates), PEDAP zero-shot, ship block, deviations list;
  - `stage_smoke` / `stage_calib` — delegate to r2's, as r3 does;
  - `main()` — argparse: `stage` in `exp0|smoke|train|select|calib|convert|evaluate|analyze`, `--run-id sweep-r4`, `--stage-name H|R|W`.
- Consumes: everything above; r1 `CHAMPION`, `_component_windows`, `_tflite_logits`, `protocol_is_committed`; r3's conversion recipe.

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_boot_seed_wired_into_gates(monkeypatch):
    import experiments.accuracy_sweep_r4 as r4
    captured = {}

    def fake_boot(y, pids, pairs, metric, n_boot, seed):
        captured["seed"] = seed
        return {"delta": 0.01, "ci95": [0.001, 0.02]}
    monkeypatch.setattr(
        "experiments.analyze_calibration_coverage.paired_cluster_bootstrap",
        fake_boot)
    y = np.array([0, 1, 2, 3, 4] * 4)
    pids = np.array(["a", "b"] * 10)
    preds = y.copy()
    out = r4.gate_results_r4(y, pids, preds, preds, preds, n_boot=10)
    assert captured["seed"] == r4.BOOT_SEED_R4
    assert out["gate1_beats_champion"] and out["gate3_int8_within_half_pp_of_float"]


def test_evaluate_refuses_without_protocol(monkeypatch):
    import experiments.accuracy_sweep_r4 as r4
    monkeypatch.setattr(r4, "protocol_is_committed", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        r4.stage_evaluate("sweep-r4-test-nonexistent")


def test_tflite_width_param_signature():
    # pure signature check: callable takes (path, x3, width); no litert import
    import inspect

    import experiments.accuracy_sweep_r4 as r4
    sig = inspect.signature(r4._tflite_outputs_r4)
    assert list(sig.parameters) == ["path", "x3", "width"]
```

- [ ] **Step 2: Run, verify fail**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r4.py -v -k "boot or refuses or width"`
Expected: FAIL.

- [ ] **Step 3: Implement (append)**

```python
def gate_results_r4(y, pids, cand_int8_preds, cand_float_preds,
                    champ_int8_preds, n_boot=2000):
    import experiments.analyze_calibration_coverage as acc
    contrast = acc.paired_cluster_bootstrap(
        y, pids, [(cand_int8_preds, champ_int8_preds)],
        metric=directional_recall, n_boot=n_boot, seed=BOOT_SEED_R4)
    ci_low = contrast["ci95"][0] if contrast["ci95"] else None
    d_float = directional_recall(acc.confusion_counts(y, cand_float_preds))
    d_int8 = directional_recall(acc.confusion_counts(y, cand_int8_preds))
    s_cand = stable_recall(acc.confusion_counts(y, cand_int8_preds))
    s_champ = stable_recall(acc.confusion_counts(y, champ_int8_preds))
    return {
        "contrast_int8_vs_champion": contrast,
        "candidate_float_directional": d_float,
        "candidate_int8_directional": d_int8,
        "champion_int8_directional": directional_recall(
            acc.confusion_counts(y, champ_int8_preds)),
        "candidate_int8_stable": s_cand,
        "champion_int8_stable": s_champ,
        "gate1_beats_champion": bool(contrast["delta"] is not None
                                     and contrast["delta"] > 0
                                     and ci_low is not None and ci_low > -0.005),
        "gate2_stable_floor": bool(s_cand >= s_champ - 0.02),
        "gate3_int8_within_half_pp_of_float": bool(d_float - d_int8 <= 0.005),
    }
```

then the remaining stages, following r3's `stage_convert` / `stage_evaluate` / `stage_analyze` line by line with these deltas (write them out fully at execution, copying r3's structure from `experiments/accuracy_sweep_r3.py:424-575`):

- `_tflite_outputs_r4(path, x3, width)` = r3's `_tflite_logits_r3` with `logits = np.empty((len(x3), width), ...)` and no other change.
- winner artifacts: `cell == "soup"` → one artifact set from `checkpoints/W_soup.pt` (names `winner_soup_*`); otherwise five per-seed sets as r3, candidate = `W.json`'s `best_seed`; `cell == "W_swa"` → the `W_swa_seed{s}.pt` checkpoints.
- load with `load_checkpoint(path, head_outputs(head))`.
- eval decode: winner outputs saved raw (`*_outputs.npy`), preds derived `decode_outputs(head, outputs)` in analyze; champion unchanged (`argmax`).
- PEDAP: `_cohort_cache_r4(HOLDOUT_COHORT)` levels → delta windows → winner decode + champion argmax.
- `stage_analyze` adds `"exp0": json.loads(EXP0_PATH.read_text())` and `"deviations": list(DEVIATIONS_R4)` (start the tuple empty; append real ones during the run).
- `main()`:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["exp0", "smoke", "train", "select",
                                          "calib", "convert", "evaluate",
                                          "analyze"])
    parser.add_argument("--run-id", default="sweep-r4")
    parser.add_argument("--stage-name", choices=["H", "R", "W"])
    args = parser.parse_args()
    if args.stage == "exp0":
        stage_exp0(args.run_id)
    elif args.stage == "smoke":
        from experiments.accuracy_sweep_r2 import stage_smoke as smoke_r2
        smoke_r2(args.run_id)
    elif args.stage == "train":
        stage_train(args.run_id, args.stage_name)
    elif args.stage == "select":
        stage_select(args.run_id, args.stage_name)
    elif args.stage == "calib":
        from experiments.accuracy_sweep_r2 import stage_calib as calib_r2
        calib_r2(args.run_id)
    elif args.stage == "convert":
        stage_convert(args.run_id)
    elif args.stage == "evaluate":
        stage_evaluate(args.run_id)
    else:
        stage_analyze(args.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the whole suite**

Run: `venv/bin/python -m pytest -v`
Expected: ALL PASS (repo-wide; no CGM data present).

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r4.py tests/test_accuracy_sweep_r4.py
git commit -m "feat(r4): convert/evaluate/analyze - head-aware widths, gates, protocol guard, CLI"
```

### Task 12: LiteRT conversion smoke for new heads

**Files:**
- Test: `tests/test_model_r4.py` (create)

**Interfaces:**
- Consumes: `training.model.TrendCNN`, `conversion.convert.convert_float`.
- Produces: CI-proof that 1-output and 4-output TrendCNN graphs convert.

- [ ] **Step 1: Write the test (mirroring `tests/test_model_r3.py`'s conversion-smoke pattern — open that file first and copy its litert import guard/skip marker exactly)**

```python
import numpy as np
import pytest
import torch

from training.model import TrendCNN


@pytest.mark.parametrize("width", [1, 4])
def test_head_width_forward_shapes(width):
    m = TrendCNN(num_classes=width, channels=(96, 144))
    out = m(torch.zeros(2, 1, 12))
    assert out.shape == (2, width)


@pytest.mark.parametrize("width", [1, 4])
def test_litert_conversion_smoke_new_heads(tmp_path, width):
    litert_torch = pytest.importorskip("litert_torch")  # match test_model_r3's guard
    from conversion.convert import convert_float
    m = TrendCNN(num_classes=width, channels=(96, 144)).eval()
    out = tmp_path / f"w{width}.tflite"
    convert_float(m, torch.zeros(1, 1, 12), out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run**

Run: `venv/bin/python -m pytest tests/test_model_r4.py -v`
Expected: shape tests PASS; conversion tests PASS where litert_torch is installed (training venv), SKIP otherwise — same behavior as the r3 smoke tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_r4.py
git commit -m "test(r4): forward-shape + LiteRT conversion smoke for 1/4-output heads"
```

### Task 13: Run the sweep (operator task — real data, MPS box)

**Files:**
- Create (generated, partly committed): `results/calibration-coverage/sweep-r4/**` (gitignored), `paper/results/calibration-coverage/exp0_sweep_r4.json`, `paper/results/calibration-coverage/protocol_sweep_r4.json`, `paper/results/calibration-coverage/accuracy_sweep_r4.json`.

This task is sequential shell work, not TDD. Every step records its output in the run dir.

- [ ] **Step 1: Experiment 0**

```bash
venv/bin/python -m experiments.accuracy_sweep_r4 exp0
git add paper/results/calibration-coverage/exp0_sweep_r4.json
git commit -m "results(r4): Experiment 0 label-noise ceiling + abort verdict"
```

- [ ] **Step 2: STOP if aborted**

If `exp0_sweep_r4.json` has `"abort": true`: skip to Step 8 (analyze publishes Exp0 alone), write the paper subsection as the abort narrative, and end the sweep. The gate is binding — do not train "just to see".

- [ ] **Step 3: Smoke (frozen batch) + H_ce reuse copy**

```bash
venv/bin/python -m experiments.accuracy_sweep_r4 smoke
# copy r3 D_base checkpoints as H_ce (bit-identical smoke rule, spec section 3)
venv/bin/python - <<'EOF'
import torch, json
from pathlib import Path
from experiments.calibration_coverage import _run_dir
src = Path(_run_dir("sweep-r3")) / "checkpoints"
dst = Path(_run_dir("sweep-r4")) / "checkpoints"
dst.mkdir(parents=True, exist_ok=True)
for s in range(5):
    obj = torch.load(src / f"D5_none_cnn_seed{s}.pt", map_location="cpu", weights_only=True)
    obj["head"] = "ce"
    torch.save(obj, dst / f"H_ce_seed{s}.pt")
    print("copied seed", s)
EOF
```

(Verify the r3 checkpoint filename stem on disk first — `ls results/calibration-coverage/sweep-r3/checkpoints/` — and adjust the `D5_none_cnn` stem if it differs.)

- [ ] **Step 4: Stage H train + select**

```bash
venv/bin/python -m experiments.accuracy_sweep_r4 train --stage-name H
venv/bin/python -m experiments.accuracy_sweep_r4 select --stage-name H
```

- [ ] **Step 5: Stage R (copy winner checkpoints as R_base, then train/select)**

```bash
venv/bin/python - <<'EOF'
import json, shutil
from pathlib import Path
from experiments.calibration_coverage import _run_dir
run = Path(_run_dir("sweep-r4"))
w = json.loads((run / "selection" / "H.json").read_text())["winner"]["name"]
for s in range(5):
    shutil.copy2(run / "checkpoints" / f"{w}_seed{s}.pt",
                 run / "checkpoints" / f"R_base_seed{s}.pt")
    ep = run / "checkpoints" / "epochs" / f"{w}_seed{s}"
    if ep.is_dir():
        shutil.copytree(ep, run / "checkpoints" / "epochs" / f"R_base_seed{s}",
                        dirs_exist_ok=True)
print("R_base <-", w)
EOF
venv/bin/python -m experiments.accuracy_sweep_r4 train --stage-name R
venv/bin/python -m experiments.accuracy_sweep_r4 select --stage-name R
```

- [ ] **Step 6: Stage W, calibration, conversion**

```bash
venv/bin/python -m experiments.accuracy_sweep_r4 select --stage-name W
venv/bin/python -m experiments.accuracy_sweep_r4 calib
venv/bin/python -m experiments.accuracy_sweep_r4 convert
```

- [ ] **Step 7: Freeze protocol, THEN evaluate**

Write `paper/results/calibration-coverage/protocol_sweep_r4.json` by hand from the template below, commit it, verify the guard refuses before the commit and runs after:

```json
{
  "run_id": "sweep-r4",
  "spec": "docs/superpowers/specs/2026-10-13-accuracy-sweep-r4-design.md",
  "frozen_at_commit": "<git rev-parse HEAD>",
  "exp0": "<copy of exp0_sweep_r4.json content>",
  "selection_trail_pointer": "results/calibration-coverage/sweep-r4/selection/",
  "winner": "<selection/W.json winner block verbatim>",
  "champion_sha256": {"float": "<sha256 of bundled trend_float.tflite>",
                      "int8": "<sha256 of bundled trend_int8.tflite>"},
  "prediction": "winner >= champion on held-out directional recall",
  "primary_endpoint": "held_out directional recall, winner INT8 (uniform_random_200_d101) vs champion INT8",
  "gates": ["delta>0 AND bootstrap CI lower > -0.5pp",
            "winner INT8 stable >= champion INT8 stable - 2pp",
            "winner INT8 within 0.5pp of own float (directional)"],
  "bootstrap": {"kind": "paired participant-cluster", "draws": 2000,
                "seed": 20261014},
  "pairing_rule": "identical eligible windows by construction; dropped counts reported"
}
```

```bash
venv/bin/python -m experiments.accuracy_sweep_r4 evaluate   # must REFUSE pre-commit
git add paper/results/calibration-coverage/protocol_sweep_r4.json
git commit -m "protocol(r4): freeze before the single confirmatory read"
venv/bin/python -m experiments.accuracy_sweep_r4 evaluate
```

- [ ] **Step 8: Analyze + publish aggregates**

```bash
venv/bin/python -m experiments.accuracy_sweep_r4 analyze
cp results/calibration-coverage/sweep-r4/analysis/sweep_r4_aggregate.json \
   paper/results/calibration-coverage/accuracy_sweep_r4.json
# participant-leak walk (the same check r2/r3 ran before committing):
venv/bin/python - <<'EOF'
import json, re, sys
d = json.load(open("paper/results/calibration-coverage/accuracy_sweep_r4.json"))
blob = json.dumps(d)
for pat in (r'"participants"', r'"keys"', r'"labels":\s*\['):
    if re.search(pat, blob):
        sys.exit(f"LEAK: {pat} present in published aggregates")
print("leak walk clean")
EOF
git add paper/results/calibration-coverage/accuracy_sweep_r4.json
git commit -m "results(r4): published aggregates"
```

- [ ] **Step 9: Record deviations**

Anything that differed from spec during the run goes into `DEVIATIONS_R4` in `experiments/accuracy_sweep_r4.py`, analyze is re-run, and the published JSON re-copied (same commit discipline r3 used).

### Task 14: paper subsection + memory update

**Files:**
- Modify: `paper/` manuscript (the r3 sweep subsection's sibling — locate the r3 text with `grep -rn "sweep r3\|r3 sweep" paper/*.tex` and mirror its structure)
- Modify: `/Users/mohamednabil/.claude/projects/-Users-mohamednabil-Documents-vibe-GlucoEdge/memory/project_status.md`

- [ ] **Step 1: Write the r4 subsection**

Content, in r3's register: the measured error structure that motivated r4 (adjacent-error share, slow→fast inflation, seed spread vs ensemble), Experiment 0 estimator + ceiling + verdict (this is novel — give it its own paragraph), stage tables (or the abort statement), confirmatory contrast with CI and gate outcomes if run, PEDAP row, the scope-reversal note (objective family reopened under the changed endpoint), val-only selection + single-read discipline sentence. One table. Page budget 15 — trim elsewhere if it spills.

- [ ] **Step 2: Build the paper**

Run: `cd paper && tectonic main.tex` (match the repo's existing build invocation — check `paper/` for a Makefile or README first).
Expected: clean build, 15 pages or fewer.

- [ ] **Step 3: Update memory**

Rewrite the "what's next" block of `memory/project_status.md`: r4 designed + run (or aborted at Exp0), outcome numbers, ship decision pending user.

- [ ] **Step 4: Commit**

```bash
git add paper/ 
git commit -m "docs(paper): accuracy sweep r4 subsection"
```

- [ ] **Step 5: PR**

Open the research PR for `feat/accuracy-sweep-r4` (research only — any ship PR is separate and explicitly user-approved, per spec section 5). PR description summarizes Exp0 verdict, stage winners, gate outcomes, and links spec + plan.

---

## Execution notes

- Tasks 1–12 are pure code + tests: no CGM data, runnable anywhere, each independently committable.
- Task 13 needs GlucoBench + DiaData present as sibling dirs and the MPS box; it is operator-paced (hours of training) and halts at two user-visible gates: the Exp0 abort verdict and the ship decision.
- Task 14 depends on Task 13's numbers.
- If Exp0 aborts: Tasks 9–12 remain merged as tested-but-unexercised infrastructure (they are cheap, already reviewed, and r5 may reuse them); Task 13 stops at Step 2; Task 14 writes the abort narrative. The spec treats the honest null as a publishable outcome.
