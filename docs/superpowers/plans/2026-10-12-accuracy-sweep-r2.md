# Accuracy Sweep r2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Data-mixture × capacity sweep on DiaData + weinstock with a directional-recall endpoint, one pre-registered confirmatory read, and frozen ship gates.

**Architecture:** A new `experiments/accuracy_sweep_r2.py` drives the stages, importing r1's proven pieces (`paired_rows`, `_tflite_logits`, `_component_windows`, `_data`, `_val_logits`, protocol-guard pattern) and the DiaData loader; `training/mixtures.py` holds the pure subject-sampler and oversampler; arms train on pre-materialized tensors via `train_model` with new device/batch knobs.

**Tech Stack:** Python 3.12, torch (MPS), ai-edge-litert, existing training/ + experiments/ + conversion/ packages, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-accuracy-sweep-r2-design.md`

## Global Constraints

- Not a medical device: no code, comment, or output string implies clinical validity or treatment guidance.
- GlucoBench and DiaData read-only; participant IDs/timestamps/keys/per-window predictions stay in gitignored `results/`.
- held_out/temporal labels read exactly once, after `protocol_sweep_r2.json` is committed clean; selection on weinstock val only.
- Suite runs without GlucoBench/DiaData. Tests via `venv/bin/python -m pytest ...`.
- Run id `sweep-r2`; branch `feat/accuracy-sweep-r2`; seeds: arms (0,1,2), final (0..4); EPOCHS=20; Adam lr 1e-3; window 12; delta-anchor encoding; loss = class-weighted CE.
- Fixed class indices (FIVE_CLASSES order): falling_fast=0, falling=1, stable=2, rising=3, rising_fast=4; fast = {0,4}; directional = all but 2.
- Window cap per arm: 15,000,000. PEDAP in no training mixture.
- r1 entry points (`experiments.accuracy_sweep` CLI and tests) must keep working.

## Review Focus

1. A mixture whose sampled DiaData subjects yield zero fast-class windows — oversampler must no-op, not crash (test in Task 2).
2. The stable-recall disqualifier must never *promote* a disqualified arm when all arms are disqualified — stage select must fail loudly instead (test in Task 4).
3. MPS unavailable (CI, other machines) — device helper falls back to CPU without changing results semantics (test in Task 1).
4. Champion stable recall computed on val must use the same eligible windows as arm selection, or the disqualifier compares different denominators (pinned by code path: both go through `_val_windows_cached`; test in Task 4 uses one synthetic val set for both).
5. Tensor cache reuse across arms with different seeds — cache key must include arm name AND sampling seed, or arm reruns silently reuse stale mixtures (test in Task 4 on cache path naming).

---

### Task 1: `train_model` batch/device knobs

**Files:**
- Modify: `training/train.py:68-94`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: existing `train_model(train_ds, val_ds, num_classes, epochs, device, model=None, criterion=None)`.
- Produces: `train_model(..., batch_size=64)` (train loader batch; val loader uses `max(batch_size, 64)`); `resolve_device(prefer: str = "mps") -> torch.device` in `training/train.py` (mps if available else cpu).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_train.py`)

```python
def test_train_model_accepts_batch_size():
    import torch
    from training.model import TrendCNN
    from training.train import train_model

    class Tiny:
        labels = [0, 1, 2, 3, 4] * 4
        def __len__(self): return 20
        def __getitem__(self, i):
            return torch.zeros(1, 12), self.labels[i]

    model = train_model(Tiny(), Tiny(), 5, 1, torch.device("cpu"),
                        model=TrendCNN(num_classes=5), batch_size=5)
    assert isinstance(model, TrendCNN)


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    import torch
    from training.train import resolve_device
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("mps").type == "cpu"
    assert resolve_device("cpu").type == "cpu"
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_train.py -k "batch_size or resolve_device" -v`
Expected: FAIL — `TypeError: train_model() got an unexpected keyword argument 'batch_size'` / ImportError on `resolve_device`.

- [ ] **Step 3: Implement** — in `training/train.py`: signature becomes `def train_model(train_ds, val_ds, num_classes, epochs, device, model=None, criterion=None, batch_size=64):`; `train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)`; `val_loader = DataLoader(val_ds, batch_size=max(batch_size, 64))`. Add:

```python
def resolve_device(prefer: str = "mps") -> "torch.device":
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_train.py -v`
Expected: all PASS (old callers unaffected: defaults unchanged).

- [ ] **Step 5: Full suite + commit**

Run: `venv/bin/python -m pytest -q` — Expected: all pass.

```bash
git add training/train.py tests/test_train.py
git commit -m "feat(training): batch_size knob and MPS device resolver for train_model"
```

### Task 2: `training/mixtures.py` — subject sampler + oversampler

**Files:**
- Create: `training/mixtures.py`
- Test: `tests/test_mixtures.py`

**Interfaces:**
- Produces: `TRAIN_COHORTS = ("CITY", "DLCP3", "RBG", "SENCE", "WISDM")`, `HOLDOUT_COHORT = "PEDAP"`, `CAP = 15_000_000`, `FAST_CLASSES = (0, 4)`;
  `sample_subjects(counts: dict[str, int], budget: int, seed: int) -> tuple[list[str], int]`;
  `oversample_extra_indices(labels, seed: int = 0) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
"""Mixture sampler tests — pure, no data needed."""
import numpy as np

from training.mixtures import (CAP, FAST_CLASSES, HOLDOUT_COHORT,
                               TRAIN_COHORTS, oversample_extra_indices,
                               sample_subjects)


def test_pedap_is_not_a_training_cohort():
    assert HOLDOUT_COHORT == "PEDAP"
    assert "PEDAP" not in TRAIN_COHORTS
    assert CAP == 15_000_000


def test_sample_subjects_whole_subjects_until_budget():
    counts = {f"s{i}": 100 for i in range(10)}
    ids, total = sample_subjects(counts, 250, seed=0)
    assert total == 300            # 3 whole subjects; never a partial subject
    assert len(ids) == 3
    ids2, _ = sample_subjects(counts, 250, seed=0)
    assert ids == ids2             # seeded determinism
    ids3, _ = sample_subjects(counts, 250, seed=1)
    assert ids != ids3             # seed changes the draw (10!/(7!3!) ways; collision ~0)


def test_sample_subjects_budget_bigger_than_data_takes_all():
    counts = {"a": 5, "b": 7}
    ids, total = sample_subjects(counts, 10_000, seed=0)
    assert sorted(ids) == ["a", "b"] and total == 12


def test_oversample_reaches_mean_of_other_classes():
    labels = np.array([0] * 10 + [1] * 100 + [2] * 130 + [3] * 70 + [4] * 50)
    extra = oversample_extra_indices(labels, seed=0)
    target = 100  # mean of classes 1,2,3
    new_counts = {c: (labels[extra] == c).sum() + (labels == c).sum() for c in range(5)}
    assert new_counts[0] == target and new_counts[4] == target
    assert all(labels[i] in FAST_CLASSES for i in extra)


def test_oversample_noops_without_fast_windows_or_when_already_enough():
    assert oversample_extra_indices(np.array([1, 2, 3, 2, 1])).size == 0
    assert oversample_extra_indices(np.array([0] * 90 + [1, 2, 3] * 10)).size == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_mixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.mixtures'`.

- [ ] **Step 3: Implement**

```python
"""Pure mixture helpers for sweep r2: subject-level sampling and
fast-class oversampling. No I/O here."""
import numpy as np

TRAIN_COHORTS = ("CITY", "DLCP3", "RBG", "SENCE", "WISDM")  # observed Database names
HOLDOUT_COHORT = "PEDAP"  # reserved for the zero-shot secondary eval
CAP = 15_000_000
FAST_CLASSES = (0, 4)     # falling_fast, rising_fast in FIVE_CLASSES order


def sample_subjects(counts, budget, seed):
    """Draw whole subjects (shuffled by seed) until the window budget is
    met or exceeded. A subject's windows are never split."""
    rng = np.random.default_rng(seed)
    ids = sorted(counts)
    rng.shuffle(ids)
    out, total = [], 0
    for sid in ids:
        if total >= budget:
            break
        out.append(sid)
        total += counts[sid]
    return out, total


def oversample_extra_indices(labels, seed=0):
    """Indices to append so each fast class reaches the mean count of the
    other three classes. Empty when nothing to do."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    counts = {c: int((labels == c).sum()) for c in range(5)}
    target = int(np.mean([counts[c] for c in range(5) if c not in FAST_CLASSES]))
    extra = []
    for c in FAST_CLASSES:
        pool = np.flatnonzero(labels == c)
        if pool.size == 0 or counts[c] >= target:
            continue
        extra.append(rng.choice(pool, target - counts[c], replace=True))
    return np.concatenate(extra) if extra else np.array([], dtype=np.int64)
```

- [ ] **Step 4: Run tests** — `venv/bin/python -m pytest tests/test_mixtures.py -v` — Expected: 5 PASS.

- [ ] **Step 5: Full suite + commit**

```bash
git add training/mixtures.py tests/test_mixtures.py
git commit -m "feat(training): mixture subject sampler and fast-class oversampler for sweep r2"
```

### Task 3: r2 metrics + protocol-guard path parameter

**Files:**
- Modify: `experiments/accuracy_sweep.py:327-339` (guard takes a path)
- Create: `experiments/accuracy_sweep_r2.py` (metrics + gates only in this task)
- Test: `tests/test_accuracy_sweep_r2.py`, `tests/test_accuracy_sweep.py` (guard still green)

**Interfaces:**
- Consumes: `confusion_counts`, `minority_macro_recall`, `paired_cluster_bootstrap` from `experiments/analyze_calibration_coverage`; r1's `protocol_is_committed`.
- Produces: in `accuracy_sweep.py`: `protocol_is_committed(path=None)` (None → r1 PROTOCOL; module-global monkeypatching still works).
  In `accuracy_sweep_r2.py`: `PROTOCOL_R2 = GLUCOEDGE_ROOT / "paper/results/calibration-coverage/protocol_sweep_r2.json"`; `stable_recall(cm) -> float`; `directional_recall(cm) -> float` (alias of minority_macro_recall); `gate_results_r2(y, pids, cand_int8, cand_float, champ_int8, n_boot=2000) -> dict` with keys `contrast_int8_vs_champion`, `candidate_{float,int8}_directional`, `champion_int8_directional`, `candidate_int8_stable`, `champion_int8_stable`, `gate1_beats_champion` (delta>0 and ci_low>-0.005), `gate2_stable_floor` (cand_stable >= champ_stable - 0.02), `gate3_int8_within_half_pp_of_float` (float−int8 ≤ 0.005, directional).

- [ ] **Step 1: Write the failing tests**

```python
"""Sweep r2 pure-logic tests — no data needed."""
import numpy as np
import pytest

from experiments.accuracy_sweep_r2 import (directional_recall, gate_results_r2,
                                           stable_recall)


def _cm(y, pred):
    from experiments.analyze_calibration_coverage import confusion_counts
    return confusion_counts(np.asarray(y), np.asarray(pred))


def test_stable_and_directional_recall_hand_computed():
    y =    [0, 0, 1, 2, 2, 2, 3, 4]
    pred = [0, 2, 1, 2, 2, 0, 1, 4]
    cm = _cm(y, pred)
    assert stable_recall(cm) == pytest.approx(2 / 3)
    # per-class recalls: c0=1/2, c1=1, c3=0, c4=1 -> directional = 0.625
    assert directional_recall(cm) == pytest.approx(0.625)


def test_gate_results_r2_known_outcomes():
    y = np.array([0, 1, 2, 3, 4] * 40)
    pids = np.array([f"p{i % 4}" for i in range(200)])
    perfect = y.copy()
    stable_killer = y.copy()
    stable_killer[y == 2] = 1          # perfect directional, zero stable recall
    worse = y.copy()
    worse[:60] = (worse[:60] + 1) % 5
    g = gate_results_r2(y, pids, cand_int8=perfect, cand_float=perfect,
                        champ_int8=worse, n_boot=200)
    assert g["gate1_beats_champion"] and g["gate2_stable_floor"] \
        and g["gate3_int8_within_half_pp_of_float"]
    g2 = gate_results_r2(y, pids, cand_int8=stable_killer, cand_float=stable_killer,
                         champ_int8=perfect, n_boot=200)
    assert g2["gate2_stable_floor"] is False


def test_protocol_guard_accepts_explicit_path(tmp_path):
    import subprocess
    from experiments.accuracy_sweep import protocol_is_committed
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    p = tmp_path / "protocol_sweep_r2.json"
    assert not protocol_is_committed(p)
    p.write_text("{}")
    subprocess.run(["git", "-C", str(tmp_path), "add", p.name], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-qm", "freeze"], check=True)
    assert protocol_is_committed(p)
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r2.py -v`
Expected: FAIL — no module `experiments.accuracy_sweep_r2`; guard TypeError on path arg.

- [ ] **Step 3: Implement.** In `accuracy_sweep.py`, change the guard (body otherwise identical, `PROTOCOL` replaced by `path`):

```python
def protocol_is_committed(path=None):
    """True iff the protocol file is git-tracked and identical to its
    committed content."""
    path = PROTOCOL if path is None else path
    if not path.exists():
        return False
    cwd = str(path.parent)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path.name],
                             cwd=cwd, capture_output=True)
    if tracked.returncode != 0:
        return False
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path.name],
                           cwd=cwd, capture_output=True)
    return dirty.returncode == 0
```

New `experiments/accuracy_sweep_r2.py` (this task's portion):

```python
"""Accuracy sweep r2: data-mixture x capacity on DiaData + weinstock,
directional-recall endpoint, one pre-registered confirmatory read.
Research code on public benchmark data - not a medical device."""
import json
import time
from pathlib import Path

import numpy as np

from experiments.calibration_coverage import GLUCOEDGE_ROOT, _run_dir, _write_json

PROTOCOL_R2 = GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage" / "protocol_sweep_r2.json"
STABLE = 2  # FIVE_CLASSES index of "stable"


def stable_recall(cm):
    cm = np.asarray(cm, dtype=float)
    denom = cm[STABLE].sum()
    return float(cm[STABLE, STABLE] / denom) if denom else 0.0


def directional_recall(cm):
    from experiments.analyze_calibration_coverage import minority_macro_recall
    return minority_macro_recall(cm)


def gate_results_r2(y, pids, cand_int8, cand_float, champ_int8, n_boot=2000):
    from experiments.analyze_calibration_coverage import (
        confusion_counts, paired_cluster_bootstrap)
    contrast = paired_cluster_bootstrap(y, pids, [(cand_int8, champ_int8)],
                                        metric=directional_recall, n_boot=n_boot)
    ci_low = contrast["ci95"][0] if contrast["ci95"] else None
    d_float = directional_recall(confusion_counts(y, cand_float))
    d_int8 = directional_recall(confusion_counts(y, cand_int8))
    s_cand = stable_recall(confusion_counts(y, cand_int8))
    s_champ = stable_recall(confusion_counts(y, champ_int8))
    return {
        "contrast_int8_vs_champion": contrast,
        "candidate_float_directional": d_float,
        "candidate_int8_directional": d_int8,
        "champion_int8_directional": directional_recall(confusion_counts(y, champ_int8)),
        "candidate_int8_stable": s_cand,
        "champion_int8_stable": s_champ,
        "gate1_beats_champion": bool(contrast["delta"] is not None
                                     and contrast["delta"] > 0
                                     and ci_low is not None and ci_low > -0.005),
        "gate2_stable_floor": bool(s_cand >= s_champ - 0.02),
        "gate3_int8_within_half_pp_of_float": bool(d_float - d_int8 <= 0.005),
    }
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r2.py tests/test_accuracy_sweep.py -v`
Expected: new tests PASS; r1 guard tests still PASS (monkeypatched PROTOCOL global keeps working because None falls back to the module global at call time).

- [ ] **Step 5: Full suite + commit**

```bash
git add experiments/accuracy_sweep.py experiments/accuracy_sweep_r2.py tests/test_accuracy_sweep_r2.py
git commit -m "feat(experiments): r2 directional metrics, three ship gates, parameterized protocol guard"
```

### Task 4: r2 arms, tensor materialization, smoke/train/select stages

**Files:**
- Modify: `experiments/accuracy_sweep_r2.py` (append)
- Test: `tests/test_accuracy_sweep_r2.py` (append)

**Interfaces:**
- Consumes: Task 1 `train_model(..., batch_size)`, `resolve_device`; Task 2 mixtures API; Task 3 metrics; r1's `_data`, `_val_logits`, `pick_winner`, `TIE_BAND`, `param_count` (import from `experiments.accuracy_sweep`); `load_diadata` + `GlucoseTrendDataset`; `delta_windows`.
- Produces: `ARM_SEEDS=(0,1,2)`, `FINAL_SEEDS=(0,1,2,3,4)`, `EPOCHS=20`, `MIXTURES=("M0","M1","M2","M3","M4")`, `stage_m_arms()`, `stage_c_arms(winner)`, `arm_r2(name, mixture, channels)` (adds `"loss": "ce", "length": 12, "pooled": False` so r1 `cost_key`/selection formats line up); `TensorView` (X, y tensors; `.labels`); `build_arm_tensors(run, a, sample_seed) -> Path` (cache `run/tensors/{name}_s{sample_seed}.npz`); `stage_smoke(run_id)`; `stage_train(run_id, stage)`; `stage_select(run_id, stage)` (directional recall + champion-floor disqualifier; raises `SystemExit` if every arm is disqualified); `champion_val_stats() -> dict` (champion INT8 directional + stable recall on the eligible weinstock val set, cached in `run/champion_val.json`); CLI `main()` with stages `smoke|train|select|calib|convert|evaluate|analyze` and `--stage-name M|C|S`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_stage_m_arms_cover_five_mixtures_at_96x144():
    from experiments.accuracy_sweep_r2 import stage_m_arms
    arms = stage_m_arms()
    assert [a["mixture"] for a in arms] == ["M0", "M1", "M2", "M3", "M4"]
    assert all(a["channels"] == [96, 144] and a["length"] == 12 for a in arms)


def test_stage_c_arms_bracket_winner_capacity():
    from experiments.accuracy_sweep_r2 import arm_r2, stage_c_arms
    w = arm_r2("M_M1", "M1", (96, 144))
    names = [a["channels"] for a in stage_c_arms(w)]
    assert names == [[24, 36], [192, 288]]
    assert all(a["mixture"] == "M1" for a in stage_c_arms(w))


def test_tensor_cache_key_includes_arm_and_sample_seed(tmp_path):
    from experiments.accuracy_sweep_r2 import _tensor_path, arm_r2
    a = arm_r2("M_M1", "M1", (96, 144))
    p1 = _tensor_path(tmp_path, a, sample_seed=0)
    p2 = _tensor_path(tmp_path, a, sample_seed=1)
    assert p1 != p2 and "M_M1" in p1.name


def test_select_disqualifies_stable_killers_and_fails_if_all_do(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r2 as r2
    arms = [r2.arm_r2("M_M0", "M0", (96, 144)), r2.arm_r2("M_M1", "M1", (96, 144))]
    # arm M0: best directional but stable 0.30; M1: lower directional, stable fine
    stats = {"M_M0": (0.60, 0.30), "M_M1": (0.50, 0.55)}
    monkeypatch.setattr(r2, "_val_stats", lambda run, a, seed: stats[a["name"]])
    monkeypatch.setattr(r2, "champion_val_stats", lambda run=None: {"directional": 0.45, "stable": 0.50})
    out = r2._select_from_arms(tmp_path, "M", arms)
    assert out["winner"]["name"] == "M_M1"
    assert out["table"]["M_M0"]["disqualified"] is True
    both_bad = {"M_M0": (0.60, 0.30), "M_M1": (0.50, 0.30)}
    monkeypatch.setattr(r2, "_val_stats", lambda run, a, seed: both_bad[a["name"]])
    with pytest.raises(SystemExit):
        r2._select_from_arms(tmp_path, "M", arms)
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r2.py -v`
Expected: new tests FAIL on missing names.

- [ ] **Step 3: Implement** (append to `accuracy_sweep_r2.py`):

```python
ARM_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 20
MIXTURES = ("M0", "M1", "M2", "M3", "M4")
SAMPLE_SEED = 20261012  # one fixed subject-draw seed; training seeds vary per run


def arm_r2(name, mixture, channels):
    return {"name": name, "mixture": mixture, "channels": list(channels),
            "length": 12, "loss": "ce", "pooled": False}


def stage_m_arms():
    return [arm_r2(f"M_{m}", m, (96, 144)) for m in MIXTURES]


def stage_c_arms(winner):
    return [arm_r2(f"C_c{c1}x{c2}", winner["mixture"], (c1, c2))
            for c1, c2 in ((24, 36), (192, 288))]


class TensorView:
    """Window tensors + labels; what train_model and class_weights need."""

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.labels = y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, k):
        import torch
        return torch.from_numpy(self.x[k]).unsqueeze(0), int(self.y[k])


def _tensor_path(run, a, sample_seed):
    return Path(run) / "tensors" / f"{a['name']}_s{sample_seed}.npz"


_DIA_CACHE = {}


def _diadata_windows(cohort):
    """(windows float32 [n,12], labels int64, subject ids) for one cohort,
    delta-anchor encoded. Cached per process."""
    if cohort not in _DIA_CACHE:
        from training.dataset import GlucoseTrendDataset
        from training.deltas import delta_windows
        from training.diadata import load_diadata
        from training.labeling import FIVE_CLASSES
        frame = load_diadata(GLUCOEDGE_ROOT.parent / "DiaData", [cohort],
                             cache_dir=GLUCOEDGE_ROOT / "results" / "diadata_cache")
        ds = GlucoseTrendDataset(frame, classes=FIVE_CLASSES, input_length=12)
        w = delta_windows(np.stack(ds.windows), True).astype(np.float32)
        _DIA_CACHE[cohort] = (w, np.asarray(ds.labels, dtype=np.int64),
                              np.asarray([str(p) for p in ds.participants]))
    return _DIA_CACHE[cohort]


def _weinstock_train():
    """Eligible weinstock train windows, delta-anchor encoded."""
    from experiments.accuracy_sweep import _data
    from experiments.calibration_coverage import _eligible
    from training.deltas import delta_windows
    ds, owner, ngf = _data("weinstock", 12)
    idx = _eligible(ds["train"], owner, ngf, "train")
    w = delta_windows(np.stack([ds["train"].windows[i] for i in idx]), True).astype(np.float32)
    y = np.asarray([ds["train"].labels[i] for i in idx], dtype=np.int64)
    return w, y


def _cohort_draw(cohort, budget, seed):
    from training.mixtures import sample_subjects
    w, y, pids = _diadata_windows(cohort)
    counts = {}
    for p in pids:
        counts[p] = counts.get(p, 0) + 1
    chosen, _ = sample_subjects(counts, budget, seed)
    mask = np.isin(pids, np.asarray(chosen))
    return w[mask], y[mask]


def build_arm_tensors(run, a, sample_seed=SAMPLE_SEED):
    """Materialize an arm's training tensors once; cached npz."""
    from training.mixtures import CAP, TRAIN_COHORTS, oversample_extra_indices
    path = _tensor_path(run, a, sample_seed)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    m = a["mixture"]
    parts = []
    if m != "M3":
        parts.append(_weinstock_train())
    if m != "M0":
        if m == "M2":  # equal window budget per cohort (weinstock counts as one)
            budgets = {c: CAP // (len(TRAIN_COHORTS) + 1) for c in TRAIN_COHORTS}
        else:  # M1, M3, M4: natural proportions (by cohort size) within the cap
            sizes = {c: len(_diadata_windows(c)[1]) for c in TRAIN_COHORTS}
            remaining = CAP - (len(parts[0][1]) if parts else 0)
            total = sum(sizes.values())
            budgets = {c: int(remaining * sizes[c] / total) for c in TRAIN_COHORTS}
        for c in TRAIN_COHORTS:
            parts.append(_cohort_draw(c, budgets[c], sample_seed))
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    if m == "M4":
        extra = oversample_extra_indices(y, seed=sample_seed)
        if extra.size:
            x, y = np.concatenate([x, x[extra]]), np.concatenate([y, y[extra]])
    np.savez_compressed(path, x=x, y=y)
    return path


def _load_view(path):
    z = np.load(path)
    return TensorView(z["x"], z["y"])


def _val_view():
    from experiments.accuracy_sweep import _data
    from experiments.calibration_coverage import _eligible
    from training.deltas import delta_windows
    ds, owner, ngf = _data("weinstock", 12)
    idx = _eligible(ds["val"], owner, ngf, "val")
    w = delta_windows(np.stack([ds["val"].windows[i] for i in idx]), True).astype(np.float32)
    y = np.asarray([ds["val"].labels[i] for i in idx], dtype=np.int64)
    return TensorView(w, y)


def _frozen_batch(run):
    smoke = Path(run) / "smoke.json"
    if not smoke.exists():
        raise SystemExit("REFUSED: run the smoke stage first (batch not frozen)")
    return json.loads(smoke.read_text())["frozen_batch"]


def _train_one(run, a, seed, train_v, val_v, batch):
    import torch
    from training.losses import make_criterion
    from training.model import TrendCNN
    from training.train import class_weights, resolve_device, train_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device("mps")
    model = TrendCNN(num_classes=5, channels=tuple(a["channels"]))
    criterion = make_criterion("ce", class_weights(train_v, 5).to(device))
    model = train_model(train_v, val_v, 5, EPOCHS, device,
                        model=model, criterion=criterion, batch_size=batch)
    ckpt = Path(run) / "checkpoints" / f"{a['name']}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)
    return ckpt


def stage_smoke(run_id):
    """Freeze the batch size: M0 at batch 64 vs 512, 2 seeds each; accept
    the largest batch within 1.0pp mean val directional recall of 64.
    Also records same-seed reproducibility on this device."""
    from experiments.analyze_calibration_coverage import confusion_counts
    run = _run_dir(run_id)
    a = arm_r2("SMOKE_M0", "M0", (96, 144))
    path = build_arm_tensors(run, a)
    train_v, val_v = _load_view(path), _val_view()
    results = {}
    for batch in (64, 512):
        per_seed = {}
        for seed in (0, 1):
            ckpt = Path(run) / "checkpoints" / f"smoke_b{batch}_seed{seed}.pt"
            if not ckpt.exists():
                t0 = time.time()
                _train_one_smoke(run, a, seed, train_v, val_v, batch, ckpt)
                print(f"smoke b{batch} seed{seed}: {time.time()-t0:.0f}s")
            per_seed[seed] = _dir_recall_from_ckpt(ckpt, val_v)
        results[batch] = per_seed
    repro_ckpt = Path(run) / "checkpoints" / "smoke_b512_seed0_repeat.pt"
    if not repro_ckpt.exists():
        _train_one_smoke(run, a, 0, train_v, val_v, 512, repro_ckpt)
    repro = abs(_dir_recall_from_ckpt(repro_ckpt, val_v) - results[512][0])
    mean64 = float(np.mean(list(results[64].values())))
    frozen = 64
    for batch in (512, 256, 128):
        if batch in results:
            if float(np.mean(list(results[batch].values()))) >= mean64 - 0.01:
                frozen = batch
                break
        else:  # only trained if 512 failed; train now
            per_seed = {}
            for seed in (0, 1):
                ckpt = Path(run) / "checkpoints" / f"smoke_b{batch}_seed{seed}.pt"
                if not ckpt.exists():
                    _train_one_smoke(run, a, seed, train_v, val_v, batch, ckpt)
                per_seed[seed] = _dir_recall_from_ckpt(ckpt, val_v)
            results[batch] = per_seed
            if float(np.mean(list(per_seed.values()))) >= mean64 - 0.01:
                frozen = batch
                break
    _write_json(Path(run) / "smoke.json", {
        "results": {str(b): {str(s): v for s, v in r.items()} for b, r in results.items()},
        "mean64": mean64, "frozen_batch": frozen,
        "mps_same_seed_abs_diff": repro,
        "mps_seed_reproducible": bool(repro < 1e-6)})
    print(json.dumps({"frozen_batch": frozen, "repro_diff": repro}, indent=1))


def _train_one_smoke(run, a, seed, train_v, val_v, batch, ckpt):
    import torch
    from training.losses import make_criterion
    from training.model import TrendCNN
    from training.train import class_weights, resolve_device, train_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device("mps")
    model = TrendCNN(num_classes=5, channels=tuple(a["channels"]))
    criterion = make_criterion("ce", class_weights(train_v, 5).to(device))
    model = train_model(train_v, val_v, 5, EPOCHS, device,
                        model=model, criterion=criterion, batch_size=batch)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)


def _dir_recall_from_ckpt(ckpt, val_v):
    import torch
    from conversion.common import load_checkpoint
    from experiments.analyze_calibration_coverage import confusion_counts
    model = load_checkpoint(ckpt, 5)
    with torch.no_grad():
        logits = model(torch.from_numpy(val_v.x).unsqueeze(1)).numpy()
    return directional_recall(confusion_counts(val_v.y, logits.argmax(axis=1)))


def _val_stats(run, a, seed):
    """(directional, stable) recall on the eligible weinstock val set."""
    from experiments.analyze_calibration_coverage import confusion_counts
    val_v = _val_view()
    ckpt = Path(run) / "checkpoints" / f"{a['name']}_seed{seed}.pt"
    import torch
    from conversion.common import load_checkpoint
    model = load_checkpoint(ckpt, 5)
    with torch.no_grad():
        logits = model(torch.from_numpy(val_v.x).unsqueeze(1)).numpy()
    cm = confusion_counts(val_v.y, logits.argmax(axis=1))
    return directional_recall(cm), stable_recall(cm)


def champion_val_stats(run=None):
    """Champion INT8 directional/stable recall on the same val set; cached."""
    from experiments.accuracy_sweep import CHAMPION, _tflite_logits
    from experiments.analyze_calibration_coverage import confusion_counts
    cache = (Path(run) / "champion_val.json") if run else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    val_v = _val_view()
    preds = _tflite_logits(CHAMPION["int8"], val_v.x).argmax(axis=1)
    cm = confusion_counts(val_v.y, preds)
    out = {"directional": directional_recall(cm), "stable": stable_recall(cm)}
    if cache:
        _write_json(cache, out)
    return out


def _read_sel(run, stage):
    return json.loads((Path(run) / "selection" / f"{stage}.json").read_text())


def _arms_for(run, stage):
    if stage == "M":
        return stage_m_arms(), ARM_SEEDS
    if stage == "C":
        return stage_c_arms(_read_sel(run, "M")["winner"]), ARM_SEEDS
    if stage == "S":
        winner = _read_sel(run, "C")["winner"]
        return [winner], (3, 4)
    raise SystemExit(f"unknown stage {stage}")


def stage_train(run_id, stage):
    run = _run_dir(run_id)
    batch = _frozen_batch(run)
    arms, seeds = _arms_for(run, stage)
    val_v = _val_view()
    for a in arms:
        train_v = _load_view(build_arm_tensors(run, a))
        for seed in seeds:
            ckpt = Path(run) / "checkpoints" / f"{a['name']}_seed{seed}.pt"
            if ckpt.exists():
                print(f"skip {ckpt.name}: exists")
                continue
            t0 = time.time()
            _train_one(run, a, seed, train_v, val_v, batch)
            print(f"{a['name']} seed {seed} done in {time.time()-t0:.0f}s")


def _select_from_arms(run, stage, arms):
    from experiments.accuracy_sweep import pick_winner
    champ = champion_val_stats(run)
    floor = champ["stable"] - 0.05
    table, means = {}, {}
    for a in arms:
        per_seed = {str(s): _val_stats(run, a, s) for s in ARM_SEEDS}
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
    eligible = [a for a in arms if a["name"] in means]
    if not eligible:
        raise SystemExit("REFUSED: every arm failed the stable-recall "
                         "disqualifier; no winner can be selected")
    winner = pick_winner(eligible, means)
    out = {"stage": stage, "champion_val": champ, "stable_floor": floor,
           "table": table, "winner": winner}
    sel = Path(run) / "selection"
    sel.mkdir(exist_ok=True)
    _write_json(sel / f"{stage}.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
    return out


def stage_select(run_id, stage):
    run = _run_dir(run_id)
    arms, _ = _arms_for(run, stage)
    if stage == "C":
        arms = arms + [_read_sel(run, "M")["winner"]]
    if stage in ("M", "C"):
        return _select_from_arms(run, stage, arms)
    # S: best single seed of the winner across FINAL_SEEDS
    a = arms[0]
    per_seed = {str(s): _val_stats(run, a, s) for s in FINAL_SEEDS}
    best_seed = max(per_seed, key=lambda s: per_seed[s][0])
    out = {"stage": "S",
           "table": {"per_seed": {s: {"directional": v[0], "stable": v[1]}
                                  for s, v in per_seed.items()}},
           "best_seed": int(best_seed),
           "winner": dict(a, ensemble=False)}
    _write_json(Path(run) / "selection" / "S.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
```

`pick_winner` needs `cost_key`, which reads `a["loss"]` and `a["length"]` — `arm_r2` supplies both, so r1's tie-breaking works unchanged.

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r2.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

```bash
git add experiments/accuracy_sweep_r2.py tests/test_accuracy_sweep_r2.py
git commit -m "feat(experiments): r2 arms, mixture tensors, smoke/train/select with stable-floor disqualifier"
```

### Task 5: calib/convert/evaluate/analyze + CLI

**Files:**
- Modify: `experiments/accuracy_sweep_r2.py` (append)
- Test: `tests/test_accuracy_sweep_r2.py` (append)

**Interfaces:**
- Consumes: r1's `stage_calib`-equivalents via imports: `_data`, `_component_windows`, `_tflite_logits`, `CHAMPION`, `paired_rows` from `experiments.accuracy_sweep`; `build_pool`, `select` from `conversion.calibration`; `convert_float`, `convert_int8` from `conversion.convert`; `load_checkpoint`.
- Produces: `stage_calib(run_id)`, `stage_convert(run_id)` (FINAL_SEEDS of the S winner), `stage_evaluate(run_id)` (REFUSES unless `protocol_is_committed(PROTOCOL_R2)`; held_out + temporal + PEDAP), `stage_analyze(run_id)` (gates via `gate_results_r2`, PEDAP descriptive block, `ship.all_gates_pass`), `main()` CLI.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_stage_evaluate_refuses_without_committed_r2_protocol(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r2 as r2
    monkeypatch.setattr(r2, "PROTOCOL_R2", tmp_path / "protocol_sweep_r2.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        r2.stage_evaluate("sweep-r2-test")
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r2.py::test_stage_evaluate_refuses_without_committed_r2_protocol -v`
Expected: FAIL — `stage_evaluate` not defined.

- [ ] **Step 3: Implement** (append):

```python
def stage_calib(run_id):
    from conversion.calibration import build_pool, select
    from experiments.accuracy_sweep import _data
    from experiments.calibration_coverage import _eligible
    run = _run_dir(run_id)
    ds, owner, ngf = _data("weinstock", 12)
    pool = build_pool(ds["val"], _eligible(ds["val"], owner, ngf, "val"))
    out = Path(run) / "selections"
    out.mkdir(exist_ok=True)
    for strategy, draw in (("sequential", 0), ("uniform_random", 101)):
        path = out / f"{strategy}_200_d{draw}.json"
        if path.exists():
            continue
        sel = select(pool, strategy, 200, seed=draw)
        sel["draw"] = draw
        _write_json(path, sel)
        print(f"selected {path.name}")


def stage_convert(run_id):
    import torch
    from conversion.common import load_checkpoint
    from conversion.convert import convert_float, convert_int8
    from experiments.accuracy_sweep import _data
    from training.deltas import delta_windows
    run = _run_dir(run_id)
    winner = _read_sel(run, "S")["winner"]
    ds, _, _ = _data("weinstock", 12)
    art = Path(run) / "artifacts"
    art.mkdir(exist_ok=True)
    sample = torch.zeros(1, 1, 12, dtype=torch.float32)
    for seed in FINAL_SEEDS:
        model = load_checkpoint(Path(run) / "checkpoints" / f"{winner['name']}_seed{seed}.pt", 5)
        fpath = art / f"winner_seed{seed}_float.tflite"
        if not fpath.exists():
            convert_float(model, sample, fpath)
            print(f"converted {fpath.name}")
        for sel_path in sorted((Path(run) / "selections").glob("*_200_d*.json")):
            ipath = art / f"winner_seed{seed}_int8_{sel_path.stem}.tflite"
            if ipath.exists():
                continue
            sel = json.loads(sel_path.read_text())
            ws = np.stack([ds["val"].windows[i] for i in sel["indices"]])
            calib = [torch.from_numpy(w[None, None, :]) for w in delta_windows(ws, True)]
            timings = convert_int8(model, sample, calib, ipath)
            _write_json(ipath.with_suffix(".json"), {"selection": sel_path.stem, **timings})
            print(f"converted {ipath.name}")


def _pedap_windows():
    from training.mixtures import HOLDOUT_COHORT
    w, y, pids = _diadata_windows(HOLDOUT_COHORT)
    return w, {"labels": y.tolist(), "participants": pids.tolist()}


def stage_evaluate(run_id):
    from experiments.accuracy_sweep import (CHAMPION, _component_windows,
                                            _tflite_logits, protocol_is_committed)
    if not protocol_is_committed(PROTOCOL_R2):
        raise SystemExit("REFUSED: commit paper/results/calibration-coverage/"
                         "protocol_sweep_r2.json (clean) before evaluate")
    run = _run_dir(run_id)
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        base = Path(run) / "eval" / comp
        base.mkdir(parents=True, exist_ok=True)
        windows, meta = _component_windows(12, split)
        if not (base / "meta.json").exists():
            _write_json(base / "meta.json", meta)
        for seed in FINAL_SEEDS:
            for art in sorted((Path(run) / "artifacts").glob(f"winner_seed{seed}_*.tflite")):
                p = base / f"{art.stem}_logits.npy"
                if not p.exists():
                    np.save(p, _tflite_logits(art, windows))
                    print(f"{comp}/{art.stem} evaluated")
        for kind in ("float", "int8"):
            p = base / f"champion_{kind}_preds.npy"
            if not p.exists():
                np.save(p, _tflite_logits(CHAMPION[kind], windows).argmax(axis=1))
                print(f"{comp}/champion_{kind} evaluated")
    # PEDAP zero-shot (secondary, descriptive; same freeze discipline)
    base = Path(run) / "eval" / "pedap"
    base.mkdir(parents=True, exist_ok=True)
    windows, meta = _pedap_windows()
    if not (base / "meta.json").exists():
        _write_json(base / "meta.json", meta)
    sel_s = _read_sel(run, "S")
    art = Path(run) / "artifacts" / f"winner_seed{sel_s['best_seed']}_int8_uniform_random_200_d101.tflite"
    for name, path in (("winner_int8", art), ("champion_int8", None)):
        p = base / f"{name}_preds.npy"
        if not p.exists():
            from experiments.accuracy_sweep import CHAMPION as CH
            np.save(p, _tflite_logits(path or CH["int8"], windows).argmax(axis=1))
            print(f"pedap/{name} evaluated")


def stage_analyze(run_id, n_boot=2000):
    from experiments.accuracy_sweep import paired_rows
    from experiments.analyze_calibration_coverage import confusion_counts
    run = _run_dir(run_id)
    sel_s = _read_sel(run, "S")
    winner, best_seed = sel_s["winner"], sel_s["best_seed"]
    ship_arm = "int8_uniform_random_200_d101"
    out = {"run_id": run_id,
           "selection_trail": {s: _read_sel(run, s) for s in ("M", "C", "S")},
           "smoke": json.loads((Path(run) / "smoke.json").read_text()),
           "components": {}}
    for comp in ("held_out", "temporal"):
        base = Path(run) / "eval" / comp
        meta = json.loads((base / "meta.json").read_text())
        y = np.array(meta["labels"])
        pids = np.array(meta["participants"])
        champ = np.load(base / "champion_int8_preds.npy")
        cand_int8 = np.load(base / f"winner_seed{best_seed}_{ship_arm}_logits.npy").argmax(axis=1)
        cand_float = np.load(base / f"winner_seed{best_seed}_float_logits.npy").argmax(axis=1)
        per_seed = {str(s): {
            k: directional_recall(confusion_counts(
                y, np.load(base / f"winner_seed{s}_{k}_logits.npy").argmax(axis=1)))
            for k in ("float", ship_arm, "int8_sequential_200_d0")}
            for s in FINAL_SEEDS}
        from experiments.analyze_calibration_coverage import macro_f1
        out["components"][comp] = {
            "windows": len(y),
            "per_seed_directional": per_seed,
            "macro_f1_candidate_int8": macro_f1(confusion_counts(y, cand_int8)),
            "macro_f1_champion_int8": macro_f1(confusion_counts(y, champ)),
            "gates": gate_results_r2(y, pids, cand_int8, cand_float, champ,
                                     n_boot=n_boot),
        }
    base = Path(run) / "eval" / "pedap"
    meta = json.loads((base / "meta.json").read_text())
    y = np.array(meta["labels"])
    ped = {}
    for name in ("winner_int8", "champion_int8"):
        preds = np.load(base / f"{name}_preds.npy")
        cm = confusion_counts(y, preds)
        from training.labeling import FIVE_CLASSES
        ped[name] = {"directional": directional_recall(cm),
                     "stable": stable_recall(cm),
                     "per_class": {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
                                   for i, c in enumerate(FIVE_CLASSES)}}
    out["pedap_zero_shot"] = {"windows": len(y), **ped,
                              "note": "descriptive; no gates; cohort never trained on"}
    g = out["components"]["held_out"]["gates"]
    out["ship"] = {
        "candidate": f"seed {best_seed}",
        "all_gates_pass": bool(g["gate1_beats_champion"] and g["gate2_stable_floor"]
                               and g["gate3_int8_within_half_pp_of_float"]),
    }
    (Path(run) / "analysis").mkdir(exist_ok=True)
    _write_json(Path(run) / "analysis" / "sweep_r2_aggregate.json", out)
    print("wrote", Path(run) / "analysis" / "sweep_r2_aggregate.json")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "train", "select", "calib",
                                          "convert", "evaluate", "analyze"])
    parser.add_argument("--run-id", default="sweep-r2")
    parser.add_argument("--stage-name", choices=["M", "C", "S"])
    args = parser.parse_args()
    if args.stage == "smoke":
        stage_smoke(args.run_id)
    elif args.stage == "train":
        stage_train(args.run_id, args.stage_name)
    elif args.stage == "select":
        stage_select(args.run_id, args.stage_name)
    elif args.stage == "calib":
        stage_calib(args.run_id)
    elif args.stage == "convert":
        stage_convert(args.run_id)
    elif args.stage == "evaluate":
        stage_evaluate(args.run_id)
    else:
        stage_analyze(args.run_id)


if __name__ == "__main__":
    main()
```

Note: the paired contrast needs no `paired_rows` here — winner and champion share length 12, so the same `meta.json` windows feed both (r1 established the identity pairing; `paired_rows` import stays for the analyze test below if needed, else drop it).

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_accuracy_sweep_r2.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

```bash
git add experiments/accuracy_sweep_r2.py tests/test_accuracy_sweep_r2.py
git commit -m "feat(experiments): r2 calib/convert/evaluate/analyze behind the r2 protocol guard"
```

### Task 6: Smoke run (freezes batch + reproducibility record)

No new files; execution.

- [ ] **Step 1:** `venv/bin/python -m experiments.accuracy_sweep_r2 smoke --run-id sweep-r2` (background; 4–6 trainings of M0 ≈ minutes to ~1h depending on batch).
Expected: `smoke.json` with `frozen_batch` and `mps_same_seed_abs_diff`. Ledger both values.
- [ ] **Step 2:** Ledger: if `mps_seed_reproducible` is false, note that per-seed checkpoints are the record (spec allows it).

### Task 7: Stage M (5 arms × 3 seeds)

- [ ] **Step 1:** `venv/bin/python -m experiments.accuracy_sweep_r2 train --run-id sweep-r2 --stage-name M` (background; first run also materializes the five tensor caches — budget disk ~15GB).
- [ ] **Step 2:** `venv/bin/python -m experiments.accuracy_sweep_r2 select --run-id sweep-r2 --stage-name M`
Expected: `selection/M.json`; winner printed; disqualified arms visible in the table. Ledger winner + table.

### Task 8: Stage C (2 capacities × 3 seeds)

- [ ] **Step 1:** `venv/bin/python -m experiments.accuracy_sweep_r2 train --run-id sweep-r2 --stage-name C` (background).
- [ ] **Step 2:** `venv/bin/python -m experiments.accuracy_sweep_r2 select --run-id sweep-r2 --stage-name C`
Expected: `selection/C.json` (M winner competes as the reused cell). Ledger.

### Task 9: Stage S (seeds 3–4 + best-seed pick)

- [ ] **Step 1:** `venv/bin/python -m experiments.accuracy_sweep_r2 train --run-id sweep-r2 --stage-name S` (background).
- [ ] **Step 2:** `venv/bin/python -m experiments.accuracy_sweep_r2 select --run-id sweep-r2 --stage-name S`
Expected: `selection/S.json` with `best_seed`. Ledger.

### Task 10: Calib + convert + protocol freeze + the one read

- [ ] **Step 1:** `venv/bin/python -m experiments.accuracy_sweep_r2 calib --run-id sweep-r2 && venv/bin/python -m experiments.accuracy_sweep_r2 convert --run-id sweep-r2`
Expected: 2 selections; 15 artifacts.
- [ ] **Step 2:** `venv/bin/python -m experiments.accuracy_sweep_r2 evaluate --run-id sweep-r2`
Expected: `REFUSED: commit ... protocol_sweep_r2.json`.
- [ ] **Step 3:** Write `paper/results/calibration-coverage/protocol_sweep_r2.json` assembled from the run directory (NO held-out/temporal/PEDAP label read yet):

```json
{
  "study": "accuracy-sweep-r2",
  "run_id": "sweep-r2",
  "spec": "docs/superpowers/specs/2026-10-12-accuracy-sweep-r2-design.md",
  "frozen_date_utc": "<today>",
  "glucoedge_commit_at_freeze": "<git rev-parse --short HEAD>",
  "smoke": "<smoke.json contents>",
  "selection_trail": "<selection/{M,C,S}.json, verbatim>",
  "winner": "<selection/S.json winner + best_seed>",
  "champion": {
    "float_sha256": "<shasum -a 256 android/app/src/main/assets/trend_float.tflite>",
    "int8_sha256": "<shasum -a 256 android/app/src/main/assets/trend_int8.tflite>"
  },
  "prediction": "winner INT8 (uniform_random_200_d101) > champion INT8 on held-out directional recall without breaking the stable floor",
  "primary_endpoint": "held_out directional recall delta, participant-cluster bootstrap, 2000 draws, seed 20261012",
  "secondary_endpoints": "temporal same recipe; macro-F1 both splits; PEDAP zero-shot descriptive",
  "gates": {
    "1": "directional delta > 0 and CI lower bound > -0.5pp",
    "2": "winner INT8 held_out stable recall >= champion INT8 stable recall - 2.0pp",
    "3": "winner INT8 within 0.5pp of own float (directional recall)"
  }
}
```

```bash
git add paper/results/calibration-coverage/protocol_sweep_r2.json
git commit -m "docs(protocol): freeze accuracy-sweep-r2 confirmatory read before any held-out label"
```

- [ ] **Step 4:** `venv/bin/python -m experiments.accuracy_sweep_r2 evaluate --run-id sweep-r2 && venv/bin/python -m experiments.accuracy_sweep_r2 analyze --run-id sweep-r2`
Expected: `analysis/sweep_r2_aggregate.json` with three gate booleans + PEDAP block. Ledger every gate outcome.

### Task 11: Publish + paper

- [ ] **Step 1:** Copy `analysis/sweep_r2_aggregate.json` to `paper/results/calibration-coverage/accuracy_sweep_r2.json` with a `protocol` pointer field; scan it for participant-level keys (same walker as r1's publish step) — must be aggregates only. (PEDAP block carries numbers only, never participant lists: verify `meta.json` content did not leak in.)
- [ ] **Step 2:** Paper subsection after the r1 sweep subsection: stage val table (directional + stable), confirmatory contrast with CI, three gate outcomes, PEDAP zero-shot row, macro-F1 cross-reference to r1. One table `tab:sweeptwo`. Numbers only from `accuracy_sweep_r2.json`.
- [ ] **Step 3:** `cd paper && tectonic main.tex && pdftoppm -r 10 main.pdf /tmp/r2pp && ls /tmp/r2pp* | wc -l` — Expected: clean build, ≤ 15 pages (trim as in prior phases if it spills).
- [ ] **Step 4:**

```bash
git add paper/results/calibration-coverage/accuracy_sweep_r2.json paper/main.tex paper/main.pdf
git commit -m "docs(paper): accuracy sweep r2 - mixture x capacity with directional endpoint"
```

### Task 12: Ship decision

- **If `ship.all_gates_pass` false:** one MODELS.md sentence ("sweep r2 did not clear the directional gates; bundled pair retained; see accuracy_sweep_r2.json"), commit, done.
- **If true: STOP and tell the human partner** — the ship PR (separate branch `feat/ship-sweep-r2-models` off main after the research PR merges) regenerates winner float + uniform INT8 into `android/app/src/main/assets/`, reruns `conversion/export_golden_vectors.py`, re-pins sha256s in both apps' parity tests, updates MODELS.md provenance, and runs Android JVM + iOS simulator suites locally before pushing. Window length is 12 and the model is single-seed, so no DeltaEncoding or averaging changes.
