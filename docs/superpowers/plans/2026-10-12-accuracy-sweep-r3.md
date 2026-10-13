# Accuracy Sweep r3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the r3 staged sweep — RT-CGM cohort reopen, time-of-day input channels, dilated-TCN arm — at 5 seeds with the r2 gate discipline, against the newly shipped champion.

**Architecture:** A new `experiments/accuracy_sweep_r3.py` imports r2's machinery (TensorView, smoke, select, calib, protocol guard) and adds hour-aware window builders, mixture `D5R`, feature flag `tod`, and arch flag `tcn`. `training/model.py` gains `in_channels` and `TrendTCN`; `training/deltas.py` gains `tod_channels`. Stage 0 is a standalone RT-CGM interval audit.

**Tech Stack:** Python 3.12 venv, PyTorch (MPS), ai-edge-litert conversion, pytest. All test commands run as `venv/bin/python -m pytest …`.

**Spec:** docs/superpowers/specs/2026-10-12-accuracy-sweep-r3-design.md

## Global Constraints

- Not a medical device; no output string implies clinical validity.
- GlucoBench and DiaData are external sibling dirs: read-only, never committed.
- Participant IDs / timestamps / window keys / per-window predictions never leave gitignored `results/`; published JSONs are aggregates only (leak walker before commit).
- held_out / temporal / PEDAP labels are read exactly once, after `protocol_sweep_r3.json` is committed clean; `stage_evaluate` refuses otherwise.
- Seeds: `ARM_SEEDS_R3 = (0, 1, 2, 3, 4)` at every selection stage; `SAMPLE_SEED = 20261012` (unchanged, comparable subject draws); `BOOT_SEED_R3 = 20261013` single-sourced and test-asserted.
- Cap 15,000,000 windows/arm ("near 15M": overshoot by at most one whole subject); whole-subject sampling; weighted CE; 20 epochs; window length 12; selection on weinstock val eligible windows; stable-floor disqualifier (champion val stable − 5pp) with loud all-disqualified failure.
- Champion = the bundled pair at `android/app/src/main/assets/` (the shipped r2 winner); `experiments.accuracy_sweep.CHAMPION` already points at those files.
- Ship decision STOPS for the user (Task 12). Memory rule: ask before pushing.

## Review Focus

1. A subject with a single reading (no intervals) must be rejected by the audit, not crash the mode computation — test in Task 2.
2. An admitted-list file that is missing or empty must abort tensor building for `D5R` loudly, not silently train DiaData-5 — test in Task 4.
3. ToD channels built from a window that crosses midnight must be continuous (sin/cos, no 23→0 jump artifacts) — test in Task 3.
4. A `[3,12]` model evaluated against `[1,12]` cached tensors (stale cache from r2) must fail loudly on shape, not broadcast — cache key includes the feature flag; shape assert in `_load_view_r3` — test in Task 4.
5. `load_checkpoint` on a legacy plain state-dict (r1/r2 checkpoints) must still return a TrendCNN — backward-compat test in Task 1.

---

### Task 1: Models — `in_channels` + `TrendTCN` + arch-aware checkpoint load

**Files:**
- Modify: `training/model.py`
- Modify: `conversion/common.py` (`load_checkpoint`)
- Test: `tests/test_model_r3.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `TrendCNN(num_classes=5, channels=(24,36), in_channels=1)`; `TrendTCN(num_classes=5, channels=96, in_channels=1)` with blocks dilations (1,2,4,8), kernel 3; `load_checkpoint(path, num_classes)` accepting both legacy plain state-dicts and new `{"state_dict", "arch", "in_channels", "channels"}` dicts, returning the right module in eval mode.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_r3.py
"""r3 model additions - no CGM data needed."""
import torch

from training.model import TrendCNN, TrendTCN


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_trendcnn_in_channels_default_unchanged():
    m = TrendCNN(num_classes=5, channels=(96, 144))
    assert m(torch.zeros(2, 1, 12)).shape == (2, 5)


def test_trendcnn_three_channel_input():
    m = TrendCNN(num_classes=5, channels=(96, 144), in_channels=3)
    assert m(torch.zeros(2, 3, 12)).shape == (2, 5)


def test_tcn_shape_params_and_receptive_field():
    tcn = TrendTCN(num_classes=5)
    assert tcn(torch.zeros(2, 1, 12)).shape == (2, 5)
    cnn = TrendCNN(num_classes=5, channels=(96, 144))
    assert _n(tcn) <= 2 * _n(cnn)          # spec: within 2x of champion arch
    # receptive field 1 + 2*(1+2+4+8) = 31 > 12: a change at t=0 reaches the output
    a, b = torch.zeros(1, 1, 12), torch.zeros(1, 1, 12)
    b[0, 0, 0] = 5.0
    tcn.eval()
    assert not torch.allclose(tcn(a), tcn(b))


def test_tcn_three_channels():
    m = TrendTCN(num_classes=5, in_channels=3)
    assert m(torch.zeros(2, 3, 12)).shape == (2, 5)


def test_load_checkpoint_legacy_and_meta(tmp_path):
    from conversion.common import load_checkpoint
    legacy = TrendCNN(num_classes=5, channels=(96, 144))
    p1 = tmp_path / "legacy.pt"
    torch.save(legacy.state_dict(), p1)
    assert isinstance(load_checkpoint(p1, 5), TrendCNN)
    tcn = TrendTCN(num_classes=5, in_channels=3)
    p2 = tmp_path / "meta.pt"
    torch.save({"state_dict": tcn.state_dict(), "arch": "tcn",
                "in_channels": 3, "channels": 96}, p2)
    loaded = load_checkpoint(p2, 5)
    assert isinstance(loaded, TrendTCN)
    assert loaded(torch.zeros(1, 3, 12)).shape == (1, 5)
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_model_r3.py -q`
Expected: FAIL — `ImportError: cannot import name 'TrendTCN'`.

- [ ] **Step 3: Implement**

In `training/model.py`, add `in_channels: int = 1` to `TrendCNN.__init__` (replace the literal `1` in `nn.Conv1d(1, c1, …)`), and append:

```python
class _TCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.pad = (3 - 1) * dilation  # causal left pad, kernel 3
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return self.relu(y) + x


class TrendTCN(nn.Module):
    """Causal dilated TCN; receptive field 31 steps (spec section 5)."""

    def __init__(self, num_classes: int = 5, channels: int = 96, in_channels: int = 1):
        super().__init__()
        self.inp = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[_TCNBlock(channels, d) for d in (1, 2, 4, 8)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.blocks(self.inp(x))
        return self.fc(self.pool(x).squeeze(-1))
```

In `conversion/common.py` `load_checkpoint`: `obj = torch.load(path, map_location="cpu", weights_only=True)` (checkpoints are our own files, but the meta dict is tensors + str/int only, so the safe loader works — keep it); if `isinstance(obj, dict) and "state_dict" in obj`, build `TrendTCN(num_classes, channels=obj["channels"], in_channels=obj["in_channels"])` when `obj["arch"] == "tcn"` else `TrendCNN(num_classes, channels=tuple(obj["channels"]), in_channels=obj["in_channels"])`, load `obj["state_dict"]`; else keep the existing plain-state-dict TrendCNN path (channel widths inferred as today). `eval()` before return in both branches.

- [ ] **Step 4: Verify pass + no regressions**

Run: `venv/bin/python -m pytest tests/test_model_r3.py tests/test_train.py tests/test_accuracy_sweep_r2.py -q`
Expected: all PASS.

- [ ] **Step 5: TCN conversion smoke (same test file)**

```python
def test_tcn_converts_to_litert(tmp_path):
    import numpy as np
    from conversion.convert import export_float  # exact name: check conversion/convert.py; it is the function stage_convert uses for float export
    tcn = TrendTCN(num_classes=5)
    p = tmp_path / "t.pt"
    torch.save({"state_dict": tcn.state_dict(), "arch": "tcn",
                "in_channels": 1, "channels": 96}, p)
    out = export_float(p, tmp_path / "t.tflite", input_length=12)
    assert out.exists() and out.stat().st_size > 1000
```

If `conversion/convert.py` exposes a different float-export entry point (read it first), adapt the call and ledger the actual name — the assertion stands.

Run: `venv/bin/python -m pytest tests/test_model_r3.py -q` — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add training/model.py conversion/common.py tests/test_model_r3.py
git commit -m "feat(training): in_channels plumbing, TrendTCN, arch-aware checkpoint load"
```

### Task 2: RT-CGM reopen audit

**Files:**
- Create: `experiments/audit_rtcgm.py`
- Test: `tests/test_audit_rtcgm.py`

**Interfaces:**
- Consumes: `training.diadata.load_diadata(root, ["RT-CGM"], cache_dir=…)` → DataFrame `[id, id_segment, time, gl]`.
- Produces: `subject_verdicts(frame) -> dict[str, dict]` (`{"admitted": bool, "modal_s": float|None, "within_tol": float}` per subject id); `main()` writing `paper/results/diadata_audit_r2.json` (aggregates) and `results/rtcgm_admitted.json` (gitignored: `{"admitted": [ids…], "rule": "...", "sample_seed_independent": true}`).

- [ ] **Step 1: Failing tests**

```python
# tests/test_audit_rtcgm.py
import numpy as np
import pandas as pd

from experiments.audit_rtcgm import subject_verdicts


def _frame(rows):
    return pd.DataFrame(rows, columns=["id", "id_segment", "time", "gl"])


def _times(start, step_s, n):
    return pd.date_range(start, periods=n, freq=f"{step_s}s")


def test_clean_5min_admitted_1min_rejected():
    rows = [("a", 0, t, 100.0) for t in _times("2020-01-01", 300, 50)]
    rows += [("b", 0, t, 100.0) for t in _times("2020-01-01", 60, 50)]
    v = subject_verdicts(_frame(rows))
    assert v["a"]["admitted"] is True and v["a"]["modal_s"] == 300.0
    assert v["b"]["admitted"] is False


def test_jitter_beyond_tolerance_rejected():
    # modal 300s but 10% of intervals at 600s -> within_tol = 0.9 < 0.95
    t, rows = pd.Timestamp("2020-01-01"), []
    for i in range(50):
        t += pd.Timedelta(seconds=600 if i % 10 == 0 else 300)
        rows.append(("j", 0, t, 100.0))
    v = subject_verdicts(_frame(rows))
    assert v["j"]["admitted"] is False and v["j"]["modal_s"] == 300.0


def test_single_reading_subject_rejected_not_crash():
    v = subject_verdicts(_frame([("solo", 0, pd.Timestamp("2020-01-01"), 100.0)]))
    assert v["solo"]["admitted"] is False and v["solo"]["modal_s"] is None


def test_intervals_never_cross_segments():
    # two segments far apart: the 1-day gap must not count as an interval
    rows = [("s", 0, t, 100.0) for t in _times("2020-01-01", 300, 20)]
    rows += [("s", 1, t, 100.0) for t in _times("2020-01-02", 300, 20)]
    v = subject_verdicts(_frame(rows))
    assert v["s"]["admitted"] is True and v["s"]["within_tol"] == 1.0
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_audit_rtcgm.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# experiments/audit_rtcgm.py
"""RT-CGM reopen audit (spec r3 section 2): admit per-subject on cadence.

Rule: modal consecutive-interval within segments == 300s AND >= 95% of
intervals in [270, 330]s. Sensor brand is unrecoverable and irrelevant.
"""
import json
from collections import Counter

import numpy as np

from experiments.calibration_coverage import GLUCOEDGE_ROOT, _write_json

TOL_LOW, TOL_HIGH, MODAL = 270.0, 330.0, 300.0
ADMITTED_PATH = GLUCOEDGE_ROOT / "results" / "rtcgm_admitted.json"
AUDIT_PATH = GLUCOEDGE_ROOT / "paper" / "results" / "diadata_audit_r2.json"
RULE = ("admit iff modal within-segment interval == 300s and >=95% of "
        "intervals in [270s, 330s]")


def subject_verdicts(frame):
    out = {}
    for sid, g in frame.groupby("id"):
        ivals = []
        for _, seg in g.groupby("id_segment"):
            d = seg["time"].sort_values().diff().dt.total_seconds().dropna()
            ivals.extend(float(x) for x in d)
        if not ivals:
            out[str(sid)] = {"admitted": False, "modal_s": None, "within_tol": 0.0}
            continue
        modal = float(Counter(ivals).most_common(1)[0][0])
        within = float(np.mean([(TOL_LOW <= x <= TOL_HIGH) for x in ivals]))
        out[str(sid)] = {"admitted": bool(modal == MODAL and within >= 0.95),
                         "modal_s": modal, "within_tol": within}
    return out


def main():
    from training.diadata import load_diadata
    frame = load_diadata(GLUCOEDGE_ROOT.parent / "DiaData", ["RT-CGM"],
                         cache_dir=GLUCOEDGE_ROOT / "results" / "diadata_cache")
    v = subject_verdicts(frame)
    admitted = sorted(s for s, r in v.items() if r["admitted"])
    _write_json(ADMITTED_PATH, {"admitted": admitted, "rule": RULE})
    hist = Counter()
    for r in v.values():
        if r["modal_s"] is not None:
            hist[int(r["modal_s"])] += 1
    _write_json(AUDIT_PATH, {
        "cohort": "RT-CGM", "rule": RULE,
        "subjects_total": len(v), "subjects_admitted": len(admitted),
        "modal_interval_histogram_s": dict(sorted(hist.items())),
    })
    print(json.dumps({"total": len(v), "admitted": len(admitted)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** Run `venv/bin/python -m pytest tests/test_audit_rtcgm.py -q` — Expected: 4 PASS. (The published JSON carries counts and a binned histogram only; `admitted` ids go only to gitignored `results/`.)

- [ ] **Step 5: Commit**

```bash
git add experiments/audit_rtcgm.py tests/test_audit_rtcgm.py
git commit -m "feat(experiments): RT-CGM cadence reopen audit with gitignored admit list"
```

### Task 3: Time-of-day channels

**Files:**
- Modify: `training/deltas.py`
- Test: `tests/test_deltas_tod.py` (new)

**Interfaces:**
- Produces: `tod_channels(anchor_hours: np.ndarray[n], length: int = 12, step_minutes: float = 5.0) -> np.ndarray[n, 2, length]` — per-step fractional hour derived backwards from the anchor at nominal 5-minute spacing (step i hour = anchor − (length−1−i)·5/60, mod 24); row 0 = sin(2πh/24), row 1 = cos(2πh/24).

- [ ] **Step 1: Failing tests**

```python
# tests/test_deltas_tod.py
import numpy as np
import pytest

from training.deltas import tod_channels


def test_values_hand_computed_at_noon():
    out = tod_channels(np.array([12.0]), length=12)
    assert out.shape == (1, 2, 12)
    # last step is the anchor itself: h=12 -> sin=0 (approx), cos=-1
    assert out[0, 0, -1] == pytest.approx(np.sin(2 * np.pi * 12 / 24), abs=1e-9)
    assert out[0, 1, -1] == pytest.approx(-1.0, abs=1e-9)
    # first step is 55 minutes earlier
    h0 = 12.0 - 11 * 5 / 60
    assert out[0, 0, 0] == pytest.approx(np.sin(2 * np.pi * h0 / 24))


def test_midnight_wrap_is_continuous():
    # anchor 00:10 -> early steps fall before midnight; sin/cos stay bounded
    out = tod_channels(np.array([10 / 60]), length=12)
    assert np.all(np.isfinite(out)) and np.all(np.abs(out) <= 1.0)
    h0 = (10 / 60 - 55 / 60) % 24       # 23.25
    assert out[0, 1, 0] == pytest.approx(np.cos(2 * np.pi * h0 / 24))


def test_float32_output():
    assert tod_channels(np.array([3.5])).dtype == np.float32
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_deltas_tod.py -q` — Expected: FAIL (name missing).

- [ ] **Step 3: Implement** (append to `training/deltas.py`)

```python
def tod_channels(anchor_hours, length=12, step_minutes=5.0):
    """sin/cos time-of-day per window step, derived backwards from the
    anchor at nominal spacing (spec r3 section 4; device-local clock)."""
    anchor_hours = np.asarray(anchor_hours, dtype=np.float64)
    offsets = (np.arange(length) - (length - 1)) * step_minutes / 60.0
    h = (anchor_hours[:, None] + offsets[None, :]) % 24.0
    ang = 2.0 * np.pi * h / 24.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)
```

- [ ] **Step 4:** Run `venv/bin/python -m pytest tests/test_deltas_tod.py -q` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add training/deltas.py tests/test_deltas_tod.py
git commit -m "feat(training): sin/cos time-of-day channels for r3"
```

### Task 4: r3 arms, mixtures, hour-aware tensors

**Files:**
- Create: `experiments/accuracy_sweep_r3.py` (part 1)
- Test: `tests/test_accuracy_sweep_r3.py` (new)

**Interfaces:**
- Consumes: from `experiments.accuracy_sweep_r2`: `TensorView`, `SAMPLE_SEED`, `_weinstock_train` (shape reference only), `stable_recall`, `directional_recall`; from `training.mixtures`: `CAP`, `TRAIN_COHORTS`, `sample_subjects`; from Task 2: `ADMITTED_PATH`; from Task 3: `tod_channels`.
- Produces: constants `PROTOCOL_R3` (`paper/results/calibration-coverage/protocol_sweep_r3.json`), `ARM_SEEDS_R3 = (0, 1, 2, 3, 4)`, `BOOT_SEED_R3 = 20261013`, `EPOCHS = 20`; `arm_r3(name, mixture, features, arch) -> dict` carrying also `channels=[96,144]`, `length=12`, `loss="ce"`, `pooled=False` (r1 `cost_key` compatibility) and `in_channels` (1 or 3); `stage_d_arms()`, `stage_f_arms(winner)`, `stage_a_arms(winner)`; `_tensor_path_r3(run, a, sample_seed)` → `tensors/{mixture}_{features}_s{seed}.npz`; `_diadata_windows_r3(cohort)` → `(w[n,12] float32, y int64, pids, anchor_hours float64)` cached npz under `results/diadata_cache/hours/`; `build_arm_tensors_r3(run, a, …)` writing npz `{x: [n, C, 12], y}`; `_load_view_r3(path, in_channels)` (shape-asserting); `_val_view_r3(features)`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_accuracy_sweep_r3.py
"""r3 pure-logic tests - no CGM data needed."""
import json

import numpy as np
import pytest

import experiments.accuracy_sweep_r3 as r3


def test_arm_r3_carries_cost_key_fields_and_in_channels():
    a = r3.arm_r3("D_base", "D5", "none", "cnn")
    assert a["channels"] == [96, 144] and a["length"] == 12
    assert a["loss"] == "ce" and a["pooled"] is False
    assert a["in_channels"] == 1
    assert r3.arm_r3("F_tod", "D5", "tod", "cnn")["in_channels"] == 3


def test_stage_arms_shapes():
    d = r3.stage_d_arms()
    assert [a["mixture"] for a in d] == ["D5", "D5R"]
    w = r3.arm_r3("D_rtcgm", "D5R", "none", "cnn")
    f = r3.stage_f_arms(w)
    assert [a["features"] for a in f] == ["none", "tod"]
    assert all(a["mixture"] == "D5R" for a in f)
    a_ = r3.stage_a_arms(f[1])
    assert [x["arch"] for x in a_] == ["cnn", "tcn"]
    assert all(x["features"] == "tod" for x in a_)


def test_tensor_path_keyed_by_mixture_features_seed(tmp_path):
    a1 = r3.arm_r3("D_base", "D5", "none", "cnn")
    a2 = r3.arm_r3("F_tod", "D5", "tod", "cnn")
    p1 = r3._tensor_path_r3(tmp_path, a1, 0)
    p2 = r3._tensor_path_r3(tmp_path, a2, 0)
    p3 = r3._tensor_path_r3(tmp_path, a1, 1)
    assert len({p1, p2, p3}) == 3 and "D5_none" in p1.name


def test_d5r_without_admit_list_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(r3, "ADMITTED_PATH", tmp_path / "missing.json")
    a = r3.arm_r3("D_rtcgm", "D5R", "none", "cnn")
    with pytest.raises(SystemExit, match="rtcgm"):
        r3.build_arm_tensors_r3(tmp_path, a)


def test_load_view_r3_asserts_channel_shape(tmp_path):
    p = tmp_path / "t.npz"
    np.savez(p, x=np.zeros((4, 1, 12), np.float32), y=np.zeros(4, np.int64))
    with pytest.raises(SystemExit, match="channels"):
        r3._load_view_r3(p, in_channels=3)
    v = r3._load_view_r3(p, in_channels=1)
    assert len(v) == 4
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_accuracy_sweep_r3.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement part 1 of `experiments/accuracy_sweep_r3.py`**

```python
"""Accuracy sweep r3: RT-CGM reopen + time-of-day + TCN (spec 2026-10-12-r3).

Reuses r2 machinery; selection stays on weinstock val; champion is the
bundled (shipped r2) pair read live from android assets.
"""
import json
from pathlib import Path

import numpy as np

from experiments.accuracy_sweep_r2 import (EPOCHS, SAMPLE_SEED, TensorView,
                                           directional_recall, stable_recall)
from experiments.audit_rtcgm import ADMITTED_PATH
from experiments.calibration_coverage import GLUCOEDGE_ROOT, _run_dir, _write_json

PROTOCOL_R3 = GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage" / "protocol_sweep_r3.json"
ARM_SEEDS_R3 = (0, 1, 2, 3, 4)
BOOT_SEED_R3 = 20261013  # pre-registered (protocol_sweep_r3.json); single source
MIXTURES_R3 = ("D5", "D5R")


def arm_r3(name, mixture, features, arch):
    return {"name": name, "mixture": mixture, "features": features,
            "arch": arch, "in_channels": 3 if features == "tod" else 1,
            "channels": [96, 144], "length": 12, "loss": "ce", "pooled": False}


def stage_d_arms():
    return [arm_r3("D_base", "D5", "none", "cnn"),
            arm_r3("D_rtcgm", "D5R", "none", "cnn")]


def stage_f_arms(winner):
    return [arm_r3("F_base", winner["mixture"], "none", "cnn"),
            arm_r3("F_tod", winner["mixture"], "tod", "cnn")]


def stage_a_arms(winner):
    return [arm_r3("A_cnn", winner["mixture"], winner["features"], "cnn"),
            arm_r3("A_tcn", winner["mixture"], winner["features"], "tcn")]


def _tensor_path_r3(run, a, sample_seed):
    return Path(run) / "tensors" / f"{a['mixture']}_{a['features']}_s{sample_seed}.npz"


_HOUR_CACHE = {}


def _diadata_windows_r3(cohort):
    """Like r2's _diadata_windows but also returns anchor hours; cached npz."""
    if cohort in _HOUR_CACHE:
        return _HOUR_CACHE[cohort]
    cache = GLUCOEDGE_ROOT / "results" / "diadata_cache" / "hours" / f"{cohort}.npz"
    if cache.exists():
        # own cache file; plain numeric + unicode arrays, no pickle needed
        z = np.load(cache)
        out = (z["w"], z["y"], z["pids"], z["hours"])
    else:
        from training.dataset import GlucoseTrendDataset
        from training.deltas import delta_windows
        from training.diadata import load_diadata
        from training.labeling import FIVE_CLASSES
        frame = load_diadata(GLUCOEDGE_ROOT.parent / "DiaData", [cohort],
                             cache_dir=GLUCOEDGE_ROOT / "results" / "diadata_cache")
        ds = GlucoseTrendDataset(frame, classes=FIVE_CLASSES, input_length=12)
        w = delta_windows(np.stack(ds.windows), True).astype(np.float32)
        hours = np.array([t.hour + t.minute / 60 + t.second / 3600
                          for t in ds.anchor_times], dtype=np.float64)
        out = (w, np.asarray(ds.labels, dtype=np.int64),
               np.asarray([str(p) for p in ds.participants]), hours)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, w=out[0], y=out[1], pids=out[2], hours=out[3])
    _HOUR_CACHE[cohort] = out
    return out


def _r3_cohorts(mixture):
    from training.mixtures import TRAIN_COHORTS
    cohorts = list(TRAIN_COHORTS)
    if mixture == "D5R":
        cohorts.append("RT-CGM")
    return cohorts


def _admitted_rtcgm():
    if not ADMITTED_PATH.exists():
        raise SystemExit("REFUSED: results/rtcgm_admitted.json missing - run "
                         "the rtcgm audit (Task 7) before building D5R tensors")
    ids = json.loads(ADMITTED_PATH.read_text())["admitted"]
    if not ids:
        raise SystemExit("REFUSED: rtcgm admit list is empty")
    return set(ids)


def _with_features(w, hours, features):
    if features == "none":
        return w[:, None, :]
    from training.deltas import tod_channels
    return np.concatenate([w[:, None, :], tod_channels(hours, w.shape[1])], axis=1)


def build_arm_tensors_r3(run, a, sample_seed=SAMPLE_SEED):
    from training.mixtures import CAP, sample_subjects
    path = _tensor_path_r3(run, a, sample_seed)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cohorts = _r3_cohorts(a["mixture"])
    admitted = _admitted_rtcgm() if a["mixture"] == "D5R" else None
    data = {}
    for c in cohorts:
        w, y, pids, hours = _diadata_windows_r3(c)
        if c == "RT-CGM":
            keep = np.isin(pids, np.asarray(sorted(admitted)))
            w, y, pids, hours = w[keep], y[keep], pids[keep], hours[keep]
        data[c] = (w, y, pids, hours)
    sizes = {c: len(d[1]) for c, d in data.items()}
    total = sum(sizes.values())
    budgets = {c: int(CAP * sizes[c] / total) for c in cohorts}
    xs, ys = [], []
    for c in cohorts:
        w, y, pids, hours = data[c]
        counts = {}
        for p in pids:
            counts[p] = counts.get(p, 0) + 1
        chosen, _ = sample_subjects(counts, budgets[c], sample_seed)
        mask = np.isin(pids, np.asarray(chosen))
        xs.append(_with_features(w[mask], hours[mask], a["features"]))
        ys.append(y[mask])
    np.savez_compressed(path, x=np.concatenate(xs), y=np.concatenate(ys))
    return path


def _load_view_r3(path, in_channels):
    z = np.load(path)
    if z["x"].ndim != 3 or z["x"].shape[1] != in_channels:
        raise SystemExit(f"REFUSED: tensor channels {z['x'].shape} do not match "
                         f"arm in_channels={in_channels} - stale cache?")
    return TensorView(z["x"], z["y"])


def _val_view_r3(features):
    from experiments.accuracy_sweep import _data
    from experiments.calibration_coverage import _eligible
    from training.deltas import delta_windows
    ds, owner, ngf = _data("weinstock", 12)
    idx = _eligible(ds["val"], owner, ngf, "val")
    t = ds["val"]
    w = delta_windows(np.stack([t.windows[i] for i in idx]), True).astype(np.float32)
    y = np.asarray([t.labels[i] for i in idx], dtype=np.int64)
    hours = np.array([t.anchor_times[i].hour + t.anchor_times[i].minute / 60
                      + t.anchor_times[i].second / 3600 for i in idx])
    return TensorView(_with_features(w, hours, features), y)
```

Note: `TensorView.__getitem__` in r2 does `torch.from_numpy(self.x[k]).unsqueeze(0)` — for `[C,12]` rows that would add a wrong axis. r3 stores `x` as `[n, C, 12]` and needs rows WITHOUT unsqueeze. Add to this file:

```python
class TensorView3(TensorView):
    """Rows are already [C, L]; no unsqueeze."""

    def __getitem__(self, k):
        import torch
        return torch.from_numpy(self.x[k]), int(self.y[k])
```

and use `TensorView3` in `_load_view_r3` / `_val_view_r3` (r2's `class_weights` and `train_model` only touch `.labels` / iteration, so they are unaffected). Update the Step 1 test `test_load_view_r3_asserts_channel_shape` expectation accordingly (`r3.TensorView3`).

- [ ] **Step 4:** Run `venv/bin/python -m pytest tests/test_accuracy_sweep_r3.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r3.py tests/test_accuracy_sweep_r3.py
git commit -m "feat(experiments): r3 arms, RT-CGM mixture, hour-aware tensors with strict cache keys"
```

### Task 5: r3 train / select with reuse cells

**Files:**
- Modify: `experiments/accuracy_sweep_r3.py` (part 2)
- Test: `tests/test_accuracy_sweep_r3.py` (extend)

**Interfaces:**
- Consumes: r2's `_frozen_batch(run)`, `stage_smoke` (reused verbatim via a thin `stage_smoke_r3(run_id)` that calls it so smoke.json lands in the r3 run dir — read r2's `stage_smoke` first; if it hardcodes M0 arms that is fine, the batch freeze transfers), `champion_val_stats(run)`, `_select_from_arms`-equivalent logic, `pick_winner` from `experiments.accuracy_sweep`.
- Produces: `_train_one_r3(run, a, seed, train_v, val_v, batch)` saving `{"state_dict", "arch", "in_channels", "channels"}` + provenance sidecar; `stage_train(run_id, stage)` for stages `D|F|A`; `stage_select(run_id, stage)` writing `selection/{stage}.json` (stage `S` = best seed of 5 by val directional with stable floor); reuse rule: a cell whose `(mixture, features, arch)` equals an already-trained cell reuses its checkpoints (`_ckpt(run, a, seed)` keyed by `f"{a['mixture']}_{a['features']}_{a['arch']}_seed{seed}"`).

- [ ] **Step 1: Failing tests (extend `tests/test_accuracy_sweep_r3.py`)**

```python
def test_ckpt_key_covers_mixture_features_arch(tmp_path):
    a1 = r3.arm_r3("D_base", "D5", "none", "cnn")
    a2 = r3.arm_r3("A_tcn", "D5", "none", "tcn")
    assert r3._ckpt(tmp_path, a1, 0) != r3._ckpt(tmp_path, a2, 0)
    # same cell under two stage names -> same path (reuse, not retrain)
    a3 = r3.arm_r3("F_base", "D5", "none", "cnn")
    assert r3._ckpt(tmp_path, a1, 0) == r3._ckpt(tmp_path, a3, 0)


def test_select_disqualifier_and_all_dq_failure(tmp_path, monkeypatch):
    arms = r3.stage_d_arms()
    stats = {"D_base": (0.52, 0.40), "D_rtcgm": (0.54, 0.55)}
    monkeypatch.setattr(r3, "_val_stats_r3", lambda run, a, seed: stats[a["name"]])
    monkeypatch.setattr(r3, "_champion_val", lambda run: {"directional": 0.50, "stable": 0.56})
    out = r3._select(tmp_path, "D", arms)
    assert out["winner"]["name"] == "D_rtcgm"
    assert out["table"]["D_base"]["disqualified"] is True
    bad = {"D_base": (0.52, 0.40), "D_rtcgm": (0.54, 0.40)}
    monkeypatch.setattr(r3, "_val_stats_r3", lambda run, a, seed: bad[a["name"]])
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        r3._select(tmp_path, "D", arms)


def test_boot_seed_single_source():
    y = np.array([0, 1, 2, 3, 4] * 40)
    pids = np.array([f"p{i % 4}" for i in range(200)])
    g = r3.gate_results_r3(y, pids, cand_int8=y, cand_float=y, champ_int8=y, n_boot=50)
    assert g["contrast_int8_vs_champion"]["seed"] == 20261013
```

- [ ] **Step 2:** Run — Expected: FAIL (`_ckpt`, `_select`, `gate_results_r3` missing).

- [ ] **Step 3: Implement part 2** (append; mirror r2's `stage_train`/`_select_from_arms`/`stage_select` shapes — read them side-by-side):

```python
def _ckpt(run, a, seed):
    return (Path(run) / "checkpoints"
            / f"{a['mixture']}_{a['features']}_{a['arch']}_seed{seed}.pt")


def _train_one_r3(run, a, seed, train_v, val_v, batch):
    import time as _t
    import torch
    from training.losses import make_criterion
    from training.model import TrendCNN, TrendTCN
    from training.train import class_weights, resolve_device, train_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device("mps")
    if a["arch"] == "tcn":
        model = TrendTCN(num_classes=5, in_channels=a["in_channels"])
        channels = 96
    else:
        model = TrendCNN(num_classes=5, channels=tuple(a["channels"]),
                         in_channels=a["in_channels"])
        channels = a["channels"]
    criterion = make_criterion("ce", class_weights(train_v, 5).to(device))
    t0 = _t.time()
    model = train_model(train_v, val_v, 5, EPOCHS, device,
                        model=model, criterion=criterion, batch_size=batch)
    ckpt = _ckpt(run, a, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "arch": a["arch"], "in_channels": a["in_channels"],
                "channels": channels}, ckpt)
    ckpt.with_suffix(".json").write_text(json.dumps({
        "arm": a["name"], "seed": seed, "device": device.type, "batch": batch,
        "train_windows": len(train_v), "val_windows": len(val_v),
        "epochs": EPOCHS, "duration_s": round(_t.time() - t0, 1)}, indent=1))
    return ckpt
```

`stage_train(run_id, stage)`: resolve arms (`stage_d_arms()` / `stage_f_arms(_read_sel(run,"D")["winner"])` / `stage_a_arms(_read_sel(run,"F")["winner"])` — port `_read_sel` from r2), for each arm × `ARM_SEEDS_R3`: skip if `_ckpt` exists (this is also the reuse rule — F_base/A_cnn resolve to already-trained paths), else `build_arm_tensors_r3` → `_load_view_r3(path, a["in_channels"])` + `_val_view_r3(a["features"])` → `_train_one_r3`.

`_val_stats_r3(run, a, seed)`: load `_ckpt` via `conversion.common.load_checkpoint`, run on `_val_view_r3(a["features"])` in batches, return `(directional_recall(cm), stable_recall(cm))` — port r2's `_val_stats` body, swapping the view and loader.

`_champion_val(run)`: delegate to r2's `champion_val_stats` (same INT8 champion on the same val windows; it reads the bundled asset which is now the r2 winner — correct by construction). Cache under the r3 run dir.

`_select(run, stage, arms)`: port r2 `_select_from_arms` verbatim (floor = champion stable − 0.05, DQ'd arms excluded, `SystemExit` if all DQ, `pick_winner` tie rule), reading stats via `_val_stats_r3` over `ARM_SEEDS_R3`.

`stage_select(run_id, stage)`: stages D/F/A over their arms; stage `S` = per-seed table over `ARM_SEEDS_R3` for the A winner, `best_seed` by directional recall (stable floor binding per seed).

`gate_results_r3(y, pids, cand_int8, cand_float, champ_int8, n_boot=2000)`: copy r2's `gate_results_r2` body exactly but pass `seed=BOOT_SEED_R3` to `paired_cluster_bootstrap`.

- [ ] **Step 4:** Run `venv/bin/python -m pytest tests/test_accuracy_sweep_r3.py -q` — Expected: PASS (all).

- [ ] **Step 5:** Full suite: `venv/bin/python -m pytest -q` — Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/accuracy_sweep_r3.py tests/test_accuracy_sweep_r3.py
git commit -m "feat(experiments): r3 train/select with cell reuse and pre-registered bootstrap seed"
```

### Task 6: r3 calib / convert / evaluate / analyze + CLI

**Files:**
- Modify: `experiments/accuracy_sweep_r3.py` (part 3)
- Test: `tests/test_accuracy_sweep_r3.py` (extend)

**Interfaces:**
- Consumes: r2's `stage_calib` pattern, `conversion` entry points used by r2's `stage_convert` (read it; reuse the same calls with `load_checkpoint` now arch-aware), `experiments.accuracy_sweep`: `protocol_is_committed(path)`, `_component_windows`, `_tflite_logits`, `CHAMPION`, `_write_json`; r2's `_pedap_windows`.
- Produces: `stage_calib(run_id)` (same two selections), `stage_convert(run_id)` (5 seeds float + 2 INT8 each, for the S winner's cell), `stage_evaluate(run_id)` refusing unless `protocol_is_committed(PROTOCOL_R3)`, evaluating held_out/temporal for winner artifacts + champion, and PEDAP zero-shot; winner windows built hour-aware when `features == "tod"` (augment `_component_windows`' output with hours from its `keys` anchor-times — parse `keys[i][1]` timestamps); `stage_analyze(run_id)` writing `analysis/sweep_r3_aggregate.json` with per-seed tables, `gate_results_r3`, `pairing_rule` + `dropped_windows` fields (identical windows by construction → both 0, reported), macro-F1 cross-ref, PEDAP block, `per_arm_train_windows` (from tensor files), `ship.all_gates_pass`; `main()` CLI `stage ∈ {smoke, train, select, calib, convert, evaluate, analyze}` with `--run-id sweep-r3 --stage-name D|F|A|S`.

- [ ] **Step 1: Failing tests (extend)**

```python
def test_stage_evaluate_refuses_without_committed_r3_protocol(monkeypatch, tmp_path):
    monkeypatch.setattr(r3, "PROTOCOL_R3", tmp_path / "protocol_sweep_r3.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        r3.stage_evaluate("sweep-r3-test")


def test_analyze_reports_pairing_rule_fields():
    # pure shape check on the dict builder, not a data run
    block = r3._pairing_block(n_winner=100, n_champion=100)
    assert block == {"pairing_rule": "identical eligible windows by construction",
                     "dropped_windows_winner": 0, "dropped_windows_champion": 0}
```

- [ ] **Step 2:** Run — Expected: FAIL.

- [ ] **Step 3: Implement part 3.** Port r2's `stage_calib` / `stage_convert` / `stage_evaluate` / `stage_analyze` / `main` bodies with these changes: protocol path `PROTOCOL_R3`; winner read from `selection/S.json`; checkpoints via `_ckpt`; conversion loads via arch-aware `load_checkpoint` (Task 1); gates via `gate_results_r3`; `_pairing_block(n_winner, n_champion)` helper returning the dict in the test, embedded in each split's gates block; `per_arm_train_windows` computed from `tensors/*.npz` lengths at analyze time; ToD winner path: wrap `_component_windows` → hours from `meta["keys"]` anchor-time strings via `pd.Timestamp(k[1])`, then `_with_features`. Champion logits always run on the plain `[1,12]` windows (champion input contract unchanged).

- [ ] **Step 4:** `venv/bin/python -m pytest tests/test_accuracy_sweep_r3.py -q` — Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r3.py tests/test_accuracy_sweep_r3.py
git commit -m "feat(experiments): r3 calib/convert/evaluate/analyze behind the r3 protocol guard"
```

### Task 7: Run Stage 0 — RT-CGM audit (real data)

- [ ] **Step 1:** `venv/bin/python -m experiments.audit_rtcgm`
Expected: JSON line with `total` 451 and a plausible `admitted` count (spec expectation 200–350; any count is a finding, not a failure — ledger it). `paper/results/diadata_audit_r2.json` and gitignored `results/rtcgm_admitted.json` exist.
- [ ] **Step 2:** Leak check: `venv/bin/python - <<'EOF'` walk `diadata_audit_r2.json` for id-like keys/lists (reuse Task 11's walker inline) `EOF` — Expected: no flags.
- [ ] **Step 3: Commit**

```bash
git add paper/results/diadata_audit_r2.json
git commit -m "docs(results): RT-CGM reopen audit - cadence-admitted subject counts"
```

### Task 8: Smoke + Stage D (run)

- [ ] **Step 1:** `venv/bin/python -m experiments.accuracy_sweep_r3 smoke --run-id sweep-r3`
Expected: `smoke.json` in the r3 run dir with `frozen_batch` (512 expected) and `mps_seed_reproducible: true`.
- [ ] **Step 2 (ruling checkpoint):** If `frozen_batch == 512`, D_base MAY reuse r2's five `M_M3_seed{0..4}.pt` checkpoints by copying them to the r3 `_ckpt` names (`D5_none_cnn_seed{N}.pt`) — identical recipe, identical batch; ledger the ruling. Legacy plain-state-dict copies load via the Task 1 backward-compat branch. If the frozen batch differs, retrain D_base.
- [ ] **Step 3:** `venv/bin/python -m experiments.accuracy_sweep_r3 train --run-id sweep-r3 --stage-name D` (background; first run materializes D5R tensors ~15GB).
- [ ] **Step 4:** `venv/bin/python -m experiments.accuracy_sweep_r3 select --run-id sweep-r3 --stage-name D`
Expected: `selection/D.json`; winner printed with full table; ledger winner + any disqualification.

### Task 9: Stages F, A, S (run)

- [ ] **Step 1:** `… train --stage-name F` (only the `tod` cell trains; `F_base` reuses) then `… select --stage-name F`. Ledger.
- [ ] **Step 2:** `… train --stage-name A` (only the `tcn` cell trains) then `… select --stage-name A`. Ledger.
- [ ] **Step 3:** `… select --stage-name S` — Expected: `selection/S.json` with per-seed table + `best_seed`. Ledger.

### Task 10: Calib + convert + protocol freeze + the one read

- [ ] **Step 1:** `venv/bin/python -m experiments.accuracy_sweep_r3 calib --run-id sweep-r3 && venv/bin/python -m experiments.accuracy_sweep_r3 convert --run-id sweep-r3`
Expected: 2 calibration selections; 15 artifacts (5 float + 10 INT8).
- [ ] **Step 2:** `venv/bin/python -m experiments.accuracy_sweep_r3 evaluate --run-id sweep-r3` — Expected: `REFUSED: … protocol_sweep_r3.json`.
- [ ] **Step 3:** Assemble and commit `paper/results/calibration-coverage/protocol_sweep_r3.json` from the run dir (NO label read): fields `study: "accuracy-sweep-r3"`, `run_id`, `spec`, `frozen_date_utc`, `glucoedge_commit_at_freeze`, `smoke` (verbatim), `selection_trail` (D/F/A/S verbatim), `winner` (+`best_seed`), `champion` sha256s computed from the live bundled assets, `prediction` ("winner INT8 (uniform_random_200_d101) > champion INT8 on held-out directional recall without breaking the stable floor"), `primary_endpoint` ("held_out directional recall delta, participant-cluster bootstrap, 2000 draws, seed 20261013"), `secondary_endpoints`, `gates` (three, verbatim from spec §6), `pairing_rule` ("identical eligible windows by construction; dropped counts reported in analyze").

```bash
git add paper/results/calibration-coverage/protocol_sweep_r3.json
git commit -m "docs(protocol): freeze accuracy-sweep-r3 confirmatory read before any held-out label"
```

- [ ] **Step 4:** `… evaluate --run-id sweep-r3 && … analyze --run-id sweep-r3`
Expected: `analysis/sweep_r3_aggregate.json` with three gate booleans per split, pairing block, PEDAP block, `per_arm_train_windows`, `ship.all_gates_pass`. Ledger every gate outcome.

### Task 11: Publish + paper

- [ ] **Step 1:** Copy aggregate → `paper/results/calibration-coverage/accuracy_sweep_r3.json` with a `protocol` pointer; run the participant-leak walker (keys matching `participant|ptid|subject|window_key|pred`, long string lists) — Expected: no flags.
- [ ] **Step 2:** Paper subsection after r2's: stage val table (D/F/A, directional + stable, 5-seed means), confirmatory contrast with CI, gates, PEDAP row, macro-F1 cross-ref, the RT-CGM reopen rule in one sentence, the device-local-clock assumption if ToD is in the winner. One table `tab:sweepthree`. Numbers only from `accuracy_sweep_r3.json`.
- [ ] **Step 3:** `cd paper && tectonic main.tex && pdftoppm -r 10 main.pdf /tmp/r3pp && ls /tmp/r3pp* | wc -l` — Expected: clean, ≤ 15 pages (trim prose as prior rounds if it spills).
- [ ] **Step 4:**

```bash
git add paper/results/calibration-coverage/accuracy_sweep_r3.json paper/main.tex paper/main.pdf
git commit -m "docs(paper): accuracy sweep r3 - RT-CGM reopen, time-of-day, TCN"
```

### Task 12: Ship decision — STOP

- **If `ship.all_gates_pass` false:** one MODELS.md sentence ("sweep r3 did not clear the directional gates; bundled pair retained; see accuracy_sweep_r3.json"), commit, report to the user with the full gate table.
- **If true: STOP and tell the user.** Ship shape depends on the winner:
  - contract-unchanged (features "none", arch cnn): r2-style ship (assets + goldens + sha re-pins + both suites + device verification);
  - `tod` winner: extended ship — Kotlin + Swift `DeltaEncoding` grow sin/cos channels, golden-vector schema v2 with per-window timestamps, parity suites updated, device re-verified;
  - `tcn` winner: assets-only ship.
  The ship PR is a separate branch off main after the research PR merges. Ask before pushing (memory rule).

---

## Self-review (done at write time)

- Spec coverage: §2→T2/T7, §3→T4/T5/T8/T9, §4→T3/T4, §5→T1, §6→T5/T6/T10, §7→T11/T12, §8→T1–T6 test steps. No gaps.
- Review Focus items each carry a named test in T1–T4.
- Type consistency: `arm_r3` dict keys consumed by `_tensor_path_r3`/`_ckpt`/`_train_one_r3` match; `TensorView3` supersedes `TensorView` for `[C,L]` rows and the T4 test asserts it.
- Known judgment points left to the executor as explicit rulings: r2-checkpoint reuse (T8 Step 2), conversion float-export entry-point name (T1 Step 5).
