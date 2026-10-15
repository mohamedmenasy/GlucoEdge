# Accuracy Sweep r5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test a cosine/best-val training recipe and a position-preserving flatten head against the shipped champion, confirmed zero-shot on RT-CGM (a cohort whose labels have never been evaluated), with full ship intent.

**Architecture:** `experiments/accuracy_sweep_r5.py` mirrors the r3 stage structure (smoke, train, select, calib, convert, protocol, evaluate, analyze) and reuses r2/r3 helpers. Two small library changes carry the new axes: `TrendCNN` gains a `head="gap"|"flat"` option and `train_model` gains a `cosine` LR-decay flag. Stage R (recipe) then Stage H (head) then Stage S (seed), greedy, 5 seeds each; the confirmation endpoint moves from weinstock held-out to RT-CGM zero-shot, guarded by an identity check (sha-equal candidate skips the read).

**Tech Stack:** Python 3.12 venv (`source venv/bin/activate`), PyTorch (MPS), numpy, ai-edge-litert (conversion env per `conversion/requirements.txt`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-accuracy-sweep-r5-design.md`

## Global Constraints

- RT-CGM is NEVER in any training arm; every r5 arm's mixture is `D5` (spec §3, §6).
- Participant IDs, timestamps, window keys, per-window predictions never leave gitignored `results/` (standing provenance rule; published JSONs carry aggregates only).
- `BOOT_SEED_R5 = 20261015`, single-sourced in `experiments/accuracy_sweep_r5.py`, test-asserted (spec §4).
- Selection floor and Gate 2 use the SAME margin: champion val stable − 2pp (`FLOOR_MARGIN_R5 = 0.02`) (spec §2).
- Tie band 0.1pp (`TIE_BAND = 0.001` from `experiments/accuracy_sweep.py`) → cost_key → list order, at EVERY selection including Stage S and the kept-epoch pick (spec §2, §3).
- Parameter cap 85,450 (2× champion); flatten head is ~50,645, GAP ~42,725 (spec §3).
- Input contract unchanged: `[1, 12]` delta/anchor encoding; D5 tensors reused from `results/calibration-coverage/sweep-r3/tensors/` (SAMPLE_SEED 20261012) — no rebuild (spec §3).
- `protocol_sweep_r5.json` committed clean before any RT-CGM label read; `stage_evaluate` refuses otherwise (spec §4).
- Identity guard: candidate float AND uniform-INT8 sha256-identical to the shipped champion pair → the confirmatory read is skipped (spec §4).
- The pytest suite runs without GlucoBench or DiaData cloned (CLAUDE.md); all r5 tests use synthetic arrays/monkeypatching.
- Long training/eval runs launch unpiped with output redirected to a scratchpad log file; read the full log (`python … > log 2>&1`, never `| tail` — pipes mask exit codes).
- Code style: lazy imports for torch/GlucoBench-touching modules inside functions, as in r3/r4.

## Review Focus

1. **Stale alias drift** — `R_base`/`H_gap` reuse `sweep-r3/checkpoints/D5_none_cnn_seed{N}.pt` by path; if those files were regenerated or differ from what r3 selected, r5 compares against a silently different model. Pinned: alias-missing refusal test (Task 3) + aliased-checkpoint sha256s recorded in the protocol (Task 9).
2. **Admit-list drift on RT-CGM** — an admitted-IDs/cache mismatch yields an empty or shrunken eval set and a meaningless "confirmation". Pinned: `keep` count guard with loud refusal + test (Task 8). (This also retires the r3 deferred minor about the missing `keep.any()` guard.)
3. **Identity-guard scope** — the guard must require BOTH artifacts (float and uniform INT8) identical; one-of-two identical is a real (quantization-path) difference and must not skip the read. Pinned: both-identical vs one-identical test (Task 8).
4. **Epoch-checkpoint head mismatch** — `train_model` saves raw per-epoch state_dicts; the epoch evaluator must rebuild the arm's own head or flat weights silently fail to load into a gap model. Pinned: epoch-table test on a flat arm (Task 5).
5. **Undefined bootstrap CI at gate time** — `paired_cluster_bootstrap` returns `ci95: None` when >5% of draws are undefined; Gate 1 must then FAIL loudly, never pass vacuously. Pinned: `ci is None → gate1 False` test (Task 9).

## File Structure

- `training/model.py` — `TrendCNN` gains `head` (+`length`) kwargs; `TrendTCN` untouched.
- `training/train.py` — `train_model` gains `cosine=False` kwarg (CosineAnnealingLR over `epochs`).
- `conversion/common.py` — `load_checkpoint` passes `head` through (absent → `"gap"`).
- `experiments/accuracy_sweep_r5.py` — new; all r5 stages + constants + selection logic.
- `tests/test_model_r5.py` — flatten head: shapes, param counts, LiteRT smoke, head-aware load + backward compat.
- `tests/test_train_cosine.py` — cosine flag wiring.
- `tests/test_accuracy_sweep_r5.py` — everything else (constants, arms, aliases, selection, guards, analyze flows, leak walk).

---

### Task 1: Flatten head on TrendCNN + head-aware checkpoint load

**Files:**
- Modify: `training/model.py:4-18` (TrendCNN)
- Modify: `conversion/common.py:9-27` (load_checkpoint)
- Test: `tests/test_model_r5.py`

**Interfaces:**
- Consumes: existing `TrendCNN(num_classes, channels, in_channels)`; `load_checkpoint(checkpoint_path, num_classes)`.
- Produces: `TrendCNN(num_classes=5, channels=(96,144), in_channels=1, head="gap"|"flat", length=12)`; `load_checkpoint` honoring an optional `"head"` key in the meta dict (absent → `"gap"`; raw state_dicts unchanged → gap CNN). Later tasks save/read meta dicts `{"state_dict", "arch": "cnn", "in_channels": 1, "channels": [96,144], "head": a["head"]}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_r5.py
import numpy as np
import pytest
import torch

from training.model import TrendCNN

PARAM_CAP = 85_450  # 2x champion (spec section 3)


def _params(m):
    return sum(p.numel() for p in m.parameters())


def test_flat_head_shapes_and_params():
    gap = TrendCNN(num_classes=5, channels=(96, 144))
    flat = TrendCNN(num_classes=5, channels=(96, 144), head="flat")
    x = torch.zeros(2, 1, 12)
    assert gap(x).shape == (2, 5)
    assert flat(x).shape == (2, 5)
    assert _params(gap) == 42_725
    assert _params(flat) == 50_645
    assert _params(flat) <= PARAM_CAP


def test_flat_head_sees_position():
    # same values, different order -> different logits (GAP can't do this)
    torch.manual_seed(0)
    flat = TrendCNN(num_classes=5, channels=(96, 144), head="flat").eval()
    a = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    b = torch.flip(a, dims=[-1])
    with torch.no_grad():
        assert not torch.allclose(flat(a), flat(b))


def test_load_checkpoint_head_aware_and_backward_compat(tmp_path):
    from conversion.common import load_checkpoint
    flat = TrendCNN(num_classes=5, channels=(96, 144), head="flat")
    gap = TrendCNN(num_classes=5, channels=(96, 144))
    meta_flat = {"state_dict": flat.state_dict(), "arch": "cnn",
                 "in_channels": 1, "channels": [96, 144], "head": "flat"}
    meta_gap_no_head = {"state_dict": gap.state_dict(), "arch": "cnn",
                        "in_channels": 1, "channels": [96, 144]}
    pf, pg, praw = tmp_path / "f.pt", tmp_path / "g.pt", tmp_path / "raw.pt"
    torch.save(meta_flat, pf)
    torch.save(meta_gap_no_head, pg)
    torch.save(gap.state_dict(), praw)  # r2-era raw dict
    mf = load_checkpoint(pf, 5)
    assert mf.fc.in_features == 144 * 12
    mg = load_checkpoint(pg, 5)        # absent head -> gap
    assert mg.fc.in_features == 144
    mr = load_checkpoint(praw, 5)      # raw state_dict -> gap
    assert mr.fc.in_features == 144


def test_flat_head_converts_to_litert(tmp_path):
    pytest.importorskip("litert_torch")
    from conversion.convert import convert_float, convert_int8
    model = TrendCNN(num_classes=5, channels=(96, 144), head="flat").eval()
    sample = torch.zeros(1, 1, 12)
    fpath = tmp_path / "flat_float.tflite"
    convert_float(model, sample, fpath)
    assert fpath.exists() and fpath.stat().st_size > 0
    calib = [torch.randn(1, 1, 12) for _ in range(8)]
    ipath = tmp_path / "flat_int8.tflite"
    convert_int8(model, sample, calib, ipath)
    assert ipath.exists() and ipath.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/test_model_r5.py -v > /private/tmp/claude-501/-Users-mohamednabil-Documents-vibe-GlucoEdge/c6b0a325-2e88-4527-b3b6-fd0299ef8c63/scratchpad/t1_red.log 2>&1; cat "$_"` (read the whole log)
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'head'` (or similar) in each test.

- [ ] **Step 3: Implement**

In `training/model.py`, replace `TrendCNN`:

```python
class TrendCNN(nn.Module):
    def __init__(self, num_classes: int = 5, channels=(24, 36),
                 in_channels: int = 1, head: str = "gap", length: int = 12):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.head = head
        if head == "flat":
            # position-preserving: classify the full [c2, length] feature map
            self.pool = None
            self.fc = nn.Linear(c2 * length, num_classes)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.pool is None:
            return self.fc(x.flatten(1))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

In `conversion/common.py`, the `"state_dict"` branch's CNN constructor becomes:

```python
        else:
            model = TrendCNN(num_classes=num_classes,
                             channels=tuple(obj["channels"]),
                             in_channels=obj["in_channels"],
                             head=obj.get("head", "gap"))
```

(The raw-state_dict fallback and TCN branch stay exactly as they are.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_r5.py -v`
Expected: PASS (4/4; the LiteRT test may SKIP in the training venv — run it under the conversion env once: `conversion/venv` per `conversion/requirements.txt`, or accept the skip here and rely on the conversion-env run in Task 12).

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest -q > .../scratchpad/t1_suite.log 2>&1; tail -5 .../scratchpad/t1_suite.log` then read the log for any failure anywhere in it.
Expected: all pass (157+ passed, skips OK).

```bash
git add training/model.py conversion/common.py tests/test_model_r5.py
git commit -m "feat(model): flatten head option on TrendCNN; head-aware checkpoint load"
```

---

### Task 2: Cosine LR decay flag in train_model

**Files:**
- Modify: `training/train.py:74-106` (train_model)
- Test: `tests/test_train_cosine.py`

**Interfaces:**
- Consumes: `train_model(train_ds, val_ds, num_classes, epochs, device, model=None, criterion=None, batch_size=64, collate_fn=None, epoch_dir=None)`.
- Produces: same signature + `cosine: bool = False`. When True, `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)` steps once per epoch (lr decays 1e-3 → ~0 over the run). Default False = bit-identical behavior to today.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_cosine.py
import numpy as np
import torch

from training.train import train_model


class _Tiny:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(64, 1, 12)).astype(np.float32)
        self.y = rng.integers(0, 5, size=64)
        self.labels = self.y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, k):
        return torch.from_numpy(self.x[k]), int(self.y[k])


def _run(cosine):
    torch.manual_seed(0)
    np.random.seed(0)
    ds = _Tiny()
    model = train_model(ds, ds, 5, 3, torch.device("cpu"),
                        batch_size=16, cosine=cosine)
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_cosine_changes_training_and_default_is_off():
    base = _run(False)
    again = _run(False)
    cos = _run(cosine=True)
    assert torch.equal(base, again)          # determinism on cpu
    assert not torch.equal(base, cos)        # scheduler actually wired
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_cosine.py -v`
Expected: FAIL — `TypeError: train_model() got an unexpected keyword argument 'cosine'`.

- [ ] **Step 3: Implement**

In `training/train.py`, `train_model` signature gains `cosine=False`; after the optimizer line add:

```python
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if cosine else None)
```

and at the end of each epoch loop body (after the val-loss print, before the `epoch_dir` save):

```python
        if scheduler is not None:
            scheduler.step()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_cosine.py tests/test_train.py -v`
Expected: PASS (new test + existing train tests untouched).

- [ ] **Step 5: Commit**

```bash
git add training/train.py tests/test_train_cosine.py
git commit -m "feat(train): optional cosine LR decay in train_model"
```

---

### Task 3: r5 module skeleton — constants, arms, checkpoint paths and the r3 alias

**Files:**
- Create: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: `TIE_BAND` from `experiments.accuracy_sweep`; `SAMPLE_SEED, EPOCHS` from `experiments.accuracy_sweep_r2`; `GLUCOEDGE_ROOT, _run_dir, _write_json` from `experiments.calibration_coverage`.
- Produces (used by every later task):
  - `ARM_SEEDS_R5 = (0, 1, 2, 3, 4)`, `BOOT_SEED_R5 = 20261015`, `FLOOR_MARGIN_R5 = 0.02`, `PARAM_CAP_R5 = 85_450`, `PROTOCOL_R5` (Path), `R3_RUN_DIR` (Path), `DEVIATIONS_R5` (tuple), `DISCLOSURE_RTCGM` (str).
  - `arm_r5(name, head, recipe) -> dict` with keys `name, mixture("D5"), features("none"), arch("cnn"), head, recipe, in_channels(1), channels([96,144]), length(12)`.
  - `stage_r_arms() -> [R_base(gap,base), R_val(gap,val)]`; `stage_h_arms(r_winner) -> [H_gap(= winner reused), H_flat(flat, winner's recipe)]`.
  - `_ckpt_r5(run, a, seed) -> Path` — gap+base aliases `R3_RUN_DIR/checkpoints/D5_none_cnn_seed{N}.pt` (refusing if missing); everything else `{run}/checkpoints/{mixture}_{features}_{arch}_{head}_{recipe}_seed{N}.pt`.
  - `_tensor_path_r5() -> Path` — `R3_RUN_DIR/tensors/D5_none_s{SAMPLE_SEED}.npz`, SystemExit if missing (no rebuild).
  - `_read_sel(run, stage)` (same shape as r3's).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_accuracy_sweep_r5.py
import json

import numpy as np
import pytest

from experiments.accuracy_sweep_r5 import (ARM_SEEDS_R5, BOOT_SEED_R5,
                                           FLOOR_MARGIN_R5, PARAM_CAP_R5,
                                           _ckpt_r5, _tensor_path_r5, arm_r5,
                                           stage_h_arms, stage_r_arms)


def test_constants():
    assert ARM_SEEDS_R5 == (0, 1, 2, 3, 4)
    assert BOOT_SEED_R5 == 20261015
    assert FLOOR_MARGIN_R5 == 0.02  # selection floor == Gate 2 margin
    assert PARAM_CAP_R5 == 85_450


def test_arms_never_touch_rtcgm():
    r = stage_r_arms()
    arms = r + stage_h_arms(r[1])
    assert all(a["mixture"] == "D5" for a in arms)
    assert [a["name"] for a in r] == ["R_base", "R_val"]
    assert [a["recipe"] for a in r] == ["base", "val"]
    assert all(a["head"] == "gap" for a in r)


def test_stage_h_inherits_r_winner_recipe():
    h = stage_h_arms(arm_r5("R_val", "gap", "val"))
    assert [a["name"] for a in h] == ["H_gap", "H_flat"]
    assert h[0]["head"] == "gap" and h[0]["recipe"] == "val"
    assert h[1]["head"] == "flat" and h[1]["recipe"] == "val"
    h2 = stage_h_arms(arm_r5("R_base", "gap", "base"))
    assert h2[1]["recipe"] == "base"  # flat trains with the winning recipe


def test_ckpt_key_covers_head_and_recipe(tmp_path):
    a = arm_r5("H_flat", "flat", "val")
    p = _ckpt_r5(tmp_path, a, 3)
    assert p.name == "D5_none_cnn_flat_val_seed3.pt"
    assert p.parent == tmp_path / "checkpoints"


def test_r_base_aliases_r3_checkpoints(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r5 as r5
    fake = tmp_path / "r3"
    (fake / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(r5, "R3_RUN_DIR", fake)
    a = arm_r5("R_base", "gap", "base")
    with pytest.raises(SystemExit, match="REFUSED"):
        _ckpt_r5(tmp_path, a, 0)  # alias target missing -> loud
    (fake / "checkpoints" / "D5_none_cnn_seed0.pt").write_bytes(b"x")
    assert _ckpt_r5(tmp_path, a, 0).name == "D5_none_cnn_seed0.pt"
    # H_gap carrying the base recipe resolves to the same alias
    hg = dict(a, name="H_gap")
    assert _ckpt_r5(tmp_path, hg, 0) == _ckpt_r5(tmp_path, a, 0)


def test_tensor_path_is_r3_d5_and_refuses_if_missing(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "R3_RUN_DIR", tmp_path)
    with pytest.raises(SystemExit, match="REFUSED"):
        _tensor_path_r5()
    t = tmp_path / "tensors"
    t.mkdir()
    (t / "D5_none_s20261012.npz").write_bytes(b"x")
    assert _tensor_path_r5().name == "D5_none_s20261012.npz"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.accuracy_sweep_r5'`.

- [ ] **Step 3: Implement the module skeleton**

```python
# experiments/accuracy_sweep_r5.py
"""Accuracy sweep r5: training recipe + position-preserving head, confirmed
zero-shot on RT-CGM (spec 2026-10-14-accuracy-sweep-r5-design.md).

Selection stays on weinstock val; champion is the bundled (shipped r2) pair
read live from android assets; the D5 tensors and R_base checkpoints are
reused from the sweep-r3 run directory (no rebuild, no retraining)."""
import json
from pathlib import Path

import numpy as np

from experiments.accuracy_sweep import TIE_BAND, protocol_is_committed
from experiments.accuracy_sweep_r2 import (EPOCHS, SAMPLE_SEED,
                                           champion_val_stats,
                                           directional_recall, stable_recall)
from experiments.calibration_coverage import GLUCOEDGE_ROOT, _run_dir, _write_json

ARM_SEEDS_R5 = (0, 1, 2, 3, 4)
BOOT_SEED_R5 = 20261015  # pre-registered (protocol_sweep_r5.json); single source
FLOOR_MARGIN_R5 = 0.02   # selection floor == Gate 2: champion val stable - 2pp
PARAM_CAP_R5 = 85_450    # 2x champion
PROTOCOL_R5 = (GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage"
               / "protocol_sweep_r5.json")
R3_RUN_DIR = GLUCOEDGE_ROOT / "results" / "calibration-coverage" / "sweep-r3"
RTCGM_CACHE = GLUCOEDGE_ROOT / "results" / "diadata_cache" / "hours" / "RT_CGM.npz"

DISCLOSURE_RTCGM = (
    "RT-CGM labels have never been evaluated in any round. Training-exposure "
    "disclosure: r3's rejected D_rtcgm arm trained on this cohort, but "
    "neither the shipped champion nor any r5 arm trains on it, and no model "
    "under comparison has seen its labels at training or evaluation time.")

# Disclosed with the published aggregate; the frozen protocol is not edited.
DEVIATIONS_R5 = ()


def arm_r5(name, head, recipe):
    return {"name": name, "mixture": "D5", "features": "none", "arch": "cnn",
            "head": head, "recipe": recipe, "in_channels": 1,
            "channels": [96, 144], "length": 12}


def stage_r_arms():
    return [arm_r5("R_base", "gap", "base"), arm_r5("R_val", "gap", "val")]


def stage_h_arms(r_winner):
    return [dict(r_winner, name="H_gap"),
            arm_r5("H_flat", "flat", r_winner["recipe"])]


def _ckpt_r5(run, a, seed):
    if a["head"] == "gap" and a["recipe"] == "base":
        # ledgered alias: r3's D_base checkpoints (= r2 M_M3), reused unchanged
        p = R3_RUN_DIR / "checkpoints" / f"D5_none_cnn_seed{seed}.pt"
        if not p.exists():
            raise SystemExit(f"REFUSED: R_base alias target missing: {p}")
        return p
    return (Path(run) / "checkpoints"
            / f"{a['mixture']}_{a['features']}_{a['arch']}_{a['head']}"
              f"_{a['recipe']}_seed{seed}.pt")


def _tensor_path_r5():
    p = R3_RUN_DIR / "tensors" / f"D5_none_s{SAMPLE_SEED}.npz"
    if not p.exists():
        raise SystemExit(f"REFUSED: reused D5 tensor missing: {p} - r5 does "
                         "not rebuild tensors (spec section 3)")
    return p


def _read_sel(run, stage):
    return json.loads((Path(run) / "selection" / f"{stage}.json").read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: PASS (6/6).

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): sweep module skeleton - arms, constants, r3 aliases"
```

---

### Task 4: Selection logic — pick_epoch, pick_winner_r5, Stage-S seed pick

**Files:**
- Modify: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: `TIE_BAND`; `best_seed_r4(per_seed: {str: (dir, stable)}, floor) -> int` from `experiments.accuracy_sweep_r4` (tie band + lowest seed + per-seed floor — exactly the r5 Stage-S rule; reused, already tested in `tests/test_accuracy_sweep_r4.py`).
- Produces:
  - `pick_epoch(per_epoch: {int: (dir, stable)}, floor) -> (epoch: int, flagged: bool)` — eligible epochs have stable ≥ floor; among eligible, tie band on directional then EARLIEST epoch (fewer steps = cheaper); all ineligible → (final epoch, True).
  - `_param_count(a) -> int` (instantiates TrendCNN, sums numel).
  - `pick_winner_r5(arms, means: {name: mean_dir}) -> arm` — tie band → cost_key (param count, then recipe base-before-val, then list order).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_accuracy_sweep_r5.py`)

```python
def test_pick_epoch_floor_tieband_and_flag():
    from experiments.accuracy_sweep_r5 import pick_epoch
    floor = 0.50
    per_epoch = {0: (0.54, 0.49),   # best directional but fails floor
                 1: (0.530, 0.52),
                 2: (0.5305, 0.51),  # inside 0.1pp band of epoch 3
                 3: (0.5309, 0.51)}
    # band is {2, 3}; earliest wins
    assert pick_epoch(per_epoch, floor) == (2, False)
    # every epoch fails the floor -> final epoch kept, flagged
    assert pick_epoch({0: (0.6, 0.1), 1: (0.5, 0.2)}, floor) == (1, True)


def test_param_count_matches_model_r5():
    from experiments.accuracy_sweep_r5 import _param_count, arm_r5
    assert _param_count(arm_r5("H_gap", "gap", "val")) == 42_725
    assert _param_count(arm_r5("H_flat", "flat", "val")) == 50_645
    assert _param_count(arm_r5("H_flat", "flat", "val")) <= 85_450


def test_pick_winner_r5_tie_band_cost_then_order():
    from experiments.accuracy_sweep_r5 import arm_r5, pick_winner_r5
    gap = arm_r5("H_gap", "gap", "val")
    flat = arm_r5("H_flat", "flat", "val")
    # inside the band: cheaper head (gap) wins even listed second
    w = pick_winner_r5([flat, gap], {"H_flat": 0.5305, "H_gap": 0.5300})
    assert w["name"] == "H_gap"
    # outside the band: argmax wins
    w = pick_winner_r5([flat, gap], {"H_flat": 0.5320, "H_gap": 0.5300})
    assert w["name"] == "H_flat"
    # recipe base before val at equal params
    base = arm_r5("R_base", "gap", "base")
    val = arm_r5("R_val", "gap", "val")
    w = pick_winner_r5([base, val], {"R_base": 0.5300, "R_val": 0.5305})
    assert w["name"] == "R_base"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy_sweep_r5.py -v -k "pick_epoch or param_count or pick_winner"`
Expected: FAIL — `ImportError: cannot import name 'pick_epoch'`.

- [ ] **Step 3: Implement** (append to `experiments/accuracy_sweep_r5.py`)

```python
def pick_epoch(per_epoch, floor):
    """Kept epoch under the recipe rule: max val directional subject to val
    stable >= floor; tie band then earliest epoch. All fail -> final epoch,
    flagged (spec section 3)."""
    ok = {e: v for e, v in per_epoch.items() if v[1] >= floor}
    if not ok:
        return max(per_epoch), True
    best = max(v[0] for v in ok.values())
    return min(e for e, v in ok.items() if best - v[0] <= TIE_BAND), False


def _param_count(a):
    from training.model import TrendCNN
    m = TrendCNN(num_classes=5, channels=tuple(a["channels"]),
                 in_channels=a["in_channels"], head=a["head"])
    return sum(p.numel() for p in m.parameters())


def pick_winner_r5(arms, means):
    best = max(means.values())
    order = [a["name"] for a in arms]
    band = [a for a in arms if a["name"] in means
            and best - means[a["name"]] <= TIE_BAND]
    band.sort(key=lambda a: (_param_count(a), a["recipe"] != "base",
                             order.index(a["name"])))
    return band[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): epoch/winner selection - aligned floor, tie band everywhere"
```

---

### Task 5: Training stage — recipes, epoch table, reuse refusal, smoke delegate

**Files:**
- Modify: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: `_load_view_r3(path, in_channels)`, `_val_for(features)` from `experiments.accuracy_sweep_r3`; `_frozen_batch(run)`, `stage_smoke` from `experiments.accuracy_sweep_r2`; `train_model(..., cosine=, epoch_dir=)` (Task 2); `pick_epoch` (Task 4); `_ckpt_r5`, `_tensor_path_r5` (Task 3).
- Produces:
  - `_epoch_table(epoch_dir, a, val_v) -> {int: (dir, stable)}` — rebuilds the ARM'S OWN head per epoch (raw state_dicts in `epoch_dir`).
  - `_val_stats_r5(run, a, seed) -> (dir, stable)` — head-aware via `load_checkpoint` on `_ckpt_r5`.
  - `_train_one_r5(run, a, seed, train_v, val_v, batch) -> Path` — recipe `"val"`: cosine + per-epoch checkpoints + `pick_epoch`; recipe `"base"` (only reachable as a trained arm through H_flat-with-base-recipe): constant LR, final epoch. Saves the meta dict with `"head"`, sidecar JSON with `kept_epoch`, `epoch_floor_flagged`, `per_epoch`.
  - `stage_train(run_id, stage)` — stages `"R"`/`"H"`; refuses to train reuse cells (gap+base alias); refuses any arm whose `mixture != "D5"`.
  - `stage_smoke(run_id)` — delegate to r2's (batch freeze + bit-repro record in the r5 run dir).

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_epoch_table_rebuilds_arm_head(tmp_path):
    import torch
    from experiments.accuracy_sweep_r5 import _epoch_table, arm_r5
    from training.model import TrendCNN

    class _V:
        x = np.zeros((4, 1, 12), dtype=np.float32)
        y = np.array([0, 1, 2, 3])
    a = arm_r5("H_flat", "flat", "val")
    ed = tmp_path / "epochs"
    ed.mkdir()
    for e in range(2):
        m = TrendCNN(num_classes=5, channels=(96, 144), head="flat")
        torch.save({k: v for k, v in m.state_dict().items()},
                   ed / f"epoch{e:02d}.pt")
    table = _epoch_table(ed, a, _V())
    assert set(table) == {0, 1}
    assert all(len(v) == 2 for v in table.values())


def test_stage_train_refuses_reuse_and_foreign_mixture(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    monkeypatch.setattr(r5, "_frozen_batch", lambda run: 512)
    # reuse cell with missing alias target -> loud refusal, no training
    monkeypatch.setattr(r5, "R3_RUN_DIR", tmp_path / "nope")
    monkeypatch.setattr(r5, "_arms_for",
                        lambda run, s: [r5.arm_r5("R_base", "gap", "base")])
    with pytest.raises(SystemExit, match="REFUSED"):
        r5.stage_train("sweep-r5-test", "R")
    # any non-D5 mixture -> refusal before any data is touched
    bad = dict(r5.arm_r5("R_val", "gap", "val"), mixture="D5R")
    monkeypatch.setattr(r5, "_arms_for", lambda run, s: [bad])
    with pytest.raises(SystemExit, match="RT-CGM|D5"):
        r5.stage_train("sweep-r5-test", "R")


def test_train_one_r5_val_recipe_keeps_selected_epoch(tmp_path, monkeypatch):
    import torch
    import experiments.accuracy_sweep_r5 as r5

    class _V:
        def __init__(self, n=32):
            rng = np.random.default_rng(0)
            self.x = rng.normal(size=(n, 1, 12)).astype(np.float32)
            self.y = rng.integers(0, 5, size=n).astype(np.int64)
            self.labels = self.y.tolist()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, k):
            return torch.from_numpy(self.x[k]), int(self.y[k])

    import training.train as tt
    monkeypatch.setattr(r5, "EPOCHS", 2)
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.5, "stable": 0.0})
    monkeypatch.setattr(tt, "resolve_device",
                        lambda prefer="mps": torch.device("cpu"))
    a = r5.arm_r5("R_val", "gap", "val")
    v = _V()
    ckpt = r5._train_one_r5(tmp_path, a, 0, v, v, batch=16)
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert obj["head"] == "gap" and obj["arch"] == "cnn"
    side = json.loads(ckpt.with_suffix(".json").read_text())
    assert "kept_epoch" in side and side["kept_epoch"] in (0, 1)
    assert side["epoch_floor_flagged"] is False  # floor 0.0-0.02 passes
    # the saved weights ARE the kept epoch's weights
    ep = (tmp_path / "checkpoints" / "epochs" / "R_val_seed0"
          / f"epoch{side['kept_epoch']:02d}.pt")
    kept = torch.load(ep, map_location="cpu", weights_only=True)
    assert all(torch.equal(obj["state_dict"][k], kept[k]) for k in kept)
```

(Note: the `resolve_device` monkeypatch targets `training.train.resolve_device`, which `_train_one_r5` imports lazily — patch it on the `training.train` module.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy_sweep_r5.py -v -k "epoch_table or refuses_reuse or keeps_selected"`
Expected: FAIL — `AttributeError`/`ImportError` on `_epoch_table`, `stage_train`, `_train_one_r5`.

- [ ] **Step 3: Implement** (append; follow the r3 shapes)

```python
def _epoch_table(epoch_dir, a, val_v):
    """Per-epoch (directional, stable) on val; rebuilds the arm's own head
    because train_model saves raw state_dicts."""
    import torch
    from experiments.analyze_calibration_coverage import confusion_counts
    from training.model import TrendCNN
    table = {}
    for p in sorted(Path(epoch_dir).glob("epoch*.pt")):
        model = TrendCNN(num_classes=5, channels=tuple(a["channels"]),
                         in_channels=a["in_channels"], head=a["head"])
        model.load_state_dict(torch.load(p, map_location="cpu",
                                         weights_only=True))
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(val_v.x), 8192):
                preds.append(model(torch.from_numpy(val_v.x[i:i + 8192]))
                             .numpy().argmax(axis=1))
        cm = confusion_counts(val_v.y, np.concatenate(preds))
        table[int(p.stem.replace("epoch", ""))] = (directional_recall(cm),
                                                   stable_recall(cm))
    return table


def _val_stats_r5(run, a, seed):
    """(directional, stable) on eligible weinstock val; head-aware load."""
    import torch
    from conversion.common import load_checkpoint
    from experiments.accuracy_sweep_r3 import _val_for
    from experiments.analyze_calibration_coverage import confusion_counts
    val_v = _val_for("none")
    model = load_checkpoint(_ckpt_r5(run, a, seed), 5)
    preds = []
    with torch.no_grad():
        for i in range(0, len(val_v.x), 8192):
            preds.append(model(torch.from_numpy(val_v.x[i:i + 8192]))
                         .numpy().argmax(axis=1))
    cm = confusion_counts(val_v.y, np.concatenate(preds))
    return directional_recall(cm), stable_recall(cm)


def _train_one_r5(run, a, seed, train_v, val_v, batch):
    import time as _t

    import torch
    from training.losses import make_criterion
    from training.model import TrendCNN
    from training.train import class_weights, resolve_device, train_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device("mps")
    model = TrendCNN(num_classes=5, channels=tuple(a["channels"]),
                     in_channels=a["in_channels"], head=a["head"])
    criterion = make_criterion("ce", class_weights(train_v, 5).to(device))
    cosine = a["recipe"] == "val"
    epoch_dir = (Path(run) / "checkpoints" / "epochs" / f"{a['name']}_seed{seed}"
                 if cosine else None)
    t0 = _t.time()
    model = train_model(train_v, val_v, 5, EPOCHS, device, model=model,
                        criterion=criterion, batch_size=batch,
                        cosine=cosine, epoch_dir=epoch_dir)
    kept_epoch, flagged, per_epoch = EPOCHS - 1, False, None
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    if cosine:
        floor = champion_val_stats(run)["stable"] - FLOOR_MARGIN_R5
        per_epoch = _epoch_table(epoch_dir, a, val_v)
        kept_epoch, flagged = pick_epoch(per_epoch, floor)
        state = torch.load(epoch_dir / f"epoch{kept_epoch:02d}.pt",
                           map_location="cpu", weights_only=True)
    ckpt = _ckpt_r5(run, a, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "arch": "cnn",
                "in_channels": a["in_channels"], "channels": a["channels"],
                "head": a["head"]}, ckpt)
    ckpt.with_suffix(".json").write_text(json.dumps({
        "arm": a["name"], "head": a["head"], "recipe": a["recipe"],
        "seed": seed, "device": device.type, "batch": batch,
        "train_windows": len(train_v), "epochs": EPOCHS,
        "kept_epoch": kept_epoch, "epoch_floor_flagged": flagged,
        "per_epoch": ({str(e): list(v) for e, v in per_epoch.items()}
                      if per_epoch else None),
        "duration_s": round(_t.time() - t0, 1)}, indent=1))
    return ckpt


def _arms_for(run, stage):
    if stage == "R":
        return stage_r_arms()
    if stage == "H":
        return stage_h_arms(_read_sel(run, "R")["winner"])
    raise SystemExit(f"unknown stage {stage}")


def stage_train(run_id, stage):
    import time as _t
    from experiments.accuracy_sweep_r3 import _load_view_r3, _val_for
    run = _run_dir(run_id)
    batch = _frozen_batch(run)  # module-level wrapper (monkeypatchable)
    for a in _arms_for(run, stage):
        if a["mixture"] != "D5":
            raise SystemExit("REFUSED: r5 trains on D5 only - RT-CGM and all "
                             f"other mixtures are out of scope, got {a['mixture']}")
        train_v = None
        for seed in ARM_SEEDS_R5:
            ckpt = _ckpt_r5(run, a, seed)  # alias check happens in here
            if ckpt.exists():
                print(f"skip {ckpt.name}: exists")
                continue
            if a["head"] == "gap" and a["recipe"] == "base":
                raise SystemExit(f"REFUSED: {a['name']} is a reuse cell - its "
                                 "checkpoints live in the sweep-r3 run dir")
            if train_v is None:
                train_v = _load_view_r3(_tensor_path_r5(), a["in_channels"])
            t0 = _t.time()
            _train_one_r5(run, a, seed, train_v, _val_for("none"), batch)
            print(f"{a['name']} seed {seed} done in {_t.time()-t0:.0f}s")


def _frozen_batch(run):
    from experiments.accuracy_sweep_r2 import _frozen_batch as _fb
    return _fb(run)


def stage_smoke(run_id):
    """Batch freeze + same-seed bit-reproduction record, r2 recipe."""
    from experiments.accuracy_sweep_r2 import stage_smoke as smoke_r2
    smoke_r2(run_id)
```

(Keep `_frozen_batch` as the thin wrapper above so tests can monkeypatch `r5._frozen_batch`; `stage_train` must call `_frozen_batch(run)`, the module-level wrapper — adjust the duplicate import accordingly.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: PASS (training test runs ~2 epochs on 32 CPU windows — seconds).

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): training stage - cosine/best-val recipe, reuse refusals, smoke"
```

---

### Task 6: Stage select — R, H (floor −2pp, tie band) and S (seed pick)

**Files:**
- Modify: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: `_val_stats_r5`, `_arms_for`, `pick_winner_r5` (Tasks 4–5); `best_seed_r4` from `experiments.accuracy_sweep_r4`; `champion_val_stats` from r2.
- Produces: `stage_select(run_id, stage)` for `"R"`, `"H"`, `"S"`. Selection JSON shape mirrors r3: `{stage, champion_val, stable_floor, table, winner}`; Stage S adds `best_seed: int`. Flagged seeds (sidecar `epoch_floor_flagged`) surface in the table as `"epoch_floor_flagged": true`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _fake_stats(stats):
    """stats: {(arm_name, seed): (dir, stable)}"""
    def f(run, a, seed):
        return stats[(a["name"], seed)]
    return f


def test_select_floor_is_2pp_and_tie_prefers_cheap(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.533, "stable": 0.5485})
    # R_val mean stable sits between -2pp and -5pp below champion: with the
    # r5 floor it is DISQUALIFIED (under r3's -5pp it would have survived)
    stats = {("R_base", s): (0.530, 0.548) for s in range(5)}
    stats.update({("R_val", s): (0.540, 0.520) for s in range(5)})  # -2.85pp
    monkeypatch.setattr(r5, "_val_stats_r5", _fake_stats(stats))
    monkeypatch.setattr(r5, "_flag_for", lambda run, a, seed: False)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    out = r5.stage_select("sweep-r5-test", "R")
    assert out["stable_floor"] == pytest.approx(0.5485 - 0.02)
    assert out["table"]["R_val"]["disqualified"] is True
    assert out["winner"]["name"] == "R_base"


def test_select_all_dq_refuses(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.533, "stable": 0.5485})
    stats = {(n, s): (0.54, 0.40) for n in ("R_base", "R_val") for s in range(5)}
    monkeypatch.setattr(r5, "_val_stats_r5", _fake_stats(stats))
    monkeypatch.setattr(r5, "_flag_for", lambda run, a, seed: False)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    with pytest.raises(SystemExit, match="REFUSED"):
        r5.stage_select("sweep-r5-test", "R")


def test_stage_s_tie_band_and_floor(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.533, "stable": 0.5485})
    (tmp_path / "selection").mkdir(parents=True)
    (tmp_path / "selection" / "H.json").write_text(json.dumps(
        {"winner": r5.arm_r5("H_flat", "flat", "val")}))
    # seeds 1 and 2 inside the 0.1pp band: lowest seed wins (r3's deviation
    # would have argmaxed to seed 2); seed 0 fails the per-seed floor
    stats = {("H_flat", 0): (0.560, 0.500),
             ("H_flat", 1): (0.5366, 0.536),
             ("H_flat", 2): (0.5367, 0.526),
             ("H_flat", 3): (0.530, 0.54),
             ("H_flat", 4): (0.531, 0.53)}
    monkeypatch.setattr(r5, "_val_stats_r5", _fake_stats(stats))
    monkeypatch.setattr(r5, "_flag_for", lambda run, a, seed: seed == 4)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    out = r5.stage_select("sweep-r5-test", "S")
    assert out["best_seed"] == 1
    assert out["table"]["per_seed"]["4"]["epoch_floor_flagged"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy_sweep_r5.py -v -k "select or stage_s"`
Expected: FAIL — `AttributeError: ... has no attribute 'stage_select'` (and `_flag_for`).

- [ ] **Step 3: Implement** (append)

```python
def _flag_for(run, a, seed):
    """True if the seed's sidecar says every epoch failed the floor."""
    side = _ckpt_r5(run, a, seed).with_suffix(".json")
    if not side.exists():
        return False  # reuse cells have no r5 sidecar
    return bool(json.loads(side.read_text()).get("epoch_floor_flagged", False))


def _select(run, stage, arms):
    champ = champion_val_stats(run)
    floor = champ["stable"] - FLOOR_MARGIN_R5
    table, means = {}, {}
    for a in arms:
        per_seed = {str(s): _val_stats_r5(run, a, s) for s in ARM_SEEDS_R5}
        flags = {str(s): _flag_for(run, a, s) for s in ARM_SEEDS_R5}
        dir_mean = float(np.mean([v[0] for v in per_seed.values()]))
        stab_mean = float(np.mean([v[1] for v in per_seed.values()]))
        dq = stab_mean < floor
        table[a["name"]] = {
            "per_seed": {s: {"directional": v[0], "stable": v[1],
                             "epoch_floor_flagged": flags[s]}
                         for s, v in per_seed.items()},
            "mean_directional": dir_mean, "mean_stable": stab_mean,
            "disqualified": bool(dq)}
        if not dq:
            means[a["name"]] = dir_mean
    if not means:
        raise SystemExit("REFUSED: every arm failed the stable-recall "
                         "disqualifier; no winner can be selected")
    winner = pick_winner_r5(arms, means)
    out = {"stage": stage, "champion_val": champ, "stable_floor": floor,
           "table": table, "winner": winner}
    sel = Path(run) / "selection"
    sel.mkdir(parents=True, exist_ok=True)
    _write_json(sel / f"{stage}.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
    return out


def stage_select(run_id, stage):
    from experiments.accuracy_sweep_r4 import best_seed_r4
    run = _run_dir(run_id)
    if stage in ("R", "H"):
        return _select(run, stage, _arms_for(run, stage))
    if stage != "S":
        raise SystemExit(f"unknown stage {stage}")
    a = _read_sel(run, "H")["winner"]
    champ = champion_val_stats(run)
    floor = champ["stable"] - FLOOR_MARGIN_R5
    per_seed = {str(s): _val_stats_r5(run, a, s) for s in ARM_SEEDS_R5}
    flags = {str(s): _flag_for(run, a, s) for s in ARM_SEEDS_R5}
    best_seed = best_seed_r4(per_seed, floor)  # tie band + lowest seed
    out = {"stage": "S", "champion_val": champ, "stable_floor": floor,
           "table": {"per_seed": {s: {"directional": v[0], "stable": v[1],
                                      "epoch_floor_flagged": flags[s]}
                                  for s, v in per_seed.items()}},
           "best_seed": int(best_seed),
           "winner": dict(a, ensemble=False)}
    (Path(run) / "selection").mkdir(parents=True, exist_ok=True)
    _write_json(Path(run) / "selection" / "S.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): stage select - R/H/S with -2pp floor and tie band at every pick"
```

---

### Task 7: Calib delegate + stage_convert

**Files:**
- Modify: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: `stage_calib` from r2 (writes `{run}/selections/sequential_200_d0.json` + `uniform_random_200_d101.json` from weinstock train — never RT-CGM/PEDAP); `load_checkpoint` (head-aware, Task 1); `convert_float(model, sample, out)` / `convert_int8(model, sample, calib, out)`; `_data` from `experiments.accuracy_sweep`; `delta_windows`.
- Produces: `stage_calib(run_id)` (delegate); `stage_convert(run_id)` — converts the Stage-S winner arm's 5 seed checkpoints: `winner_seed{N}_float.tflite` + `winner_seed{N}_int8_{selection}.tflite` per calib selection, into `{run}/artifacts/`. Sample input `torch.zeros(1, 1, 12)`. Artifact NAMES are the contract Tasks 8–9 glob on.

- [ ] **Step 1: Write the failing test** (append — naming contract only; real conversion is covered by Task 1's LiteRT test and the execution run)

```python
def test_convert_artifact_naming_contract(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection" / "S.json").write_text(json.dumps(
        {"winner": r5.arm_r5("H_flat", "flat", "val"), "best_seed": 2}))
    (tmp_path / "selections").mkdir()
    for stem in ("sequential_200_d0", "uniform_random_200_d101"):
        (tmp_path / "selections" / f"{stem}.json").write_text(
            json.dumps({"indices": [0, 1]}))
    calls = []
    monkeypatch.setattr(r5, "_convert_seed",
                        lambda run, winner, seed: calls.append(seed))
    r5.stage_convert("sweep-r5-test")
    assert calls == [0, 1, 2, 3, 4]
    # candidate artifact paths the evaluate/analyze stages will look for
    fl, i8 = r5._candidate_arts(tmp_path)
    assert fl.name == "winner_seed2_float.tflite"
    assert i8.name == "winner_seed2_int8_uniform_random_200_d101.tflite"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_accuracy_sweep_r5.py -v -k naming_contract`
Expected: FAIL — no `stage_convert` / `_candidate_arts` / `_convert_seed`.

- [ ] **Step 3: Implement** (append)

```python
def stage_calib(run_id):
    from experiments.accuracy_sweep_r2 import stage_calib as calib_r2
    calib_r2(run_id)


def _candidate_arts(run):
    best = _read_sel(run, "S")["best_seed"]
    art = Path(run) / "artifacts"
    return (art / f"winner_seed{best}_float.tflite",
            art / f"winner_seed{best}_int8_uniform_random_200_d101.tflite")


def _convert_seed(run, winner, seed):
    import torch
    from conversion.common import load_checkpoint
    from conversion.convert import convert_float, convert_int8
    from experiments.accuracy_sweep import _data
    from training.deltas import delta_windows
    ds, _, _ = _data("weinstock", 12)  # lazy: tests stub _convert_seed whole
    art = Path(run) / "artifacts"
    art.mkdir(exist_ok=True)
    model = load_checkpoint(_ckpt_r5(run, winner, seed), 5)
    sample = torch.zeros(1, 1, 12, dtype=torch.float32)
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
        dws = delta_windows(ws, True).astype(np.float32)[:, None, :]
        calib = [torch.from_numpy(w[None]) for w in dws]
        timings = convert_int8(model, sample, calib, ipath)
        _write_json(ipath.with_suffix(".json"),
                    {"selection": sel_path.stem, **timings})
        print(f"converted {ipath.name}")


def stage_convert(run_id):
    run = _run_dir(run_id)
    winner = _read_sel(run, "S")["winner"]
    for seed in ARM_SEEDS_R5:
        _convert_seed(run, winner, seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): calib delegate and winner conversion"
```

---

### Task 8: RT-CGM eval windows, identity guard, stage_evaluate

**Files:**
- Modify: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: `RTCGM_CACHE`, `_candidate_arts` (Task 7); `_admitted_rtcgm` from `experiments.accuracy_sweep_r3`; `CHAMPION`, `_component_windows`, `_tflite_logits`, `protocol_is_committed` from `experiments.accuracy_sweep`; `_pedap_windows_r3` pattern (via the hours cache).
- Produces:
  - `_sha256(path) -> str` (full hex digest).
  - `identity_with_champion(run) -> dict` — `{"identical": bool, "candidate_float_sha256": …, "candidate_int8_sha256": …, "champion_float_sha256": …, "champion_int8_sha256": …}`; `identical` True only when BOTH float==float and int8==int8.
  - `_rtcgm_windows() -> (x float32 [n,12], y int64 [n], pids str [n])` — hours cache filtered to the admitted list; refuses if cache or admit list missing, or if the kept count is 0 or below half the cached rows (drift guard).
  - `stage_evaluate(run_id)` — refuses pre-freeze; identity → writes `{run}/eval/identity.json` and returns WITHOUT reading any labels; else evaluates primary RT-CGM (winner float + uniform INT8 + champion INT8 preds `.npy`, meta as `meta.npz` with `y`/`pids` — npz, not JSON: 12.7M rows), then secondary weinstock held_out/temporal (same three artifacts + champion, r3's `_component_windows`) and PEDAP (winner INT8 + champion INT8). Everything lands under gitignored `results/`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_identity_guard_requires_both_artifacts_equal(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection" / "S.json").write_text(json.dumps(
        {"winner": r5.arm_r5("H_gap", "gap", "val"), "best_seed": 0}))
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "winner_seed0_float.tflite").write_bytes(b"FLOAT")
    (art / "winner_seed0_int8_uniform_random_200_d101.tflite").write_bytes(b"INT8")
    champ_f, champ_i = tmp_path / "cf.tflite", tmp_path / "ci.tflite"
    champ_f.write_bytes(b"FLOAT")
    champ_i.write_bytes(b"INT8")
    monkeypatch.setattr(r5, "_champion_paths",
                        lambda: {"float": champ_f, "int8": champ_i})
    assert r5.identity_with_champion(tmp_path)["identical"] is True
    champ_i.write_bytes(b"DIFFERENT")  # one-of-two equal is NOT identity
    assert r5.identity_with_champion(tmp_path)["identical"] is False


def test_rtcgm_windows_filters_admitted_and_guards_drift(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    cache = tmp_path / "RT_CGM.npz"
    np.savez_compressed(cache,
                        w=np.zeros((6, 12), dtype=np.float32),
                        y=np.arange(6, dtype=np.int64) % 5,
                        pids=np.array(["a", "a", "b", "b", "c", "c"]),
                        hours=np.zeros(6))
    monkeypatch.setattr(r5, "RTCGM_CACHE", cache)
    monkeypatch.setattr(r5, "_admitted_rtcgm_r5", lambda: {"a", "b", "c"})
    x, y, pids = r5._rtcgm_windows()
    assert len(x) == 6 and x.shape[1] == 12
    # admit-list drift: keep count collapses below half the cache -> refuse
    monkeypatch.setattr(r5, "_admitted_rtcgm_r5", lambda: {"a"})
    with pytest.raises(SystemExit, match="REFUSED"):
        r5._rtcgm_windows()
    monkeypatch.setattr(r5, "_admitted_rtcgm_r5", lambda: {"zz"})
    with pytest.raises(SystemExit, match="REFUSED"):
        r5._rtcgm_windows()


def test_evaluate_refuses_without_committed_protocol(monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "protocol_is_committed", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        r5.stage_evaluate("sweep-r5-test")


def test_evaluate_identity_skips_the_read(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "protocol_is_committed", lambda p: True)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    monkeypatch.setattr(r5, "identity_with_champion", lambda run: {
        "identical": True, "candidate_float_sha256": "x",
        "candidate_int8_sha256": "y", "champion_float_sha256": "x",
        "champion_int8_sha256": "y"})

    def boom():
        raise AssertionError("labels were read despite identity")
    monkeypatch.setattr(r5, "_rtcgm_windows", boom)
    r5.stage_evaluate("sweep-r5-test")
    assert json.loads((tmp_path / "eval" / "identity.json").read_text())[
        "identical"] is True
    assert not (tmp_path / "eval" / "rtcgm").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy_sweep_r5.py -v -k "identity or rtcgm or evaluate"`
Expected: FAIL — missing `identity_with_champion`, `_rtcgm_windows`, `stage_evaluate`, `_champion_paths`, `_admitted_rtcgm_r5`.

- [ ] **Step 3: Implement** (append)

```python
def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _champion_paths():
    from experiments.accuracy_sweep import CHAMPION
    return {"float": CHAMPION["float"], "int8": CHAMPION["int8"]}


def identity_with_champion(run):
    """r4 lesson: a sha-identical candidate spends no fresh labels."""
    fl, i8 = _candidate_arts(run)
    champ = _champion_paths()
    out = {"candidate_float_sha256": _sha256(fl),
           "candidate_int8_sha256": _sha256(i8),
           "champion_float_sha256": _sha256(champ["float"]),
           "champion_int8_sha256": _sha256(champ["int8"])}
    out["identical"] = bool(
        out["candidate_float_sha256"] == out["champion_float_sha256"]
        and out["candidate_int8_sha256"] == out["champion_int8_sha256"])
    return out


def _admitted_rtcgm_r5():
    from experiments.accuracy_sweep_r3 import _admitted_rtcgm
    return _admitted_rtcgm()


def _rtcgm_windows():
    """Eligible RT-CGM eval windows: delta-encoded hours cache filtered to
    the cadence-admitted subjects. Loud on any drift."""
    if not RTCGM_CACHE.exists():
        raise SystemExit(f"REFUSED: RT-CGM hours cache missing: {RTCGM_CACHE}")
    z = np.load(RTCGM_CACHE)
    admitted = _admitted_rtcgm_r5()
    keep = np.isin(z["pids"], np.asarray(sorted(admitted)))
    n_total, n_keep = len(z["pids"]), int(keep.sum())
    if n_keep == 0 or n_keep < n_total // 2:
        raise SystemExit(f"REFUSED: admit-list drift - {n_keep}/{n_total} "
                         "cached RT-CGM windows admitted; audit before eval")
    return (z["w"][keep].astype(np.float32),
            z["y"][keep].astype(np.int64), z["pids"][keep])


def stage_evaluate(run_id):
    from experiments.accuracy_sweep import (CHAMPION, _component_windows,
                                            _tflite_logits)
    if not protocol_is_committed(PROTOCOL_R5):
        raise SystemExit("REFUSED: commit paper/results/calibration-coverage/"
                         "protocol_sweep_r5.json (clean) before evaluate")
    run = _run_dir(run_id)
    ident = identity_with_champion(run)
    (Path(run) / "eval").mkdir(parents=True, exist_ok=True)
    if ident["identical"]:
        _write_json(Path(run) / "eval" / "identity.json", ident)
        print("identity outcome: candidate == shipped champion; "
              "confirmatory read skipped, no labels spent")
        return
    fl, i8 = _candidate_arts(run)
    # --- primary: RT-CGM zero-shot (first and only label read) -------------
    base = Path(run) / "eval" / "rtcgm"
    base.mkdir(parents=True, exist_ok=True)
    if not (base / "meta.npz").exists():
        x, y, pids = _rtcgm_windows()
        np.savez_compressed(base / "meta.npz", y=y, pids=pids)
        np.save(base / "_x.npy", x)  # transient input cache, gitignored
    x = np.load(base / "_x.npy")
    for name, path in (("winner_float", fl), ("winner_int8", i8),
                       ("champion_int8", CHAMPION["int8"])):
        p = base / f"{name}_preds.npy"
        if not p.exists():
            np.save(p, _tflite_logits(path, x).argmax(axis=1))
            print(f"rtcgm/{name} evaluated")
    # --- secondary: weinstock held-out/temporal (disclosed reused) ----------
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        cbase = Path(run) / "eval" / comp
        cbase.mkdir(parents=True, exist_ok=True)
        windows, meta = _component_windows(12, split)
        if not (cbase / "meta.json").exists():
            _write_json(cbase / "meta.json", meta)
        for name, path in (("winner_float", fl), ("winner_int8", i8),
                           ("champion_int8", CHAMPION["int8"])):
            p = cbase / f"{name}_preds.npy"
            if not p.exists():
                np.save(p, _tflite_logits(path, windows).argmax(axis=1))
                print(f"{comp}/{name} evaluated")
    # --- secondary: PEDAP zero-shot (descriptive) ---------------------------
    pbase = Path(run) / "eval" / "pedap"
    pbase.mkdir(parents=True, exist_ok=True)
    from experiments.accuracy_sweep_r3 import _diadata_windows_r3
    from training.mixtures import HOLDOUT_COHORT
    w, y, pids, _hours = _diadata_windows_r3(HOLDOUT_COHORT)
    if not (pbase / "meta.npz").exists():
        np.savez_compressed(pbase / "meta.npz", y=y, pids=pids)
    for name, path in (("winner_int8", i8), ("champion_int8", CHAMPION["int8"])):
        p = pbase / f"{name}_preds.npy"
        if not p.exists():
            np.save(p, _tflite_logits(path, w).argmax(axis=1))
            print(f"pedap/{name} evaluated")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): RT-CGM zero-shot evaluate with identity guard and drift refusals"
```

---

### Task 9: Gates, protocol writer, stage_analyze (identity + full flows)

**Files:**
- Modify: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: `paired_cluster_bootstrap`, `confusion_counts`, `macro_f1` from `experiments.analyze_calibration_coverage`; `BOOT_SEED_R5`; eval artifacts from Task 8; `_sha256`, `_champion_paths`, `identity_with_champion`.
- Produces:
  - `gate_results_r5(y, pids, cand_int8, cand_float, champ_int8, n_boot=2000) -> dict` — r3's shape; bootstrap seeded `BOOT_SEED_R5`; `gate1_beats_champion` requires `delta > 0 AND ci_low is not None AND ci_low > -0.005` (None CI → False); `gate2_stable_floor` = cand stable ≥ champ stable − 0.02; `gate3` = float − int8 ≤ 0.005 (directional).
  - `stage_protocol(run_id)` — writes `PROTOCOL_R5` (committed by hand afterwards): round, spec path, boot seed + n_boot, selection trail (R/H/S), smoke record, champion sha256s FROM LIVE ASSETS, R_base alias sha256s, primary endpoint text, gates text, `DISCLOSURE_RTCGM`, selection-rule changes vs r3, identity-guard rule, calibration provenance. Aggregates only — no participants/keys.
  - `stage_analyze(run_id, n_boot=2000)` — identity flow (eval/identity.json present → publishes selection trail + identity shas, `ship.all_gates_pass = False`, outcome text, no label-derived numbers) and full flow (rtcgm primary with gates; held_out/temporal descriptive with `gate_results_r5` output under `"descriptive_non_gating": true`; pedap descriptive; `ship.all_gates_pass` from rtcgm gates only). Writes `{run}/analysis/sweep_r5_aggregate.json`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_gates_r5_boot_seed_and_none_ci(monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    captured = {}

    def fake_boot(y, pids, pairs, metric, n_boot, seed):
        captured["seed"] = seed
        return {"delta": 0.01, "ci95": None, "undefined_fraction": 0.2,
                "n_boot": n_boot, "seed": seed}
    import experiments.analyze_calibration_coverage as acc
    monkeypatch.setattr(acc, "paired_cluster_bootstrap", fake_boot)
    y = np.array([0, 1, 2, 3, 4] * 4)
    pids = np.array(["a", "b"] * 10)
    g = r5.gate_results_r5(y, pids, y.copy(), y.copy(), y.copy(), n_boot=7)
    assert captured["seed"] == r5.BOOT_SEED_R5 == 20261015
    assert g["gate1_beats_champion"] is False  # undefined CI never passes


def test_gates_r5_pass_case(monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    import experiments.analyze_calibration_coverage as acc
    monkeypatch.setattr(acc, "paired_cluster_bootstrap",
                        lambda *a, **k: {"delta": 0.008,
                                         "ci95": [0.001, 0.015],
                                         "undefined_fraction": 0.0,
                                         "n_boot": k.get("n_boot"),
                                         "seed": k.get("seed")})
    y = np.array([0, 1, 2, 3, 4] * 4)
    pids = np.array(["a", "b"] * 10)
    g = r5.gate_results_r5(y, pids, y.copy(), y.copy(), y.copy())
    assert g["gate1_beats_champion"] is True
    assert g["gate2_stable_floor"] is True
    assert g["gate3_int8_within_half_pp_of_float"] is True


def test_analyze_identity_flow(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    (tmp_path / "selection").mkdir()
    for s in ("R", "H"):
        (tmp_path / "selection" / f"{s}.json").write_text(json.dumps(
            {"stage": s, "winner": r5.arm_r5("R_base", "gap", "base")}))
    (tmp_path / "selection" / "S.json").write_text(json.dumps(
        {"stage": "S", "winner": r5.arm_r5("R_base", "gap", "base"),
         "best_seed": 1}))
    (tmp_path / "smoke.json").write_text(json.dumps({"frozen_batch": 512}))
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "identity.json").write_text(json.dumps(
        {"identical": True, "candidate_int8_sha256": "y"}))
    out = r5.stage_analyze("sweep-r5-test")
    assert out["ship"]["all_gates_pass"] is False
    assert "identity" in out["ship"]["outcome"]
    assert "components" not in out
    assert (tmp_path / "analysis" / "sweep_r5_aggregate.json").exists()


def test_protocol_has_no_row_level_fields(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    monkeypatch.setattr(r5, "PROTOCOL_R5", tmp_path / "protocol_sweep_r5.json")
    monkeypatch.setattr(r5, "_champion_paths", lambda: {
        "float": tmp_path / "cf", "int8": tmp_path / "ci"})
    (tmp_path / "cf").write_bytes(b"F")
    (tmp_path / "ci").write_bytes(b"I")
    monkeypatch.setattr(r5, "_ckpt_r5",
                        lambda run, a, s: tmp_path / "cf")  # alias sha stub
    (tmp_path / "selection").mkdir()
    for s, extra in (("R", {}), ("H", {}), ("S", {"best_seed": 2})):
        (tmp_path / "selection" / f"{s}.json").write_text(json.dumps(
            {"stage": s, "winner": r5.arm_r5("H_gap", "gap", "val"), **extra}))
    (tmp_path / "smoke.json").write_text(json.dumps({"frozen_batch": 512}))
    out = r5.stage_protocol("sweep-r5-test")
    blob = (tmp_path / "protocol_sweep_r5.json").read_text()
    for pat in ('"participants"', '"keys"', '"pids"', '"anchor_times"'):
        assert pat not in blob
    assert out["boot_seed"] == 20261015
    assert "disclosure" in out and "D_rtcgm" in out["disclosure"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy_sweep_r5.py -v -k "gates_r5 or analyze_identity or protocol_has"`
Expected: FAIL — missing `gate_results_r5`, `stage_analyze`, `stage_protocol`.

- [ ] **Step 3: Implement** (append)

```python
def gate_results_r5(y, pids, cand_int8, cand_float, champ_int8, n_boot=2000):
    import experiments.analyze_calibration_coverage as acc
    contrast = acc.paired_cluster_bootstrap(
        y, pids, [(cand_int8, champ_int8)], metric=directional_recall,
        n_boot=n_boot, seed=BOOT_SEED_R5)
    ci_low = contrast["ci95"][0] if contrast["ci95"] else None
    d_float = directional_recall(acc.confusion_counts(y, cand_float))
    d_int8 = directional_recall(acc.confusion_counts(y, cand_int8))
    s_cand = stable_recall(acc.confusion_counts(y, cand_int8))
    s_champ = stable_recall(acc.confusion_counts(y, champ_int8))
    return {
        "contrast_int8_vs_champion": contrast,
        "candidate_float_directional": d_float,
        "candidate_int8_directional": d_int8,
        "champion_int8_directional": directional_recall(
            acc.confusion_counts(y, champ_int8)),
        "candidate_int8_stable": s_cand,
        "champion_int8_stable": s_champ,
        "gate1_beats_champion": bool(contrast["delta"] is not None
                                     and contrast["delta"] > 0
                                     and ci_low is not None
                                     and ci_low > -0.005),
        "gate2_stable_floor": bool(s_cand >= s_champ - FLOOR_MARGIN_R5),
        "gate3_int8_within_half_pp_of_float": bool(d_float - d_int8 <= 0.005),
    }


def stage_protocol(run_id):
    """Draft the pre-registration; the user commits it (clean) before any
    RT-CGM label read - stage_evaluate enforces the commit."""
    run = _run_dir(run_id)
    champ = _champion_paths()
    base_arm = arm_r5("R_base", "gap", "base")
    out = {
        "round": "r5",
        "spec": "docs/superpowers/specs/2026-10-14-accuracy-sweep-r5-design.md",
        "boot_seed": BOOT_SEED_R5, "n_boot": 2000,
        "champion_sha256": {k: _sha256(v) for k, v in champ.items()},
        "r_base_alias_sha256": {str(s): _sha256(_ckpt_r5(run, base_arm, s))
                                for s in ARM_SEEDS_R5},
        "selection_trail": {s: _read_sel(run, s) for s in ("R", "H", "S")},
        "smoke": json.loads((Path(run) / "smoke.json").read_text()),
        "primary_endpoint": (
            "RT-CGM zero-shot directional recall: winner uniform INT8 vs "
            "shipped champion INT8, paired on identical eligible windows, "
            "participant-cluster bootstrap, 2000 draws"),
        "gates": [
            "gate1: directional delta > 0 AND 95% CI lower bound > -0.5pp",
            "gate2: winner INT8 stable recall >= champion INT8 stable - 2.0pp",
            "gate3: winner INT8 within 0.5pp of its own float (directional)"],
        "secondary_disclosed_reused": (
            "weinstock held-out (3 prior confirmatory reads) and temporal: "
            "descriptive only; macro-F1 all splits; PEDAP zero-shot "
            "descriptive. None gate."),
        "calibration": ("sequential_200_d0 + uniform_random_200_d101 from "
                        "weinstock train only; uniform INT8 is the ship "
                        "candidate"),
        "disclosure": DISCLOSURE_RTCGM,
        "selection_rule_changes_vs_r3": [
            "selection floor aligned to Gate 2: champion val stable - 2pp "
            "(r3 used -5pp at selection)",
            "tie band 0.1pp -> cost_key -> list order applied at every "
            "selection including Stage S and the kept-epoch pick "
            "(r3 Stage S used strict argmax)"],
        "identity_guard": ("candidate float AND uniform INT8 sha256-identical "
                           "to the shipped pair -> confirmatory read skipped, "
                           "identity outcome published (r4 lesson)"),
        "prediction": ("flatten head wins val selection and transfers: "
                       "positive directional delta on RT-CGM with CI lower "
                       "bound above -0.5pp, stable within 2pp"),
    }
    _write_json(PROTOCOL_R5, out)
    print("wrote", PROTOCOL_R5, "- commit it (clean) before stage evaluate")
    return out


def _analyze_identity(run_id, run):
    run = Path(run)
    out = {"protocol": "paper/results/calibration-coverage/protocol_sweep_r5.json",
           "run_id": run_id,
           "selection_trail": {s: _read_sel(run, s) for s in ("R", "H", "S")
                               if (run / "selection" / f"{s}.json").exists()},
           "identity_outcome": json.loads(
               (run / "eval" / "identity.json").read_text())}
    if (run / "smoke.json").exists():
        out["smoke"] = json.loads((run / "smoke.json").read_text())
    out["ship"] = {"all_gates_pass": False,
                   "outcome": "identity outcome: candidate converts "
                              "sha256-identical to the shipped champion; "
                              "no confirmatory read performed, RT-CGM "
                              "labels unspent"}
    out["deviations"] = list(DEVIATIONS_R5)
    (run / "analysis").mkdir(exist_ok=True)
    _write_json(run / "analysis" / "sweep_r5_aggregate.json", out)
    print("wrote", run / "analysis" / "sweep_r5_aggregate.json", "(identity flow)")
    return out


def stage_analyze(run_id, n_boot=2000):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    from training.labeling import FIVE_CLASSES
    run = _run_dir(run_id)
    if (Path(run) / "eval" / "identity.json").exists():
        return _analyze_identity(run_id, run)
    best_seed = _read_sel(run, "S")["best_seed"]
    out = {"protocol": "paper/results/calibration-coverage/protocol_sweep_r5.json",
           "run_id": run_id,
           "selection_trail": {s: _read_sel(run, s) for s in ("R", "H", "S")},
           "smoke": json.loads((Path(run) / "smoke.json").read_text()),
           "reused_tensor": _tensor_path_r5().name,
           "components": {}}
    # primary: RT-CGM (gates live here and only here)
    base = Path(run) / "eval" / "rtcgm"
    meta = np.load(base / "meta.npz")
    y, pids = meta["y"], meta["pids"]
    cand_int8 = np.load(base / "winner_int8_preds.npy")
    cand_float = np.load(base / "winner_float_preds.npy")
    champ = np.load(base / "champion_int8_preds.npy")
    out["components"]["rtcgm_primary"] = {
        "windows": int(len(y)),
        "participants": int(len(np.unique(pids))),
        "pairing_rule": "identical eligible windows by construction",
        "macro_f1_candidate_int8": macro_f1(confusion_counts(y, cand_int8)),
        "macro_f1_champion_int8": macro_f1(confusion_counts(y, champ)),
        "gates": gate_results_r5(y, pids, cand_int8, cand_float, champ,
                                 n_boot=n_boot),
        "disclosure": DISCLOSURE_RTCGM,
    }
    # secondary: weinstock held-out/temporal, descriptive (non-gating)
    for comp in ("held_out", "temporal"):
        cbase = Path(run) / "eval" / comp
        cmeta = json.loads((cbase / "meta.json").read_text())
        cy = np.array(cmeta["labels"])
        cpids = np.array(cmeta["participants"])
        ci8 = np.load(cbase / "winner_int8_preds.npy")
        cfl = np.load(cbase / "winner_float_preds.npy")
        cch = np.load(cbase / "champion_int8_preds.npy")
        out["components"][comp] = {
            "descriptive_non_gating": True,
            "windows": int(len(cy)),
            "macro_f1_candidate_int8": macro_f1(confusion_counts(cy, ci8)),
            "macro_f1_champion_int8": macro_f1(confusion_counts(cy, cch)),
            "contrast": gate_results_r5(cy, cpids, ci8, cfl, cch,
                                        n_boot=n_boot),
        }
    # secondary: PEDAP descriptive
    pbase = Path(run) / "eval" / "pedap"
    pmeta = np.load(pbase / "meta.npz")
    py = pmeta["y"]
    ped = {}
    for name in ("winner_int8", "champion_int8"):
        cm = confusion_counts(py, np.load(pbase / f"{name}_preds.npy"))
        ped[name] = {"directional": directional_recall(cm),
                     "stable": stable_recall(cm),
                     "per_class": {c: float(cm[i, i] / cm[i].sum())
                                   if cm[i].sum() else 0.0
                                   for i, c in enumerate(FIVE_CLASSES)}}
    out["pedap_zero_shot"] = {"windows": int(len(py)), **ped,
                              "note": "descriptive; no gates; cohort never "
                                      "trained on"}
    g = out["components"]["rtcgm_primary"]["gates"]
    out["ship"] = {
        "candidate": f"seed {best_seed}",
        "all_gates_pass": bool(g["gate1_beats_champion"]
                               and g["gate2_stable_floor"]
                               and g["gate3_int8_within_half_pp_of_float"]),
    }
    out["deviations"] = list(DEVIATIONS_R5)
    (Path(run) / "analysis").mkdir(exist_ok=True)
    _write_json(Path(run) / "analysis" / "sweep_r5_aggregate.json", out)
    print("wrote", Path(run) / "analysis" / "sweep_r5_aggregate.json")
    return out
```

(Note for `stage_analyze` identity-flow test: `_analyze_identity` must not call `_tensor_path_r5()` — only the full flow records `reused_tensor`. The identity test's tmp run has no tensors; if the implementation accidentally touches `_tensor_path_r5` there, the test fails loudly — that is intended.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accuracy_sweep_r5.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): gates on RT-CGM, protocol writer, analyze with identity flow"
```

---

### Task 10: CLI main + published-JSON leak test

**Files:**
- Modify: `experiments/accuracy_sweep_r5.py`
- Test: `tests/test_accuracy_sweep_r5.py`

**Interfaces:**
- Consumes: every stage function above.
- Produces: `python -m experiments.accuracy_sweep_r5 {smoke,train,select,calib,convert,protocol,evaluate,analyze} [--run-id sweep-r5] [--stage-name {R,H,S}]`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_main_stage_choices():
    import experiments.accuracy_sweep_r5 as r5
    import inspect
    src = inspect.getsource(r5.main)
    for stage in ("smoke", "train", "select", "calib", "convert",
                  "protocol", "evaluate", "analyze"):
        assert f'"{stage}"' in src


def test_published_r5_jsons_leak_free():
    # row-level identifiers never leave gitignored results/
    from pathlib import Path as _P
    root = (_P(__file__).resolve().parents[1] / "paper" / "results"
            / "calibration-coverage")
    names = ("protocol_sweep_r5.json", "accuracy_sweep_r5.json")
    published = [n for n in names if (root / n).exists()]
    if not published:
        pytest.skip("r5 not published yet")
    for name in published:
        blob = (root / name).read_text()
        for pat in ('"participants": [', '"keys"', '"pids"', '"row_ids"',
                    '"anchor_times"', '"labels": ['):
            assert pat not in blob, (name, pat)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy_sweep_r5.py -v -k "main_stage or leak_free"`
Expected: `test_main_stage_choices` FAIL (no `main`); `test_published_r5_jsons_leak_free` SKIP.

- [ ] **Step 3: Implement** (append)

```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "train", "select", "calib",
                                          "convert", "protocol", "evaluate",
                                          "analyze"])
    parser.add_argument("--run-id", default="sweep-r5")
    parser.add_argument("--stage-name", choices=["R", "H", "S"])
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
    elif args.stage == "protocol":
        stage_protocol(args.run_id)
    elif args.stage == "evaluate":
        stage_evaluate(args.run_id)
    else:
        stage_analyze(args.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the FULL suite**

Run: `pytest -q > .../scratchpad/t10_suite.log 2>&1` then read the whole log.
Expected: all pass (one skip for the unpublished-r5 leak test, LiteRT skips as environment dictates).

- [ ] **Step 5: Commit**

```bash
git add experiments/accuracy_sweep_r5.py tests/test_accuracy_sweep_r5.py
git commit -m "feat(r5): CLI entry point and published-JSON leak walk"
```

---

## Execution tasks (the sweep itself — operational, not TDD)

Every long run: activate the training venv, launch via `Bash` with `run_in_background`, redirect to a scratchpad log (`python -m experiments.accuracy_sweep_r5 <stage> ... > <scratchpad>/r5_<stage>.log 2>&1`), NEVER pipe through `tail`, and read the full log afterwards (tracebacks can appear mid-file). Conversion stages run in the conversion env (`conversion/requirements.txt`).

### Task 11: Smoke + Stage R (train + select)

- [ ] `python -m experiments.accuracy_sweep_r5 smoke --run-id sweep-r5` — Expected: `smoke.json` in the run dir with `frozen_batch` and `mps_seed_reproducible: true`. If reproducibility fails, STOP: the R_base reuse justification collapses — ledger and ask.
- [ ] `... train --stage-name R` — trains R_val seeds 0–4 only (R_base refuses to train; its alias files must exist). Expected: 5 checkpoints `D5_none_cnn_gap_val_seed{0-4}.pt` with sidecars carrying `kept_epoch` + `per_epoch`; epoch dirs with 20 `epoch*.pt` each.
- [ ] `... select --stage-name R` — Expected: `selection/R.json` with floor = champion val stable − 0.02, both arms' 5-seed tables, a winner.
- [ ] Record stage results in the ledger (means, floor, winner, any flagged seeds).

### Task 12: Stage H (train + select) + Stage S

- [ ] `... train --stage-name H` — trains H_flat seeds 0–4 (H_gap reuses the R winner's checkpoints). Expected: 5 `D5_none_cnn_flat_{recipe}_seed{0-4}.pt`.
- [ ] `... select --stage-name H` then `... select --stage-name S`. Expected: `selection/H.json`, `selection/S.json` with `best_seed`; tie-band behavior visible in the printed summary if any band occurs — note it in the ledger.

### Task 13: Calib + convert + protocol freeze

- [ ] `... calib --run-id sweep-r5` (training env) — Expected: `selections/sequential_200_d0.json` + `uniform_random_200_d101.json`.
- [ ] `... convert --run-id sweep-r5` (conversion env) — Expected: 15 artifacts (5 seeds × float + 2 INT8 variants).
- [ ] `... protocol --run-id sweep-r5` — Expected: `paper/results/calibration-coverage/protocol_sweep_r5.json` written with champion shas, alias shas, trail, disclosure, prediction.
- [ ] Review the protocol content, then commit it CLEAN as its own commit: `git add paper/results/calibration-coverage/protocol_sweep_r5.json && git commit -m "docs(protocol): freeze accuracy sweep r5 pre-registration"`. No RT-CGM label has been read up to this point.

### Task 14: Evaluate + analyze

- [ ] `... evaluate --run-id sweep-r5` (conversion env; this is the one RT-CGM label read, ~12.7M windows × 3 artifacts — budget hours; background + log). If the identity guard fires, Expected: `eval/identity.json`, no label read; proceed to analyze's identity flow.
- [ ] `... analyze --run-id sweep-r5` (2,000-draw bootstrap over 12.7M windows is heavy — background + log). Expected: `analysis/sweep_r5_aggregate.json` with rtcgm_primary gates (or identity outcome) and `ship.all_gates_pass`.
- [ ] Report the gate verdicts to the user BEFORE any ship action.

### Task 15: Publish + ship-or-note

- [ ] Copy the aggregate to `paper/results/calibration-coverage/accuracy_sweep_r5.json`; append any accrued `DEVIATIONS_R5` entries to the module tuple first and regenerate so the published file carries them; run `pytest tests/test_accuracy_sweep_r5.py -k leak_free -v` — Expected: PASS (no longer skipped).
- [ ] Paper: add the r5 subsection + `tab:sweepfive` (mirror `tab:sweepthree`'s format: stage means, confirmatory deltas in recall points ×100 with CIs, identity note if applicable; `\resizebox{\columnwidth}{!}` on any wide table). Rebuild: `cd paper && rm -f main.aux main.bbl main.blg main.log main.out && tectonic --keep-intermediates main.tex`. Expected: clean build; page count noted (17-page state, no cap).
- [ ] **If `all_gates_pass`:** full r2-style ship — winner float + uniform INT8 replace the bundled pair in `android/app/src/main/assets/`, goldens regenerated and re-pinned, shas updated wherever pinned, Android JVM + iOS simulator suites green, device verification on the iPhone 18 Pro Max, MODELS.md updated. STOP for the user before the asset swap (ship decision is theirs).
- [ ] **If gates fail or identity:** MODELS.md note (r3's "did not clear the ship gates" wording, or the r4 identity wording), bundled pair retained.
- [ ] Full suite green; ask the user before any push (memory rule).

---

## Self-review notes

- Spec coverage: §1–§7 all mapped (recipe→T2/T5, head→T1, floor/tie→T4/T6, RT-CGM primary+gates→T8/T9, identity guard→T8, protocol→T9/T13, ship→T15, non-goals enforced by the D5-only refusal in T5 and the absence of any other arm).
- Spec's enumerated tests all present: head-aware load + compat (T1), flatten shape + param bound (T1/T4), epoch-select floor (T4/T5), tie band incl. Stage S (T4/T6), seed single source (T9 gates test), evaluate refusal (T8), RT-CGM never trains (T3/T5), identity guard (T8).
- Type consistency: arm dict keys fixed in T3 and consumed unchanged; `_ckpt_r5(run, a, seed)` signature uniform; `(directional, stable)` tuples everywhere; preds `.npy` names fixed in T8 and read in T9.
- Known cost accepted (ponytail): the naive 2,000-draw bootstrap over 12.7M rows (~tens of minutes to ~1h, one-time); a per-participant sufficient-statistics fast path was considered and skipped — add only if the analyze run proves intolerable, as a ledgered ruling.
