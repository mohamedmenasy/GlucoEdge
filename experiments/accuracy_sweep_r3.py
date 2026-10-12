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

PROTOCOL_R3 = (GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage"
               / "protocol_sweep_r3.json")
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


class TensorView3(TensorView):
    """Rows are already [C, L]; no unsqueeze."""

    def __getitem__(self, k):
        import torch
        return torch.from_numpy(self.x[k]), int(self.y[k])


_HOUR_CACHE = {}


def _diadata_windows_r3(cohort):
    """Like r2's _diadata_windows but also returns anchor hours; cached npz."""
    if cohort in _HOUR_CACHE:
        return _HOUR_CACHE[cohort]
    safe = cohort.replace("-", "_")
    cache = GLUCOEDGE_ROOT / "results" / "diadata_cache" / "hours" / f"{safe}.npz"
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
                         "the rtcgm audit before building D5R tensors")
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
    return TensorView3(z["x"], z["y"])


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
    return TensorView3(_with_features(w, hours, features), y)
