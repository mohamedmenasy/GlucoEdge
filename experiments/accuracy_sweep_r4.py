"""Accuracy sweep r4: noise-ceiling gate + objective heads + robustness +
weight averaging (spec 2026-10-13-accuracy-sweep-r4-design.md).

Selection stays on weinstock val; Experiment 0 gates training; champion is
the bundled (shipped r2) pair read live from android assets."""
import json
from pathlib import Path

import numpy as np

from experiments.accuracy_sweep import protocol_is_committed
from experiments.accuracy_sweep_r2 import (SAMPLE_SEED, TensorView,
                                           champion_val_stats,
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

# Disclosed with the published aggregate; the frozen protocol is not edited.
DEVIATIONS_R4 = ()


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
    np.savez_compressed(cache, levels=out[0], horizons=out[1], y=out[2],
                        pids=out[3])
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
        chosen, _ = sample_subjects(counts, int(CAP * sizes[c] / total),
                                    sample_seed)
        mask = np.isin(pids, np.asarray(chosen))
        lv.append(levels[mask])
        hz.append(horizons[mask])
        ys.append(y[mask])
    levels = np.concatenate(lv)
    horizons = np.concatenate(hz)
    y = np.concatenate(ys)
    r = (horizons[:, -1] - levels[:, -1]) / 15.0
    lsq = lsq_slope(np.concatenate([levels[:, -1:], horizons], axis=1))
    if not (np.isfinite(r).all() and np.isfinite(lsq).all()):
        raise SystemExit("REFUSED: non-finite rate targets in arm tensors")
    x = delta_windows(levels, True).astype(np.float32)[:, None, :]
    np.savez_compressed(path, x=x, levels=levels, y=y,
                        r=r.astype(np.float32), lsq=lsq.astype(np.float32))
    return path


def _train_targets(a, z):
    """Arm's labels/head -> training target array from a tensor npz dict."""
    rates = z["lsq"] if a["labels"] == "lsq" else z["r"]
    if a["head"] == "reg":
        return np.asarray(rates, dtype=np.float32)
    if a["labels"] == "lsq":
        from training.labeling import FIVE_CLASSES, classify_rate
        idx = {c: i for i, c in enumerate(FIVE_CLASSES)}
        return np.array([idx[classify_rate(float(v))] for v in rates],
                        dtype=np.int64)
    return np.asarray(z["y"], dtype=np.int64)


class R4View(TensorView):
    """Rows already (1, 12); int targets."""

    def __getitem__(self, k):
        import torch
        return torch.from_numpy(self.x[k]), int(self.y[k])


class RateView:
    def __init__(self, x, r):
        self.x = x
        self.y = np.asarray(r, dtype=np.float32)

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
