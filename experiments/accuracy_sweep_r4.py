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
        noise[f"sigma_{sigma:g}"] = exp0_agreement(
            y, _classify_rates(rate))["ceiling_directional"]
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


def _ckpt_r4(run, a, seed):
    return Path(run) / "checkpoints" / f"{a['name']}_seed{seed}.pt"


def _val_view_r4():
    from training.deltas import delta_windows
    levels, horizons, y = _val_label_views()
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
    from conversion.common import load_checkpoint
    from experiments.analyze_calibration_coverage import confusion_counts
    from training.heads import decode_outputs, head_outputs
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
                                 "its checkpoints (spec section 3), do not train")
            if z is None:
                z = dict(np.load(build_arm_tensors_r4(run, a)))
            t0 = _t.time()
            _train_one_r4(run, a, seed, z, batch)
            print(f"{a['name']} seed {seed} done in {_t.time()-t0:.0f}s")


COST_ORDER = ("ce", "wtemp", "reg", "corn")  # cheaper decode first (tie rule)


def pick_winner_r4(arms, means):
    best = max(means.values())
    names = [x["name"] for x in arms]
    band = [a for a in arms if a["name"] in means
            and best - means[a["name"]] <= 0.001]
    band.sort(key=lambda a: (COST_ORDER.index(a["head"]),
                             names.index(a["name"])))
    return band[0]


def best_seed_r4(per_seed, floor):
    ok = {s: v for s, v in per_seed.items() if v[1] >= floor}
    if not ok:
        raise SystemExit("REFUSED: every seed failed the stable floor")
    best = max(v[0] for v in ok.values())
    band = sorted(int(s) for s, v in ok.items() if best - v[0] <= 0.001)
    return band[0]


def _select(run, stage, arms):
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
    if stage == "W":
        return stage_select_w(run)
    raise SystemExit(f"unknown stage {stage}")
