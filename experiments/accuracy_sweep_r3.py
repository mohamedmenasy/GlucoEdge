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
        channels = model.fc.in_features
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


_VAL_CACHE = {}


def _val_for(features):
    if features not in _VAL_CACHE:
        _VAL_CACHE[features] = _val_view_r3(features)
    return _VAL_CACHE[features]


def _val_stats_r3(run, a, seed):
    """(directional, stable) recall on the eligible weinstock val set."""
    import torch
    from conversion.common import load_checkpoint
    from experiments.analyze_calibration_coverage import confusion_counts
    val_v = _val_for(a["features"])
    model = load_checkpoint(_ckpt(run, a, seed), 5)
    preds = []
    with torch.no_grad():
        for i in range(0, len(val_v), 8192):
            preds.append(model(torch.from_numpy(val_v.x[i:i + 8192]))
                         .numpy().argmax(axis=1))
    cm = confusion_counts(val_v.y, np.concatenate(preds))
    return directional_recall(cm), stable_recall(cm)


def _champion_val(run):
    from experiments.accuracy_sweep_r2 import champion_val_stats
    return champion_val_stats(run)


def _read_sel(run, stage):
    return json.loads((Path(run) / "selection" / f"{stage}.json").read_text())


def _arms_for(run, stage):
    if stage == "D":
        return stage_d_arms()
    if stage == "F":
        return stage_f_arms(_read_sel(run, "D")["winner"])
    if stage == "A":
        return stage_a_arms(_read_sel(run, "F")["winner"])
    raise SystemExit(f"unknown stage {stage}")


def stage_train(run_id, stage):
    import time as _t
    from experiments.accuracy_sweep_r2 import _frozen_batch
    run = _run_dir(run_id)
    batch = _frozen_batch(run)
    for a in _arms_for(run, stage):
        train_v = None
        for seed in ARM_SEEDS_R3:
            ckpt = _ckpt(run, a, seed)
            if ckpt.exists():
                print(f"skip {ckpt.name}: exists")
                continue
            if train_v is None:
                train_v = _load_view_r3(build_arm_tensors_r3(run, a),
                                        a["in_channels"])
            t0 = _t.time()
            _train_one_r3(run, a, seed, train_v, _val_for(a["features"]), batch)
            print(f"{a['name']} seed {seed} done in {_t.time()-t0:.0f}s")


def _select(run, stage, arms):
    from experiments.accuracy_sweep import pick_winner
    champ = _champion_val(run)
    floor = champ["stable"] - 0.05
    table, means = {}, {}
    for a in arms:
        per_seed = {str(s): _val_stats_r3(run, a, s) for s in ARM_SEEDS_R3}
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
    sel.mkdir(parents=True, exist_ok=True)
    _write_json(sel / f"{stage}.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
    return out


def stage_select(run_id, stage):
    run = _run_dir(run_id)
    if stage in ("D", "F", "A"):
        return _select(run, stage, _arms_for(run, stage))
    if stage != "S":
        raise SystemExit(f"unknown stage {stage}")
    a = _read_sel(run, "A")["winner"]
    champ = _champion_val(run)
    floor = champ["stable"] - 0.05
    per_seed = {str(s): _val_stats_r3(run, a, s) for s in ARM_SEEDS_R3}
    ok = {s: v for s, v in per_seed.items() if v[1] >= floor}
    if not ok:
        raise SystemExit("REFUSED: every seed failed the stable floor")
    best_seed = max(ok, key=lambda s: ok[s][0])
    out = {"stage": "S",
           "table": {"per_seed": {s: {"directional": v[0], "stable": v[1]}
                                  for s, v in per_seed.items()}},
           "best_seed": int(best_seed),
           "winner": dict(a, ensemble=False)}
    (Path(run) / "selection").mkdir(parents=True, exist_ok=True)
    _write_json(Path(run) / "selection" / "S.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
    return out


def gate_results_r3(y, pids, cand_int8, cand_float, champ_int8, n_boot=2000):
    from experiments.analyze_calibration_coverage import (
        confusion_counts, paired_cluster_bootstrap)
    contrast = paired_cluster_bootstrap(y, pids, [(cand_int8, champ_int8)],
                                        metric=directional_recall, n_boot=n_boot,
                                        seed=BOOT_SEED_R3)
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
