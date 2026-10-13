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
        # 0-dim float32 tensor: default collate must NOT widen to float64 (MPS)
        return torch.from_numpy(self.x[k]), torch.tensor(self.y[k])


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
        if self.float_targets:
            return torch.from_numpy(x), torch.tensor(np.float32(self.y[k]))
        return torch.from_numpy(x), int(self.y[k])


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


def _winner_ckpts(run):
    """W winner -> [(label, checkpoint path)] making up the artifact set."""
    w = _read_sel(run, "W")["winner"]
    run = Path(run)
    if w["cell"] == "W_soup":
        return [("soup", run / "checkpoints" / "W_soup.pt")]
    if w["cell"] == "W_swa":
        return [(f"seed{s}", run / "checkpoints" / f"W_swa_seed{s}.pt")
                for s in ARM_SEEDS_R4]
    return [(f"seed{s}", _ckpt_r4(run, w, s)) for s in ARM_SEEDS_R4]


def _candidate_label(run):
    w = _read_sel(run, "W")["winner"]
    return "soup" if w["cell"] == "W_soup" else f"seed{w['best_seed']}"


def _tflite_outputs_r4(path, x3, width):
    """Raw float outputs for [n, 1, 12] inputs; dequantizes INT8 IO."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    outputs = np.empty((len(x3), width), dtype=np.float32)
    is_int8 = inp["dtype"] == np.int8
    scale, zp = inp["quantization"]
    o_scale, o_zp = out["quantization"]
    for k in range(len(x3)):
        x = x3[k][None].astype(np.float32)
        if is_int8:
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / scale) + zp, info.min, info.max).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] == np.int8:
            y = (y - o_zp) * o_scale
        outputs[k] = y
    return outputs


def stage_convert(run_id):
    import torch
    from conversion.common import load_checkpoint
    from conversion.convert import convert_float, convert_int8
    from experiments.accuracy_sweep import _data
    from training.deltas import delta_windows
    from training.heads import head_outputs
    run = _run_dir(run_id)
    head = _read_sel(run, "W")["winner"]["head"]
    ds, _, _ = _data("weinstock", 12)
    art = Path(run) / "artifacts"
    art.mkdir(exist_ok=True)
    sample = torch.zeros(1, 1, 12, dtype=torch.float32)
    for label, ckpt in _winner_ckpts(run):
        model = load_checkpoint(ckpt, head_outputs(head))
        fpath = art / f"winner_{label}_float.tflite"
        if not fpath.exists():
            convert_float(model, sample, fpath)
            print(f"converted {fpath.name}")
        for sel_path in sorted((Path(run) / "selections").glob("*_200_d*.json")):
            ipath = art / f"winner_{label}_int8_{sel_path.stem}.tflite"
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


def _pedap_windows_r4():
    from training.deltas import delta_windows
    from training.mixtures import HOLDOUT_COHORT
    levels, _, y, pids = _cohort_cache_r4(HOLDOUT_COHORT)
    x = delta_windows(levels, True).astype(np.float32)[:, None, :]
    return x, {"labels": y.tolist(), "participants": pids.tolist()}


def stage_evaluate(run_id):
    from experiments.accuracy_sweep import (CHAMPION, _component_windows,
                                            _tflite_logits)
    from training.heads import head_outputs
    if not protocol_is_committed(PROTOCOL_R4):
        raise SystemExit("REFUSED: commit paper/results/calibration-coverage/"
                         "protocol_sweep_r4.json (clean) before evaluate")
    run = _run_dir(run_id)
    width = head_outputs(_read_sel(run, "W")["winner"]["head"])
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        base = Path(run) / "eval" / comp
        base.mkdir(parents=True, exist_ok=True)
        windows, meta = _component_windows(12, split)
        if not (base / "meta.json").exists():
            _write_json(base / "meta.json", meta)
        x3 = windows.astype(np.float32)[:, None, :] if windows.ndim == 2 else windows
        for label, _ckpt in _winner_ckpts(run):
            for art in sorted((Path(run) / "artifacts").glob(f"winner_{label}_*.tflite")):
                p = base / f"{art.stem}_outputs.npy"
                if not p.exists():
                    np.save(p, _tflite_outputs_r4(art, x3, width))
                    print(f"{comp}/{art.stem} evaluated")
        for kind in ("float", "int8"):
            p = base / f"champion_{kind}_preds.npy"
            if not p.exists():
                np.save(p, _tflite_logits(CHAMPION[kind], windows).argmax(axis=1))
                print(f"{comp}/champion_{kind} evaluated")
    base = Path(run) / "eval" / "pedap"
    base.mkdir(parents=True, exist_ok=True)
    x3, meta = _pedap_windows_r4()
    if not (base / "meta.json").exists():
        _write_json(base / "meta.json", meta)
    cand = (Path(run) / "artifacts" /
            f"winner_{_candidate_label(run)}_int8_uniform_random_200_d101.tflite")
    p = base / "winner_int8_outputs.npy"
    if not p.exists():
        np.save(p, _tflite_outputs_r4(cand, x3, width))
        print("pedap/winner_int8 evaluated")
    p = base / "champion_int8_preds.npy"
    if not p.exists():
        from experiments.accuracy_sweep import CHAMPION, _tflite_logits
        np.save(p, _tflite_logits(CHAMPION["int8"], x3[:, 0, :]).argmax(axis=1))
        print("pedap/champion_int8 evaluated")


def _pairing_block(n):
    return {"pairing_rule": "identical eligible windows by construction",
            "dropped_windows_winner": 0, "dropped_windows_champion": 0,
            "windows": n}


def stage_analyze(run_id, n_boot=2000):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    from training.heads import decode_outputs
    from training.labeling import FIVE_CLASSES
    run = _run_dir(run_id)
    head = _read_sel(run, "W")["winner"]["head"]
    cand = _candidate_label(run)
    ship_arm = "int8_uniform_random_200_d101"
    labels = [label for label, _ in _winner_ckpts(run)]
    out = {"protocol": "paper/results/calibration-coverage/protocol_sweep_r4.json",
           "run_id": run_id,
           "exp0": json.loads(EXP0_PATH.read_text()),
           "selection_trail": {s: _read_sel(run, s) for s in ("H", "R", "W")},
           "smoke": json.loads((Path(run) / "smoke.json").read_text()),
           "per_arm_train_windows": {
               p.stem: int(np.load(p)["y"].shape[0])
               for p in sorted((Path(run) / "tensors").glob("*.npz"))},
           "components": {}}
    for comp in ("held_out", "temporal"):
        base = Path(run) / "eval" / comp
        meta = json.loads((base / "meta.json").read_text())
        y = np.array(meta["labels"])
        pids = np.array(meta["participants"])
        champ = np.load(base / "champion_int8_preds.npy")
        cand_int8 = decode_outputs(head, np.load(
            base / f"winner_{cand}_{ship_arm}_outputs.npy"))
        cand_float = decode_outputs(head, np.load(
            base / f"winner_{cand}_float_outputs.npy"))
        per_label = {lab: {
            k: directional_recall(confusion_counts(y, decode_outputs(
                head, np.load(base / f"winner_{lab}_{k}_outputs.npy"))))
            for k in ("float", ship_arm, "int8_sequential_200_d0")}
            for lab in labels}
        out["components"][comp] = {
            **_pairing_block(len(y)),
            "per_artifact_directional": per_label,
            "macro_f1_candidate_int8": macro_f1(confusion_counts(y, cand_int8)),
            "macro_f1_champion_int8": macro_f1(confusion_counts(y, champ)),
            "gates": gate_results_r4(y, pids, cand_int8, cand_float, champ,
                                     n_boot=n_boot),
        }
    base = Path(run) / "eval" / "pedap"
    meta = json.loads((base / "meta.json").read_text())
    y = np.array(meta["labels"])
    ped = {}
    winner_preds = decode_outputs(head, np.load(base / "winner_int8_outputs.npy"))
    for name, preds in (("winner_int8", winner_preds),
                        ("champion_int8", np.load(base / "champion_int8_preds.npy"))):
        cm = confusion_counts(y, preds)
        ped[name] = {"directional": directional_recall(cm),
                     "stable": stable_recall(cm),
                     "per_class": {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
                                   for i, c in enumerate(FIVE_CLASSES)}}
    out["pedap_zero_shot"] = {"windows": int(len(y)), **ped,
                              "note": "descriptive; no gates; cohort never trained on"}
    g = out["components"]["held_out"]["gates"]
    out["ship"] = {
        "candidate": cand,
        "all_gates_pass": bool(g["gate1_beats_champion"] and g["gate2_stable_floor"]
                               and g["gate3_int8_within_half_pp_of_float"]),
    }
    out["deviations"] = list(DEVIATIONS_R4)
    (Path(run) / "analysis").mkdir(exist_ok=True)
    _write_json(Path(run) / "analysis" / "sweep_r4_aggregate.json", out)
    print("wrote", Path(run) / "analysis" / "sweep_r4_aggregate.json")
    return out


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
