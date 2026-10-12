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


def stage_smoke(run_id):
    """Batch freeze reused from r2's recipe, written into the r3 run dir."""
    from experiments.accuracy_sweep_r2 import stage_smoke as smoke_r2
    smoke_r2(run_id)


def stage_calib(run_id):
    from experiments.accuracy_sweep_r2 import stage_calib as calib_r2
    calib_r2(run_id)


def _winner_ckpt(run, winner, seed):
    return _ckpt(run, winner, seed)


def _hours_from_keys(keys):
    import pandas as pd
    ts = [pd.Timestamp(k[1]) for k in keys]
    return np.array([t.hour + t.minute / 60 + t.second / 3600 for t in ts])


def _tflite_logits_r3(path, x3):
    """Float logits for [n, C, 12] inputs; dequantizes INT8 IO."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    logits = np.empty((len(x3), 5), dtype=np.float32)
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
        logits[k] = y
    return logits


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
    nch = winner["in_channels"]
    sample = torch.zeros(1, nch, 12, dtype=torch.float32)
    for seed in ARM_SEEDS_R3:
        model = load_checkpoint(_winner_ckpt(run, winner, seed), 5)
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
            dws = delta_windows(ws, True).astype(np.float32)
            hours = np.array([ds["val"].anchor_times[i].hour
                              + ds["val"].anchor_times[i].minute / 60
                              + ds["val"].anchor_times[i].second / 3600
                              for i in sel["indices"]])
            x3 = _with_features(dws, hours, winner["features"])
            calib = [torch.from_numpy(w[None]) for w in x3]
            timings = convert_int8(model, sample, calib, ipath)
            _write_json(ipath.with_suffix(".json"), {"selection": sel_path.stem, **timings})
            print(f"converted {ipath.name}")


def _pedap_windows_r3(features):
    from training.mixtures import HOLDOUT_COHORT
    w, y, pids, hours = _diadata_windows_r3(HOLDOUT_COHORT)
    return (_with_features(w, hours, features),
            w[:, None, :],
            {"labels": y.tolist(), "participants": pids.tolist()})


def stage_evaluate(run_id):
    from experiments.accuracy_sweep import (CHAMPION, _component_windows,
                                            _tflite_logits, protocol_is_committed)
    if not protocol_is_committed(PROTOCOL_R3):
        raise SystemExit("REFUSED: commit paper/results/calibration-coverage/"
                         "protocol_sweep_r3.json (clean) before evaluate")
    run = _run_dir(run_id)
    winner = _read_sel(run, "S")["winner"]
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        base = Path(run) / "eval" / comp
        base.mkdir(parents=True, exist_ok=True)
        windows, meta = _component_windows(12, split)
        if not (base / "meta.json").exists():
            _write_json(base / "meta.json", meta)
        hours = _hours_from_keys(meta["keys"])
        x3 = _with_features(windows.astype(np.float32), hours, winner["features"])
        for seed in ARM_SEEDS_R3:
            for art in sorted((Path(run) / "artifacts").glob(f"winner_seed{seed}_*.tflite")):
                p = base / f"{art.stem}_logits.npy"
                if not p.exists():
                    np.save(p, _tflite_logits_r3(art, x3))
                    print(f"{comp}/{art.stem} evaluated")
        for kind in ("float", "int8"):
            p = base / f"champion_{kind}_preds.npy"
            if not p.exists():
                np.save(p, _tflite_logits(CHAMPION[kind], windows).argmax(axis=1))
                print(f"{comp}/champion_{kind} evaluated")
    base = Path(run) / "eval" / "pedap"
    base.mkdir(parents=True, exist_ok=True)
    x3, x1, meta = _pedap_windows_r3(winner["features"])
    if not (base / "meta.json").exists():
        _write_json(base / "meta.json", meta)
    sel_s = _read_sel(run, "S")
    winner_art = (Path(run) / "artifacts" /
                  f"winner_seed{sel_s['best_seed']}_int8_uniform_random_200_d101.tflite")
    for name, path, x in (("winner_int8", winner_art, x3),
                          ("champion_int8", CHAMPION["int8"], x1)):
        p = base / f"{name}_preds.npy"
        if not p.exists():
            np.save(p, _tflite_logits_r3(path, x).argmax(axis=1))
            print(f"pedap/{name} evaluated")


def _pairing_block(n_winner, n_champion):
    return {"pairing_rule": "identical eligible windows by construction",
            "dropped_windows_winner": 0, "dropped_windows_champion": 0}


def stage_analyze(run_id, n_boot=2000):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    from training.labeling import FIVE_CLASSES
    run = _run_dir(run_id)
    sel_s = _read_sel(run, "S")
    best_seed = sel_s["best_seed"]
    ship_arm = "int8_uniform_random_200_d101"
    out = {"run_id": run_id,
           "selection_trail": {s: _read_sel(run, s) for s in ("D", "F", "A", "S")},
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
        cand_int8 = np.load(base / f"winner_seed{best_seed}_{ship_arm}_logits.npy").argmax(axis=1)
        cand_float = np.load(base / f"winner_seed{best_seed}_float_logits.npy").argmax(axis=1)
        per_seed = {str(s): {
            k: directional_recall(confusion_counts(
                y, np.load(base / f"winner_seed{s}_{k}_logits.npy").argmax(axis=1)))
            for k in ("float", ship_arm, "int8_sequential_200_d0")}
            for s in ARM_SEEDS_R3}
        out["components"][comp] = {
            "windows": len(y),
            **_pairing_block(len(y), len(y)),
            "per_seed_directional": per_seed,
            "macro_f1_candidate_int8": macro_f1(confusion_counts(y, cand_int8)),
            "macro_f1_champion_int8": macro_f1(confusion_counts(y, champ)),
            "gates": gate_results_r3(y, pids, cand_int8, cand_float, champ,
                                     n_boot=n_boot),
        }
    base = Path(run) / "eval" / "pedap"
    meta = json.loads((base / "meta.json").read_text())
    y = np.array(meta["labels"])
    ped = {}
    for name in ("winner_int8", "champion_int8"):
        preds = np.load(base / f"{name}_preds.npy")
        cm = confusion_counts(y, preds)
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
    _write_json(Path(run) / "analysis" / "sweep_r3_aggregate.json", out)
    print("wrote", Path(run) / "analysis" / "sweep_r3_aggregate.json")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "train", "select", "calib",
                                          "convert", "evaluate", "analyze"])
    parser.add_argument("--run-id", default="sweep-r3")
    parser.add_argument("--stage-name", choices=["D", "F", "A", "S"])
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
