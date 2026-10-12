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


def _train_one(run, a, seed, train_v, val_v, batch, ckpt=None):
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
    if ckpt is None:
        ckpt = Path(run) / "checkpoints" / f"{a['name']}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)
    return ckpt


def _dir_recall_from_ckpt(ckpt, val_v):
    import torch
    from conversion.common import load_checkpoint
    from experiments.analyze_calibration_coverage import confusion_counts
    model = load_checkpoint(ckpt, 5)
    with torch.no_grad():
        logits = model(torch.from_numpy(val_v.x).unsqueeze(1)).numpy()
    return directional_recall(confusion_counts(val_v.y, logits.argmax(axis=1)))


def stage_smoke(run_id):
    """Freeze the batch size: M0 at batch 64 vs candidates, 2 seeds each;
    accept the largest batch within 1.0pp mean val directional recall of
    64. Also records same-seed reproducibility on this device."""
    run = _run_dir(run_id)
    a = arm_r2("SMOKE_M0", "M0", (96, 144))
    path = build_arm_tensors(run, a)
    train_v, val_v = _load_view(path), _val_view()
    results = {}

    def measure(batch):
        per_seed = {}
        for seed in (0, 1):
            ckpt = Path(run) / "checkpoints" / f"smoke_b{batch}_seed{seed}.pt"
            if not ckpt.exists():
                t0 = time.time()
                _train_one(run, a, seed, train_v, val_v, batch, ckpt)
                print(f"smoke b{batch} seed{seed}: {time.time()-t0:.0f}s")
            per_seed[seed] = _dir_recall_from_ckpt(ckpt, val_v)
        results[batch] = per_seed
        return float(np.mean(list(per_seed.values())))

    mean64 = measure(64)
    frozen = 64
    for batch in (512, 256, 128):
        if measure(batch) >= mean64 - 0.01:
            frozen = batch
            break
    repro_ckpt = Path(run) / "checkpoints" / f"smoke_b{frozen}_seed0_repeat.pt"
    if not repro_ckpt.exists():
        _train_one(run, a, 0, train_v, val_v, frozen, repro_ckpt)
    base_ckpt = Path(run) / "checkpoints" / f"smoke_b{frozen}_seed0.pt"
    repro = abs(_dir_recall_from_ckpt(repro_ckpt, val_v)
                - _dir_recall_from_ckpt(base_ckpt, val_v))
    _write_json(Path(run) / "smoke.json", {
        "results": {str(b): {str(s): v for s, v in r.items()} for b, r in results.items()},
        "mean64": mean64, "frozen_batch": frozen,
        "mps_same_seed_abs_diff": repro,
        "mps_seed_reproducible": bool(repro < 1e-6)})
    print(json.dumps({"frozen_batch": frozen, "repro_diff": repro}, indent=1))


def _val_stats(run, a, seed):
    """(directional, stable) recall on the eligible weinstock val set."""
    import torch
    from conversion.common import load_checkpoint
    from experiments.analyze_calibration_coverage import confusion_counts
    val_v = _val_view()
    ckpt = Path(run) / "checkpoints" / f"{a['name']}_seed{seed}.pt"
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
        return [_read_sel(run, "C")["winner"]], (3, 4)
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
    sel.mkdir(parents=True, exist_ok=True)
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
    a = arms[0]
    per_seed = {str(s): _val_stats(run, a, s) for s in FINAL_SEEDS}
    best_seed = max(per_seed, key=lambda s: per_seed[s][0])
    out = {"stage": "S",
           "table": {"per_seed": {s: {"directional": v[0], "stable": v[1]}
                                  for s, v in per_seed.items()}},
           "best_seed": int(best_seed),
           "winner": dict(a, ensemble=False)}
    (Path(run) / "selection").mkdir(parents=True, exist_ok=True)
    _write_json(Path(run) / "selection" / "S.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))


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
    winner_art = (Path(run) / "artifacts" /
                  f"winner_seed{sel_s['best_seed']}_int8_uniform_random_200_d101.tflite")
    for name, path in (("winner_int8", winner_art), ("champion_int8", CHAMPION["int8"])):
        p = base / f"{name}_preds.npy"
        if not p.exists():
            np.save(p, _tflite_logits(path, windows).argmax(axis=1))
            print(f"pedap/{name} evaluated")


def stage_analyze(run_id, n_boot=2000):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    from training.labeling import FIVE_CLASSES
    run = _run_dir(run_id)
    sel_s = _read_sel(run, "S")
    best_seed = sel_s["best_seed"]
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
