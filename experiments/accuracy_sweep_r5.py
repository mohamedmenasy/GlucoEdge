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


def _frozen_batch(run):
    from experiments.accuracy_sweep_r2 import _frozen_batch as _fb
    return _fb(run)


def stage_train(run_id, stage):
    import time as _t
    from experiments.accuracy_sweep_r3 import _load_view_r3, _val_for
    run = _run_dir(run_id)
    batch = _frozen_batch(run)
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


def stage_smoke(run_id):
    """Batch freeze + same-seed bit-reproduction record, r2 recipe."""
    from experiments.accuracy_sweep_r2 import stage_smoke as smoke_r2
    smoke_r2(run_id)


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
