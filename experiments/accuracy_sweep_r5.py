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
