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
