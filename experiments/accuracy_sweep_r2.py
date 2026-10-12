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
