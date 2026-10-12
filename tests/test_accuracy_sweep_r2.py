"""Sweep r2 pure-logic tests — no data needed."""
import numpy as np
import pytest

from experiments.accuracy_sweep_r2 import (directional_recall, gate_results_r2,
                                           stable_recall)


def _cm(y, pred):
    from experiments.analyze_calibration_coverage import confusion_counts
    return confusion_counts(np.asarray(y), np.asarray(pred))


def test_stable_and_directional_recall_hand_computed():
    y =    [0, 0, 1, 2, 2, 2, 3, 4]
    pred = [0, 2, 1, 2, 2, 0, 1, 4]
    cm = _cm(y, pred)
    assert stable_recall(cm) == pytest.approx(2 / 3)
    # per-class recalls: c0=1/2, c1=1, c3=0, c4=1 -> directional = 0.625
    assert directional_recall(cm) == pytest.approx(0.625)


def test_gate_results_r2_known_outcomes():
    y = np.array([0, 1, 2, 3, 4] * 40)
    pids = np.array([f"p{i % 4}" for i in range(200)])
    perfect = y.copy()
    stable_killer = y.copy()
    stable_killer[y == 2] = 1          # perfect directional, zero stable recall
    worse = y.copy()
    worse[:60] = (worse[:60] + 1) % 5
    g = gate_results_r2(y, pids, cand_int8=perfect, cand_float=perfect,
                        champ_int8=worse, n_boot=200)
    assert g["gate1_beats_champion"] and g["gate2_stable_floor"] \
        and g["gate3_int8_within_half_pp_of_float"]
    g2 = gate_results_r2(y, pids, cand_int8=stable_killer, cand_float=stable_killer,
                         champ_int8=perfect, n_boot=200)
    assert g2["gate2_stable_floor"] is False


def test_protocol_guard_accepts_explicit_path(tmp_path):
    import subprocess
    from experiments.accuracy_sweep import protocol_is_committed
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    p = tmp_path / "protocol_sweep_r2.json"
    assert not protocol_is_committed(p)
    p.write_text("{}")
    subprocess.run(["git", "-C", str(tmp_path), "add", p.name], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-qm", "freeze"], check=True)
    assert protocol_is_committed(p)
