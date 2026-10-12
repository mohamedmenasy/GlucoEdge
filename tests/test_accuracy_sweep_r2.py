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


def test_gate_bootstrap_uses_preregistered_r2_seed():
    y = np.array([0, 1, 2, 3, 4] * 40)
    pids = np.array([f"p{i % 4}" for i in range(200)])
    g = gate_results_r2(y, pids, cand_int8=y, cand_float=y, champ_int8=y, n_boot=50)
    assert g["contrast_int8_vs_champion"]["seed"] == 20261012


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


def test_stage_m_arms_cover_five_mixtures_at_96x144():
    from experiments.accuracy_sweep_r2 import stage_m_arms
    arms = stage_m_arms()
    assert [a["mixture"] for a in arms] == ["M0", "M1", "M2", "M3", "M4"]
    assert all(a["channels"] == [96, 144] and a["length"] == 12 for a in arms)


def test_stage_c_arms_bracket_winner_capacity():
    from experiments.accuracy_sweep_r2 import arm_r2, stage_c_arms
    w = arm_r2("M_M1", "M1", (96, 144))
    names = [a["channels"] for a in stage_c_arms(w)]
    assert names == [[24, 36], [192, 288]]
    assert all(a["mixture"] == "M1" for a in stage_c_arms(w))


def test_tensor_cache_key_includes_arm_and_sample_seed(tmp_path):
    from experiments.accuracy_sweep_r2 import _tensor_path, arm_r2
    a = arm_r2("M_M1", "M1", (96, 144))
    p1 = _tensor_path(tmp_path, a, sample_seed=0)
    p2 = _tensor_path(tmp_path, a, sample_seed=1)
    assert p1 != p2 and "M_M1" in p1.name


def test_select_disqualifies_stable_killers_and_fails_if_all_do(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r2 as r2
    arms = [r2.arm_r2("M_M0", "M0", (96, 144)), r2.arm_r2("M_M1", "M1", (96, 144))]
    stats = {"M_M0": (0.60, 0.30), "M_M1": (0.50, 0.55)}
    monkeypatch.setattr(r2, "_val_stats", lambda run, a, seed: stats[a["name"]])
    monkeypatch.setattr(r2, "champion_val_stats", lambda run=None: {"directional": 0.45, "stable": 0.50})
    out = r2._select_from_arms(tmp_path, "M", arms)
    assert out["winner"]["name"] == "M_M1"
    assert out["table"]["M_M0"]["disqualified"] is True
    both_bad = {"M_M0": (0.60, 0.30), "M_M1": (0.50, 0.30)}
    monkeypatch.setattr(r2, "_val_stats", lambda run, a, seed: both_bad[a["name"]])
    with pytest.raises(SystemExit):
        r2._select_from_arms(tmp_path, "M", arms)


def test_stage_evaluate_refuses_without_committed_r2_protocol(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r2 as r2
    monkeypatch.setattr(r2, "PROTOCOL_R2", tmp_path / "protocol_sweep_r2.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        r2.stage_evaluate("sweep-r2-test")
