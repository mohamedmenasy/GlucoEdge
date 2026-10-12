"""r3 pure-logic tests - no CGM data needed."""
import json

import numpy as np
import pytest

import experiments.accuracy_sweep_r3 as r3


def test_arm_r3_carries_cost_key_fields_and_in_channels():
    a = r3.arm_r3("D_base", "D5", "none", "cnn")
    assert a["channels"] == [96, 144] and a["length"] == 12
    assert a["loss"] == "ce" and a["pooled"] is False
    assert a["in_channels"] == 1
    assert r3.arm_r3("F_tod", "D5", "tod", "cnn")["in_channels"] == 3


def test_stage_arms_shapes():
    d = r3.stage_d_arms()
    assert [a["mixture"] for a in d] == ["D5", "D5R"]
    w = r3.arm_r3("D_rtcgm", "D5R", "none", "cnn")
    f = r3.stage_f_arms(w)
    assert [a["features"] for a in f] == ["none", "tod"]
    assert all(a["mixture"] == "D5R" for a in f)
    a_ = r3.stage_a_arms(f[1])
    assert [x["arch"] for x in a_] == ["cnn", "tcn"]
    assert all(x["features"] == "tod" for x in a_)


def test_tensor_path_keyed_by_mixture_features_seed(tmp_path):
    a1 = r3.arm_r3("D_base", "D5", "none", "cnn")
    a2 = r3.arm_r3("F_tod", "D5", "tod", "cnn")
    p1 = r3._tensor_path_r3(tmp_path, a1, 0)
    p2 = r3._tensor_path_r3(tmp_path, a2, 0)
    p3 = r3._tensor_path_r3(tmp_path, a1, 1)
    assert len({p1, p2, p3}) == 3 and "D5_none" in p1.name


def test_d5r_without_admit_list_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(r3, "ADMITTED_PATH", tmp_path / "missing.json")
    a = r3.arm_r3("D_rtcgm", "D5R", "none", "cnn")
    with pytest.raises(SystemExit, match="rtcgm"):
        r3.build_arm_tensors_r3(tmp_path, a)


def test_load_view_r3_asserts_channel_shape(tmp_path):
    p = tmp_path / "t.npz"
    np.savez(p, x=np.zeros((4, 1, 12), np.float32), y=np.zeros(4, np.int64))
    with pytest.raises(SystemExit, match="channels"):
        r3._load_view_r3(p, in_channels=3)
    v = r3._load_view_r3(p, in_channels=1)
    assert isinstance(v, r3.TensorView3) and len(v) == 4
    x0, y0 = v[0]
    assert tuple(x0.shape) == (1, 12) and y0 == 0


def test_ckpt_key_covers_mixture_features_arch(tmp_path):
    a1 = r3.arm_r3("D_base", "D5", "none", "cnn")
    a2 = r3.arm_r3("A_tcn", "D5", "none", "tcn")
    assert r3._ckpt(tmp_path, a1, 0) != r3._ckpt(tmp_path, a2, 0)
    # same cell under two stage names -> same path (reuse, not retrain)
    a3 = r3.arm_r3("F_base", "D5", "none", "cnn")
    assert r3._ckpt(tmp_path, a1, 0) == r3._ckpt(tmp_path, a3, 0)


def test_select_disqualifier_and_all_dq_failure(tmp_path, monkeypatch):
    arms = r3.stage_d_arms()
    stats = {"D_base": (0.52, 0.40), "D_rtcgm": (0.54, 0.55)}
    monkeypatch.setattr(r3, "_val_stats_r3", lambda run, a, seed: stats[a["name"]])
    monkeypatch.setattr(r3, "_champion_val", lambda run: {"directional": 0.50, "stable": 0.56})
    out = r3._select(tmp_path, "D", arms)
    assert out["winner"]["name"] == "D_rtcgm"
    assert out["table"]["D_base"]["disqualified"] is True
    bad = {"D_base": (0.52, 0.40), "D_rtcgm": (0.54, 0.40)}
    monkeypatch.setattr(r3, "_val_stats_r3", lambda run, a, seed: bad[a["name"]])
    with pytest.raises(SystemExit):
        r3._select(tmp_path, "D", arms)


def test_boot_seed_single_source():
    y = np.array([0, 1, 2, 3, 4] * 40)
    pids = np.array([f"p{i % 4}" for i in range(200)])
    g = r3.gate_results_r3(y, pids, cand_int8=y, cand_float=y, champ_int8=y, n_boot=50)
    assert g["contrast_int8_vs_champion"]["seed"] == 20261013
