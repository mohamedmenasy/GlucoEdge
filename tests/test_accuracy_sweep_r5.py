import json

import numpy as np
import pytest

from experiments.accuracy_sweep_r5 import (ARM_SEEDS_R5, BOOT_SEED_R5,
                                           FLOOR_MARGIN_R5, PARAM_CAP_R5,
                                           _ckpt_r5, _tensor_path_r5, arm_r5,
                                           stage_h_arms, stage_r_arms)


def test_constants():
    assert ARM_SEEDS_R5 == (0, 1, 2, 3, 4)
    assert BOOT_SEED_R5 == 20261015
    assert FLOOR_MARGIN_R5 == 0.02  # selection floor == Gate 2 margin
    assert PARAM_CAP_R5 == 85_450


def test_arms_never_touch_rtcgm():
    r = stage_r_arms()
    arms = r + stage_h_arms(r[1])
    assert all(a["mixture"] == "D5" for a in arms)
    assert [a["name"] for a in r] == ["R_base", "R_val"]
    assert [a["recipe"] for a in r] == ["base", "val"]
    assert all(a["head"] == "gap" for a in r)


def test_stage_h_inherits_r_winner_recipe():
    h = stage_h_arms(arm_r5("R_val", "gap", "val"))
    assert [a["name"] for a in h] == ["H_gap", "H_flat"]
    assert h[0]["head"] == "gap" and h[0]["recipe"] == "val"
    assert h[1]["head"] == "flat" and h[1]["recipe"] == "val"
    h2 = stage_h_arms(arm_r5("R_base", "gap", "base"))
    assert h2[1]["recipe"] == "base"  # flat trains with the winning recipe


def test_ckpt_key_covers_head_and_recipe(tmp_path):
    a = arm_r5("H_flat", "flat", "val")
    p = _ckpt_r5(tmp_path, a, 3)
    assert p.name == "D5_none_cnn_flat_val_seed3.pt"
    assert p.parent == tmp_path / "checkpoints"


def test_r_base_aliases_r3_checkpoints(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r5 as r5
    fake = tmp_path / "r3"
    (fake / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(r5, "R3_RUN_DIR", fake)
    a = arm_r5("R_base", "gap", "base")
    with pytest.raises(SystemExit, match="REFUSED"):
        _ckpt_r5(tmp_path, a, 0)  # alias target missing -> loud
    (fake / "checkpoints" / "D5_none_cnn_seed0.pt").write_bytes(b"x")
    assert _ckpt_r5(tmp_path, a, 0).name == "D5_none_cnn_seed0.pt"
    # H_gap carrying the base recipe resolves to the same alias
    hg = dict(a, name="H_gap")
    assert _ckpt_r5(tmp_path, hg, 0) == _ckpt_r5(tmp_path, a, 0)


def test_tensor_path_is_r3_d5_and_refuses_if_missing(monkeypatch, tmp_path):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "R3_RUN_DIR", tmp_path)
    with pytest.raises(SystemExit, match="REFUSED"):
        _tensor_path_r5()
    t = tmp_path / "tensors"
    t.mkdir()
    (t / "D5_none_s20261012.npz").write_bytes(b"x")
    assert _tensor_path_r5().name == "D5_none_s20261012.npz"
