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


def test_pick_epoch_floor_tieband_and_flag():
    from experiments.accuracy_sweep_r5 import pick_epoch
    floor = 0.50
    per_epoch = {0: (0.54, 0.49),   # best directional but fails floor
                 1: (0.528, 0.52),
                 2: (0.5305, 0.51),  # inside 0.1pp band of epoch 3
                 3: (0.5309, 0.51)}
    # band is {2, 3}; earliest wins
    assert pick_epoch(per_epoch, floor) == (2, False)
    # every epoch fails the floor -> final epoch kept, flagged
    assert pick_epoch({0: (0.6, 0.1), 1: (0.5, 0.2)}, floor) == (1, True)


def test_param_count_matches_model_r5():
    from experiments.accuracy_sweep_r5 import _param_count, arm_r5
    assert _param_count(arm_r5("H_gap", "gap", "val")) == 42_725
    assert _param_count(arm_r5("H_flat", "flat", "val")) == 50_645
    assert _param_count(arm_r5("H_flat", "flat", "val")) <= 85_450


def test_pick_winner_r5_tie_band_cost_then_order():
    from experiments.accuracy_sweep_r5 import arm_r5, pick_winner_r5
    gap = arm_r5("H_gap", "gap", "val")
    flat = arm_r5("H_flat", "flat", "val")
    # inside the band: cheaper head (gap) wins even listed second
    w = pick_winner_r5([flat, gap], {"H_flat": 0.5305, "H_gap": 0.5300})
    assert w["name"] == "H_gap"
    # outside the band: argmax wins
    w = pick_winner_r5([flat, gap], {"H_flat": 0.5320, "H_gap": 0.5300})
    assert w["name"] == "H_flat"
    # recipe base before val at equal params
    base = arm_r5("R_base", "gap", "base")
    val = arm_r5("R_val", "gap", "val")
    w = pick_winner_r5([base, val], {"R_base": 0.5300, "R_val": 0.5305})
    assert w["name"] == "R_base"
