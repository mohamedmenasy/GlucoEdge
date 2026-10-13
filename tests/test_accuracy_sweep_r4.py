import json

import numpy as np
import pytest

from experiments.accuracy_sweep_r4 import (ARM_SEEDS_R4, BOOT_SEED_R4,
                                           HEADROOM_MIN, NoisyLevelView,
                                           RateView, _tensor_path_r4,
                                           _train_targets, arm_r4,
                                           stage_h_arms, stage_r_arms)


def test_constants():
    assert ARM_SEEDS_R4 == (0, 1, 2, 3, 4)
    assert BOOT_SEED_R4 == 20261014
    assert HEADROOM_MIN == 0.02


def test_stage_h_arms():
    arms = stage_h_arms()
    assert [a["name"] for a in arms] == ["H_ce", "H_wtemp", "H_reg", "H_corn"]
    assert [a["head"] for a in arms] == ["ce", "wtemp", "reg", "corn"]
    assert all(a["mixture"] == "D5" and a["in_channels"] == 1 for a in arms)


def test_stage_r_arms_mixup_gated_on_reg():
    reg_winner = arm_r4("H_reg", "reg")
    ce_winner = arm_r4("H_ce", "ce")
    assert [a["name"] for a in stage_r_arms(reg_winner)] == [
        "R_base", "R_labels", "R_noise", "R_mixup"]
    assert [a["name"] for a in stage_r_arms(ce_winner)] == [
        "R_base", "R_labels", "R_noise"]
    # R arms inherit the winning head
    assert all(a["head"] == "ce" for a in stage_r_arms(ce_winner))


def test_tensor_key_ignores_head_but_not_labels_or_aug(tmp_path):
    a1 = arm_r4("H_ce", "ce")
    a2 = arm_r4("H_corn", "corn")
    a3 = arm_r4("R_labels", "ce", labels="lsq")
    a4 = arm_r4("R_noise", "ce", aug="noise")
    p = {k: _tensor_path_r4(tmp_path, a, 1) for k, a in
         {"a1": a1, "a2": a2, "a3": a3, "a4": a4}.items()}
    assert p["a1"] == p["a2"]                     # heads share tensors
    assert len({p["a1"], p["a3"], p["a4"]}) == 3  # labels/aug do not


def test_train_targets_routing():
    z = {"y": np.array([2, 3]), "r": np.array([0.5, 1.2]),
         "lsq": np.array([0.4, 1.1]),
         "levels": np.zeros((2, 12), dtype=np.float32)}
    assert _train_targets(arm_r4("H_ce", "ce"), z).tolist() == [2, 3]
    assert np.allclose(_train_targets(arm_r4("H_reg", "reg"), z), [0.5, 1.2])
    lsq_reg = _train_targets(arm_r4("R_labels", "reg", labels="lsq"), z)
    assert np.allclose(lsq_reg, [0.4, 1.1])
    lsq_cls = _train_targets(arm_r4("R_labels", "ce", labels="lsq"), z)
    assert lsq_cls.tolist() == [2, 3]  # 0.4 -> stable(2), 1.1 -> rising(3)


def test_rate_view_yields_float_targets():
    v = RateView(np.zeros((3, 1, 12), dtype=np.float32),
                 np.array([0.5, -1.5, 2.5], dtype=np.float32))
    x, r = v[1]
    assert x.shape == (1, 12) and isinstance(r, float) and r == -1.5


def test_noisy_view_reencodes_per_access():
    levels = np.full((2, 12), 120.0, dtype=np.float32)
    v = NoisyLevelView(levels, np.array([2, 2]), sigma=2.0, seed=0,
                       float_targets=False)
    x1, _ = v[0]
    x2, _ = v[0]
    assert x1.shape == (1, 12) and not np.array_equal(x1.numpy(), x2.numpy())


def test_build_tensors_refuses_non_finite_rates(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r4 as r4

    def fake_cohort(cohort):
        levels = np.full((3, 12), 100.0, dtype=np.float32)
        horizons = np.full((3, 3), np.nan, dtype=np.float32)  # corrupted
        y = np.array([2, 2, 2], dtype=np.int64)
        pids = np.array(["p1", "p2", "p3"])
        return levels, horizons, y, pids

    monkeypatch.setattr(r4, "_cohort_cache_r4", fake_cohort)
    with pytest.raises(SystemExit, match="non-finite"):
        r4.build_arm_tensors_r4(tmp_path, r4.arm_r4("H_ce", "ce"), sample_seed=1)
