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


def test_epoch_table_rebuilds_arm_head(tmp_path):
    import torch
    from experiments.accuracy_sweep_r5 import _epoch_table, arm_r5
    from training.model import TrendCNN

    class _V:
        x = np.zeros((4, 1, 12), dtype=np.float32)
        y = np.array([0, 1, 2, 3])
    a = arm_r5("H_flat", "flat", "val")
    ed = tmp_path / "epochs"
    ed.mkdir()
    for e in range(2):
        m = TrendCNN(num_classes=5, channels=(96, 144), head="flat")
        torch.save({k: v for k, v in m.state_dict().items()},
                   ed / f"epoch{e:02d}.pt")
    table = _epoch_table(ed, a, _V())
    assert set(table) == {0, 1}
    assert all(len(v) == 2 for v in table.values())


def test_stage_train_refuses_reuse_and_foreign_mixture(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    monkeypatch.setattr(r5, "_frozen_batch", lambda run: 512)
    # reuse cell with missing alias target -> loud refusal, no training
    monkeypatch.setattr(r5, "R3_RUN_DIR", tmp_path / "nope")
    monkeypatch.setattr(r5, "_arms_for",
                        lambda run, s: [r5.arm_r5("R_base", "gap", "base")])
    with pytest.raises(SystemExit, match="REFUSED"):
        r5.stage_train("sweep-r5-test", "R")
    # any non-D5 mixture -> refusal before any data is touched
    bad = dict(r5.arm_r5("R_val", "gap", "val"), mixture="D5R")
    monkeypatch.setattr(r5, "_arms_for", lambda run, s: [bad])
    with pytest.raises(SystemExit, match="RT-CGM|D5"):
        r5.stage_train("sweep-r5-test", "R")


def test_train_one_r5_val_recipe_keeps_selected_epoch(tmp_path, monkeypatch):
    import torch
    import experiments.accuracy_sweep_r5 as r5
    import training.train as tt

    class _V:
        def __init__(self, n=32):
            rng = np.random.default_rng(0)
            self.x = rng.normal(size=(n, 1, 12)).astype(np.float32)
            self.y = rng.integers(0, 5, size=n).astype(np.int64)
            self.labels = self.y.tolist()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, k):
            return torch.from_numpy(self.x[k]), int(self.y[k])

    monkeypatch.setattr(r5, "EPOCHS", 2)
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.5, "stable": 0.0})
    monkeypatch.setattr(tt, "resolve_device",
                        lambda prefer="mps": torch.device("cpu"))
    a = r5.arm_r5("R_val", "gap", "val")
    v = _V()
    ckpt = r5._train_one_r5(tmp_path, a, 0, v, v, batch=16)
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert obj["head"] == "gap" and obj["arch"] == "cnn"
    side = json.loads(ckpt.with_suffix(".json").read_text())
    assert "kept_epoch" in side and side["kept_epoch"] in (0, 1)
    assert side["epoch_floor_flagged"] is False  # floor 0.0-0.02 passes
    # the saved weights ARE the kept epoch's weights
    ep = (tmp_path / "checkpoints" / "epochs" / "R_val_seed0"
          / f"epoch{side['kept_epoch']:02d}.pt")
    kept = torch.load(ep, map_location="cpu", weights_only=True)
    assert all(torch.equal(obj["state_dict"][k], kept[k]) for k in kept)
