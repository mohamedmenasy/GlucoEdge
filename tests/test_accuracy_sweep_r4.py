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
    assert x.shape == (1, 12) and float(r) == -1.5


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


def test_exp0_agreement_hand_case():
    from experiments.accuracy_sweep_r4 import exp0_agreement
    endpoint = np.array([0, 1, 2, 3, 4, 2, 2, 1])
    smoothed = np.array([0, 2, 2, 3, 4, 2, 3, 1])  # misses one falling, one stable
    out = exp0_agreement(endpoint, smoothed)
    # directional = mean recall over classes 0,1,3,4 = (1 + 0.5 + 1 + 1)/4
    assert abs(out["ceiling_directional"] - 0.875) < 1e-9
    assert abs(out["ceiling_stable"] - 2 / 3) < 1e-9


def test_exp0_verdict_boundary_exact_2pp_proceeds():
    from experiments.accuracy_sweep_r4 import exp0_verdict
    assert exp0_verdict(0.5530, 0.5330)["abort"] is False      # exactly 2.0pp
    assert exp0_verdict(0.55299, 0.5330)["abort"] is True      # just under
    assert abs(exp0_verdict(0.60, 0.5330)["headroom"] - 0.067) < 1e-9


def test_exp0_gate_refuses_without_file(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r4 as r4
    monkeypatch.setattr(r4, "EXP0_PATH", tmp_path / "exp0_sweep_r4.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        r4._exp0_gate()


def test_exp0_gate_refuses_on_abort(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r4 as r4
    p = tmp_path / "exp0_sweep_r4.json"
    p.write_text(json.dumps({"abort": True}))
    monkeypatch.setattr(r4, "EXP0_PATH", p)
    monkeypatch.setattr(r4, "protocol_is_committed", lambda path: True)
    with pytest.raises(SystemExit, match="REFUSED"):
        r4._exp0_gate()


def test_best_seed_tie_rule_lowest_in_band():
    from experiments.accuracy_sweep_r4 import best_seed_r4
    per_seed = {"0": (0.5300, 0.55), "1": (0.5308, 0.55), "2": (0.5309, 0.55),
                "3": (0.5200, 0.55), "4": (0.5100, 0.40)}
    # floor 0.50: seed 4 disqualified; top = 0.5309, band includes 1 and 2 and 0
    # (0.5300 is 0.09pp below top -> inside 0.1pp band) -> lowest index: 0
    assert best_seed_r4(per_seed, floor=0.50) == 0


def test_best_seed_all_below_floor_refused():
    from experiments.accuracy_sweep_r4 import best_seed_r4
    with pytest.raises(SystemExit, match="stable floor"):
        best_seed_r4({"0": (0.53, 0.30)}, floor=0.50)


def test_train_view_routing(tmp_path):
    from experiments.accuracy_sweep_r4 import _train_view, arm_r4
    z = {"x": np.zeros((4, 1, 12), dtype=np.float32),
         "levels": np.full((4, 12), 100.0, dtype=np.float32),
         "y": np.array([2, 2, 3, 1]), "r": np.array([0.1, 0.2, 1.1, -1.2]),
         "lsq": np.array([0.1, 0.2, 1.1, -1.2])}
    v, collate = _train_view(tmp_path, arm_r4("H_ce", "ce"), z)
    assert type(v).__name__ == "R4View" and collate is None
    v, collate = _train_view(tmp_path, arm_r4("H_reg", "reg"), z)
    assert type(v).__name__ == "RateView" and collate is None
    v, collate = _train_view(tmp_path, arm_r4("R_noise", "corn", aug="noise"), z)
    assert type(v).__name__ == "NoisyLevelView"
    v, collate = _train_view(tmp_path, arm_r4("R_mixup", "reg", aug="mixup"), z)
    assert type(v).__name__ == "RateView" and collate is not None


def test_val_stats_decode_plumbing(tmp_path, monkeypatch):
    # a fake 1-output checkpoint decoding through thresholds end to end
    import torch

    import experiments.accuracy_sweep_r4 as r4
    from training.model import TrendCNN

    m = TrendCNN(num_classes=1, channels=(4, 6))
    torch.save({"state_dict": m.state_dict(), "arch": "cnn", "in_channels": 1,
                "channels": [4, 6], "head": "reg"}, tmp_path / "w.pt")
    fake_val = r4.R4View(np.zeros((6, 1, 12), dtype=np.float32),
                         np.array([0, 1, 2, 3, 4, 2]))
    monkeypatch.setattr(r4, "_val_r4", lambda: fake_val)
    d, s = r4._val_stats_r4(tmp_path, tmp_path / "w.pt", "reg")
    assert 0.0 <= d <= 1.0 and 0.0 <= s <= 1.0


def test_stage_w_cells(tmp_path, monkeypatch):
    import torch

    import experiments.accuracy_sweep_r4 as r4
    from training.model import TrendCNN

    run = tmp_path
    a = r4.arm_r4("R_noise", "ce", aug="noise")
    (run / "selection").mkdir()
    (run / "selection" / "R.json").write_text(json.dumps(
        {"winner": a, "champion_val": {"directional": 0.5, "stable": 0.5}}))
    for s in r4.ARM_SEEDS_R4:
        m = TrendCNN(num_classes=5, channels=(4, 6))
        p = r4._ckpt_r4(run, a, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": m.state_dict(), "arch": "cnn",
                    "in_channels": 1, "channels": [4, 6], "head": "ce"}, p)
        ed = run / "checkpoints" / "epochs" / f"R_noise_seed{s}"
        ed.mkdir(parents=True)
        for e in range(20):
            # distinct constant per epoch so the SWA slice is verifiable
            torch.save({k: torch.full_like(v, float(e))
                        for k, v in m.state_dict().items()},
                       ed / f"epoch{e:02d}.pt")
    fake_val = r4.R4View(np.zeros((4, 1, 12), dtype=np.float32),
                         np.array([0, 1, 3, 4]))
    monkeypatch.setattr(r4, "_val_r4", lambda: fake_val)
    monkeypatch.setattr(r4, "champion_val_stats",
                        lambda run: {"directional": 0.0, "stable": 0.0})
    out = r4.stage_select_w(run)
    assert set(out["table"]) == {"W_best", "W_soup", "W_swa"}
    assert (run / "checkpoints" / "W_soup.pt").exists()
    assert out["winner"]["cell"] in {"W_best", "W_soup", "W_swa"}
    # SWA window off-by-one guard: epochs hold constants 0..19, so the
    # last-5 average must be exactly mean(15..19) = 17.0
    swa0 = torch.load(run / "checkpoints" / "W_swa_seed0.pt",
                      map_location="cpu", weights_only=True)["state_dict"]
    assert all(torch.allclose(v, torch.full_like(v, 17.0)) for v in swa0.values())


def test_stage_w_swa_skipped_without_epochs(tmp_path, monkeypatch):
    import torch

    import experiments.accuracy_sweep_r4 as r4
    from training.model import TrendCNN

    run = tmp_path
    a = r4.arm_r4("R_base", "ce")
    (run / "selection").mkdir()
    (run / "selection" / "R.json").write_text(json.dumps(
        {"winner": a, "champion_val": {"directional": 0.5, "stable": 0.5}}))
    for s in r4.ARM_SEEDS_R4:
        m = TrendCNN(num_classes=5, channels=(4, 6))
        p = r4._ckpt_r4(run, a, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": m.state_dict(), "arch": "cnn",
                    "in_channels": 1, "channels": [4, 6], "head": "ce"}, p)
    fake_val = r4.R4View(np.zeros((4, 1, 12), dtype=np.float32),
                         np.array([0, 1, 3, 4]))
    monkeypatch.setattr(r4, "_val_r4", lambda: fake_val)
    monkeypatch.setattr(r4, "champion_val_stats",
                        lambda run: {"directional": 0.0, "stable": 0.0})
    out = r4.stage_select_w(run)
    assert out["table"]["W_swa"] == {"skipped": "no epoch history"}


def test_boot_seed_wired_into_gates(monkeypatch):
    import experiments.accuracy_sweep_r4 as r4
    captured = {}

    def fake_boot(y, pids, pairs, metric, n_boot, seed):
        captured["seed"] = seed
        return {"delta": 0.01, "ci95": [0.001, 0.02]}
    monkeypatch.setattr(
        "experiments.analyze_calibration_coverage.paired_cluster_bootstrap",
        fake_boot)
    y = np.array([0, 1, 2, 3, 4] * 4)
    pids = np.array(["a", "b"] * 10)
    preds = y.copy()
    out = r4.gate_results_r4(y, pids, preds, preds, preds, n_boot=10)
    assert captured["seed"] == r4.BOOT_SEED_R4
    assert out["gate1_beats_champion"] and out["gate3_int8_within_half_pp_of_float"]


def test_evaluate_refuses_without_protocol(monkeypatch):
    import experiments.accuracy_sweep_r4 as r4
    monkeypatch.setattr(r4, "protocol_is_committed", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        r4.stage_evaluate("sweep-r4-test-nonexistent")


def test_tflite_width_param_signature():
    # pure signature check: callable takes (path, x3, width); no litert import
    import inspect

    import experiments.accuracy_sweep_r4 as r4
    sig = inspect.signature(r4._tflite_outputs_r4)
    assert list(sig.parameters) == ["path", "x3", "width"]


def test_winner_ckpts_cells(tmp_path):
    import experiments.accuracy_sweep_r4 as r4
    a = r4.arm_r4("R_noise", "ce", aug="noise")
    (tmp_path / "selection").mkdir()
    for cell, best, expected in [
            ("W_soup", "soup", [("soup", "W_soup.pt")]),
            ("W_best", 3, [(f"seed{s}", f"R_noise_seed{s}.pt") for s in range(5)]),
            ("W_swa", 1, [(f"seed{s}", f"W_swa_seed{s}.pt") for s in range(5)])]:
        (tmp_path / "selection" / "W.json").write_text(json.dumps(
            {"winner": dict(a, cell=cell, best_seed=best)}))
        got = [(label, p.name) for label, p in r4._winner_ckpts(tmp_path)]
        assert got == expected, cell


def test_rate_views_collate_float32_for_mps():
    # MPS cannot take float64; default collate over python floats makes
    # float64, so rate targets must come out of views as float32 tensors
    import torch
    from torch.utils.data import DataLoader

    from experiments.accuracy_sweep_r4 import NoisyLevelView, RateView
    v = RateView(np.zeros((4, 1, 12), dtype=np.float32),
                 np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32))
    _, y = next(iter(DataLoader(v, batch_size=4)))
    assert y.dtype == torch.float32
    nv = NoisyLevelView(np.full((2, 12), 100.0, dtype=np.float32),
                        np.array([0.5, 1.0], dtype=np.float32),
                        sigma=1.0, seed=0, float_targets=True)
    _, y = next(iter(DataLoader(nv, batch_size=2)))
    assert y.dtype == torch.float32


def test_analyze_no_read_flows(tmp_path, monkeypatch):
    # identity/abort outcomes publish without eval dirs (review finding)
    from pathlib import Path as _P

    import experiments.accuracy_sweep_r4 as r4
    monkeypatch.setattr(r4, "_run_dir", lambda rid: str(tmp_path))
    exp0 = tmp_path / "exp0.json"
    exp0.write_text(json.dumps({"abort": False, "headroom": 0.42}))
    monkeypatch.setattr(r4, "EXP0_PATH", exp0)
    # abort flow: nothing but exp0 exists
    out = r4.stage_analyze("sweep-r4-test")
    assert out["exp0"]["headroom"] == 0.42
    assert out["selection_trail"] == {} and "components" not in out
    assert out["ship"]["all_gates_pass"] is False
    # identity flow: selections exist, still no eval dirs
    (tmp_path / "selection").mkdir()
    for s in ("H", "R", "W"):
        (tmp_path / "selection" / f"{s}.json").write_text(json.dumps(
            {"stage": s, "winner": r4.arm_r4("R_base", "ce")
             if s != "W" else dict(r4.arm_r4("R_base", "ce"),
                                   cell="W_best", best_seed=1)}))
    (tmp_path / "smoke.json").write_text(json.dumps({"frozen_batch": 512}))
    out = r4.stage_analyze("sweep-r4-test")
    assert set(out["selection_trail"]) == {"H", "R", "W"}
    assert out["smoke"]["frozen_batch"] == 512
    assert "components" not in out and "pedap_zero_shot" not in out
    assert (_P(tmp_path) / "analysis" / "sweep_r4_aggregate.json").exists()


def test_published_r4_jsons_leak_free():
    # the plan's leak walker, pinned as a test instead of prose
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1] / "paper" / "results" / "calibration-coverage"
    for name in ("exp0_sweep_r4.json", "accuracy_sweep_r4.json"):
        blob = (root / name).read_text()
        for pat in ('"participants"', '"keys"', '"pids"', '"row_ids"',
                    '"anchor_times"', '"labels": ['):
            assert pat not in blob, (name, pat)
