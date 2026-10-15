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


def _fake_stats(stats):
    """stats: {(arm_name, seed): (dir, stable)}"""
    def f(run, a, seed):
        return stats[(a["name"], seed)]
    return f


def test_select_floor_is_2pp_and_tie_prefers_cheap(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.533, "stable": 0.5485})
    # R_val mean stable sits between -2pp and -5pp below champion: with the
    # r5 floor it is DISQUALIFIED (under r3's -5pp it would have survived)
    stats = {("R_base", s): (0.530, 0.548) for s in range(5)}
    stats.update({("R_val", s): (0.540, 0.520) for s in range(5)})  # -2.85pp
    monkeypatch.setattr(r5, "_val_stats_r5", _fake_stats(stats))
    monkeypatch.setattr(r5, "_flag_for", lambda run, a, seed: False)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    out = r5.stage_select("sweep-r5-test", "R")
    assert out["stable_floor"] == pytest.approx(0.5485 - 0.02)
    assert out["table"]["R_val"]["disqualified"] is True
    assert out["winner"]["name"] == "R_base"


def test_select_all_dq_refuses(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.533, "stable": 0.5485})
    stats = {(n, s): (0.54, 0.40) for n in ("R_base", "R_val") for s in range(5)}
    monkeypatch.setattr(r5, "_val_stats_r5", _fake_stats(stats))
    monkeypatch.setattr(r5, "_flag_for", lambda run, a, seed: False)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    with pytest.raises(SystemExit, match="REFUSED"):
        r5.stage_select("sweep-r5-test", "R")


def test_stage_s_tie_band_and_floor(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "champion_val_stats",
                        lambda run: {"directional": 0.533, "stable": 0.5485})
    (tmp_path / "selection").mkdir(parents=True)
    (tmp_path / "selection" / "H.json").write_text(json.dumps(
        {"winner": r5.arm_r5("H_flat", "flat", "val")}))
    # seeds 1 and 2 inside the 0.1pp band: lowest seed wins (r3's deviation
    # would have argmaxed to seed 2); seed 0 fails the per-seed floor
    stats = {("H_flat", 0): (0.560, 0.500),
             ("H_flat", 1): (0.5366, 0.536),
             ("H_flat", 2): (0.5367, 0.526),
             ("H_flat", 3): (0.530, 0.54),
             ("H_flat", 4): (0.531, 0.53)}
    monkeypatch.setattr(r5, "_val_stats_r5", _fake_stats(stats))
    monkeypatch.setattr(r5, "_flag_for", lambda run, a, seed: seed == 4)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    out = r5.stage_select("sweep-r5-test", "S")
    assert out["best_seed"] == 1
    assert out["table"]["per_seed"]["4"]["epoch_floor_flagged"] is True


def test_convert_artifact_naming_contract(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection" / "S.json").write_text(json.dumps(
        {"winner": r5.arm_r5("H_flat", "flat", "val"), "best_seed": 2}))
    (tmp_path / "selections").mkdir()
    for stem in ("sequential_200_d0", "uniform_random_200_d101"):
        (tmp_path / "selections" / f"{stem}.json").write_text(
            json.dumps({"indices": [0, 1]}))
    calls = []
    monkeypatch.setattr(r5, "_convert_seed",
                        lambda run, winner, seed: calls.append(seed))
    r5.stage_convert("sweep-r5-test")
    assert calls == [0, 1, 2, 3, 4]
    # candidate artifact paths the evaluate/analyze stages will look for
    fl, i8 = r5._candidate_arts(tmp_path)
    assert fl.name == "winner_seed2_float.tflite"
    assert i8.name == "winner_seed2_int8_uniform_random_200_d101.tflite"


def test_identity_guard_requires_both_artifacts_equal(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection" / "S.json").write_text(json.dumps(
        {"winner": r5.arm_r5("H_gap", "gap", "val"), "best_seed": 0}))
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "winner_seed0_float.tflite").write_bytes(b"FLOAT")
    (art / "winner_seed0_int8_uniform_random_200_d101.tflite").write_bytes(b"INT8")
    champ_f, champ_i = tmp_path / "cf.tflite", tmp_path / "ci.tflite"
    champ_f.write_bytes(b"FLOAT")
    champ_i.write_bytes(b"INT8")
    monkeypatch.setattr(r5, "_champion_paths",
                        lambda: {"float": champ_f, "int8": champ_i})
    assert r5.identity_with_champion(tmp_path)["identical"] is True
    champ_i.write_bytes(b"DIFFERENT")  # one-of-two equal is NOT identity
    assert r5.identity_with_champion(tmp_path)["identical"] is False


def test_rtcgm_windows_filters_admitted_and_guards_drift(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    cache = tmp_path / "RT_CGM.npz"
    np.savez_compressed(cache,
                        w=np.zeros((6, 12), dtype=np.float32),
                        y=np.arange(6, dtype=np.int64) % 5,
                        pids=np.array(["a", "a", "b", "b", "c", "c"]),
                        hours=np.zeros(6))
    monkeypatch.setattr(r5, "RTCGM_CACHE", cache)
    monkeypatch.setattr(r5, "_admitted_rtcgm_r5", lambda: {"a", "b", "c"})
    x, y, pids = r5._rtcgm_windows()
    assert len(x) == 6 and x.shape[1] == 12
    # admit-list drift: keep count collapses below half the cache -> refuse
    monkeypatch.setattr(r5, "_admitted_rtcgm_r5", lambda: {"a"})
    with pytest.raises(SystemExit, match="REFUSED"):
        r5._rtcgm_windows()
    monkeypatch.setattr(r5, "_admitted_rtcgm_r5", lambda: {"zz"})
    with pytest.raises(SystemExit, match="REFUSED"):
        r5._rtcgm_windows()


def test_evaluate_refuses_without_committed_protocol(monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "protocol_is_committed", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        r5.stage_evaluate("sweep-r5-test")


def test_evaluate_identity_skips_the_read(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "protocol_is_committed", lambda p: True)
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    monkeypatch.setattr(r5, "identity_with_champion", lambda run: {
        "identical": True, "candidate_float_sha256": "x",
        "candidate_int8_sha256": "y", "champion_float_sha256": "x",
        "champion_int8_sha256": "y"})

    def boom():
        raise AssertionError("labels were read despite identity")
    monkeypatch.setattr(r5, "_rtcgm_windows", boom)
    r5.stage_evaluate("sweep-r5-test")
    assert json.loads((tmp_path / "eval" / "identity.json").read_text())[
        "identical"] is True
    assert not (tmp_path / "eval" / "rtcgm").exists()


def test_gates_r5_boot_seed_and_none_ci(monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    captured = {}

    def fake_boot(y, pids, pairs, metric, n_boot, seed):
        captured["seed"] = seed
        return {"delta": 0.01, "ci95": None, "undefined_fraction": 0.2,
                "n_boot": n_boot, "seed": seed}
    import experiments.analyze_calibration_coverage as acc
    monkeypatch.setattr(acc, "paired_cluster_bootstrap", fake_boot)
    y = np.array([0, 1, 2, 3, 4] * 4)
    pids = np.array(["a", "b"] * 10)
    g = r5.gate_results_r5(y, pids, y.copy(), y.copy(), y.copy(), n_boot=7)
    assert captured["seed"] == r5.BOOT_SEED_R5 == 20261015
    assert g["gate1_beats_champion"] is False  # undefined CI never passes


def test_gates_r5_pass_case(monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    import experiments.analyze_calibration_coverage as acc
    monkeypatch.setattr(acc, "paired_cluster_bootstrap",
                        lambda *a, **k: {"delta": 0.008,
                                         "ci95": [0.001, 0.015],
                                         "undefined_fraction": 0.0,
                                         "n_boot": k.get("n_boot"),
                                         "seed": k.get("seed")})
    y = np.array([0, 1, 2, 3, 4] * 4)
    pids = np.array(["a", "b"] * 10)
    g = r5.gate_results_r5(y, pids, y.copy(), y.copy(), y.copy())
    assert g["gate1_beats_champion"] is True
    assert g["gate2_stable_floor"] is True
    assert g["gate3_int8_within_half_pp_of_float"] is True


def test_analyze_identity_flow(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    (tmp_path / "selection").mkdir()
    for s in ("R", "H"):
        (tmp_path / "selection" / f"{s}.json").write_text(json.dumps(
            {"stage": s, "winner": r5.arm_r5("R_base", "gap", "base")}))
    (tmp_path / "selection" / "S.json").write_text(json.dumps(
        {"stage": "S", "winner": r5.arm_r5("R_base", "gap", "base"),
         "best_seed": 1}))
    (tmp_path / "smoke.json").write_text(json.dumps({"frozen_batch": 512}))
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "identity.json").write_text(json.dumps(
        {"identical": True, "candidate_int8_sha256": "y"}))
    out = r5.stage_analyze("sweep-r5-test")
    assert out["ship"]["all_gates_pass"] is False
    assert "identity" in out["ship"]["outcome"]
    assert "components" not in out
    assert (tmp_path / "analysis" / "sweep_r5_aggregate.json").exists()


def test_protocol_has_no_row_level_fields(tmp_path, monkeypatch):
    import experiments.accuracy_sweep_r5 as r5
    monkeypatch.setattr(r5, "_run_dir", lambda rid: str(tmp_path))
    monkeypatch.setattr(r5, "PROTOCOL_R5", tmp_path / "protocol_sweep_r5.json")
    monkeypatch.setattr(r5, "_champion_paths", lambda: {
        "float": tmp_path / "cf", "int8": tmp_path / "ci"})
    (tmp_path / "cf").write_bytes(b"F")
    (tmp_path / "ci").write_bytes(b"I")
    monkeypatch.setattr(r5, "_ckpt_r5",
                        lambda run, a, s: tmp_path / "cf")  # alias sha stub
    (tmp_path / "selection").mkdir()
    for s, extra in (("R", {}), ("H", {}), ("S", {"best_seed": 2})):
        (tmp_path / "selection" / f"{s}.json").write_text(json.dumps(
            {"stage": s, "winner": r5.arm_r5("H_gap", "gap", "val"), **extra}))
    (tmp_path / "smoke.json").write_text(json.dumps({"frozen_batch": 512}))
    out = r5.stage_protocol("sweep-r5-test")
    blob = (tmp_path / "protocol_sweep_r5.json").read_text()
    for pat in ('"participants"', '"keys"', '"pids"', '"anchor_times"'):
        assert pat not in blob
    assert out["boot_seed"] == 20261015
    assert "disclosure" in out and "D_rtcgm" in out["disclosure"]


def test_main_stage_choices():
    import inspect

    import experiments.accuracy_sweep_r5 as r5
    src = inspect.getsource(r5.main)
    for stage in ("smoke", "train", "select", "calib", "convert",
                  "protocol", "evaluate", "analyze"):
        assert f'"{stage}"' in src


def test_published_r5_jsons_leak_free():
    # row-level identifiers never leave gitignored results/
    from pathlib import Path as _P
    root = (_P(__file__).resolve().parents[1] / "paper" / "results"
            / "calibration-coverage")
    names = ("protocol_sweep_r5.json", "accuracy_sweep_r5.json")
    published = [n for n in names if (root / n).exists()]
    if not published:
        pytest.skip("r5 not published yet")
    for name in published:
        blob = (root / name).read_text()
        for pat in ('"participants": [', '"keys"', '"pids"', '"row_ids"',
                    '"anchor_times"', '"labels": ['):
            assert pat not in blob, (name, pat)
