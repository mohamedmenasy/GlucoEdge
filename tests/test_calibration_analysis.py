"""Hand-computable checks for the calibration-coverage analysis
(plan Section 4) and the quantization controls (Section 3 / H3)."""
import numpy as np
import pytest

from experiments.analyze_calibration_coverage import (
    accuracy, confusion_counts, macro_recall, minority_macro_recall,
    paired_cluster_bootstrap, recalls, resample_rows)
from experiments.calibration_coverage import clip_only, qdq, qdq_from_detail

# --- metrics -----------------------------------------------------------------


def test_confusion_counts_match_hand_computation():
    y_true = np.array([0, 0, 2, 2, 4])
    y_pred = np.array([0, 2, 2, 2, 0])
    cm = confusion_counts(y_true, y_pred, n_classes=5)
    assert cm.shape == (5, 5)
    assert cm[0, 0] == 1 and cm[0, 2] == 1
    assert cm[2, 2] == 2
    assert cm[4, 0] == 1
    assert cm.sum() == 5


def test_recalls_and_aggregates_match_hand_computation():
    # Class 0: 1/2 correct; class 2: 2/2; class 4: 0/1; classes 1,3 absent.
    cm = confusion_counts(np.array([0, 0, 2, 2, 4]), np.array([0, 2, 2, 2, 0]), 5)
    r = recalls(cm)
    assert r[0] == 0.5 and r[2] == 1.0 and r[4] == 0.0
    assert np.isnan(r[1]) and np.isnan(r[3])
    assert accuracy(cm) == pytest.approx(3 / 5)
    # Macro and minority means are undefined while any needed class is absent.
    assert np.isnan(macro_recall(cm))
    assert np.isnan(minority_macro_recall(cm))


def test_minority_macro_recall_averages_the_four_directional_classes():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 3, 4, 2])
    y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 0, 4, 0])
    cm = confusion_counts(y_true, y_pred, 5)
    # recalls: c0 1/2, c1 2/2, c3 1/2, c4 2/2 -> mean 0.75 (stable ignored).
    assert minority_macro_recall(cm) == pytest.approx(0.75)


# --- bootstrap ---------------------------------------------------------------


def test_resample_rows_preserves_whole_participant_clusters():
    pids = np.array(["a", "a", "a", "b", "b", "c"])
    clusters = {p: np.flatnonzero(pids == p) for p in ("a", "b", "c")}
    rng = np.random.default_rng(3)
    for _ in range(20):
        rows = resample_rows(pids, rng)
        # Decompose: the drawn rows must be a concatenation of whole clusters.
        i = 0
        while i < len(rows):
            matched = False
            for p, members in clusters.items():
                k = len(members)
                if np.array_equal(rows[i:i + k], members):
                    i += k
                    matched = True
                    break
            assert matched, f"rows {rows[i:]} do not start with a whole cluster"


def test_identical_predictions_give_zero_delta_and_degenerate_interval():
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(5), 40)  # every cluster holds all five classes
    pred = (y + rng.integers(0, 2, 200)) % 5
    pids = np.repeat(np.arange(10), 20)
    out = paired_cluster_bootstrap(y, pids, [(pred, pred)], n_boot=50, seed=1)
    assert out["delta"] == 0.0
    assert out["ci95"] == [0.0, 0.0]
    assert out["undefined_fraction"] == 0.0


def test_bootstrap_reports_undefined_fraction_and_omits_interval_past_5pct():
    # Participant "solo" is the only source of class 4: draws that omit it
    # have undefined minority recall.
    y = np.array([0, 1, 2, 3] * 10 + [4])
    pred = y.copy()
    pids = np.array(["p0", "p1"] * 20 + ["solo"])
    out = paired_cluster_bootstrap(y, pids, [(pred, pred)], n_boot=200, seed=2)
    assert 0.0 < out["undefined_fraction"] < 1.0
    if out["undefined_fraction"] > 0.05:
        assert out["ci95"] is None


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 5, 300)
    a = (y + rng.integers(0, 2, 300)) % 5
    b = (y + rng.integers(0, 3, 300)) % 5
    pids = np.repeat(np.arange(15), 20)
    o1 = paired_cluster_bootstrap(y, pids, [(a, b)], n_boot=100, seed=7)
    o2 = paired_cluster_bootstrap(y, pids, [(a, b)], n_boot=100, seed=7)
    assert o1 == o2


# --- quantization controls ---------------------------------------------------

SCALE, ZP = np.float32(0.9408126), -128  # the shipped artifact's parameters
LO = SCALE * (-128 - ZP)  # 0.0
HI = SCALE * (127 - ZP)   # 239.907...


def test_qdq_maps_boundaries_and_clips_just_outside():
    x = np.array([LO, HI, LO - 1.0, HI + 1.0, HI - 0.1], dtype=np.float32)
    out = qdq(x, SCALE, ZP)
    assert out[0] == pytest.approx(LO)
    assert out[1] == pytest.approx(HI)
    assert out[2] == pytest.approx(LO)   # below range clips to the bottom code
    assert out[3] == pytest.approx(HI)   # above range clips to the top code
    assert abs(out[4] - (HI - 0.1)) <= SCALE / 2  # inside: rounds to the grid


def test_qdq_rounds_ties_half_to_even():
    # With scale 2 and zero point 0, x=3.0 sits exactly between codes 1 and 2:
    # half-to-even picks code 2 -> 4.0; x=1.0 sits between 0 and 1 -> 0.0.
    out = qdq(np.array([3.0, 1.0], dtype=np.float32), np.float32(2.0), 0)
    assert out[0] == pytest.approx(4.0)
    assert out[1] == pytest.approx(0.0)


def test_clip_only_truncates_range_without_regridding():
    x = np.array([HI + 50.0, HI - 0.1, LO - 5.0], dtype=np.float32)
    out = clip_only(x, SCALE, ZP)
    assert out[0] == pytest.approx(HI)
    assert out[1] == pytest.approx(HI - 0.1)  # untouched: no rounding to grid
    assert out[2] == pytest.approx(LO)


def test_qdq_from_detail_uses_the_artifact_scale_and_zero_point():
    detail = {"dtype": np.int8, "quantization": (2.0, 10)}
    x = np.array([0.0, 500.0], dtype=np.float32)
    out = qdq_from_detail(x, detail)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(2.0 * (127 - 10))  # clipped at the top code
