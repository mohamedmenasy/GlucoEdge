"""Hand-computable checks for the panel-revision re-analyses
(plan 2026-10-16-paper-panel-revision.md, Task 1). Synthetic arrays only."""
import numpy as np
import pytest

from experiments.analyze_calibration_coverage import (
    confusion_counts, minority_macro_recall, paired_cluster_bootstrap)
from experiments.panel_revision import (
    censor_shares, cluster_bootstrap_stat, confusion_summary,
    level_stratified_recall, mean_contributions, participant_contributions,
    sign_flip_test)


def _mmr(y, p):
    return minority_macro_recall(confusion_counts(y, p))


def _random_three_participants(seed=0, n=300):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    pids = np.repeat(["a", "b", "c"], n // 3)
    a = np.where(rng.random(n) < 0.6, y, rng.integers(0, 5, n))
    b = np.where(rng.random(n) < 0.4, y, rng.integers(0, 5, n))
    return y, pids, a, b


# --- per-participant contributions and sign-flip test -------------------------


def test_contributions_sum_to_pooled_difference():
    y, pids, a, b = _random_three_participants()
    contribs = participant_contributions(y, pids, a, b)
    assert set(contribs) == {"a", "b", "c"}
    assert sum(contribs.values()) == pytest.approx(
        100 * (_mmr(y, a) - _mmr(y, b)), abs=1e-9)


def test_mean_contributions_sum_to_mean_pooled_difference():
    y, pids, a, b = _random_three_participants(seed=1)
    _, _, c, d = _random_three_participants(seed=2)
    pairs = [(a, b), (c, d)]
    contribs = mean_contributions(y, pids, pairs)
    expected = np.mean([_mmr(y, p) - _mmr(y, q) for p, q in pairs])
    assert sum(contribs.values()) == pytest.approx(100 * expected, abs=1e-9)


def test_sign_flip_all_positive_six():
    out = sign_flip_test([1, 2, 3, 4, 5, 6])
    assert out["observed"] == 21
    assert out["n_patterns"] == 64
    # Only all-plus and all-minus reach |sum| >= 21.
    assert out["p_two_sided"] == pytest.approx(2 / 64)


def test_sign_flip_symmetric():
    assert sign_flip_test([1, -1])["p_two_sided"] == 1.0


# --- bootstrap for arbitrary statistics --------------------------------------


def test_cluster_bootstrap_reproducible():
    y, pids, a, b = _random_three_participants(seed=3, n=600)
    pids = np.repeat(np.arange(12), 50)

    def stat(yy, rows):
        return _mmr(yy, a[rows]) - _mmr(yy, b[rows])

    o1 = cluster_bootstrap_stat(y, pids, stat, n_boot=200, seed=7)
    o2 = cluster_bootstrap_stat(y, pids, stat, n_boot=200, seed=7)
    assert o1["ci95"] == o2["ci95"]
    assert o1["ci95"][0] <= o1["point"] <= o1["ci95"][1]


def test_cluster_bootstrap_matches_paired_bootstrap():
    # Same draws, percentile and undefined rule as the published contrasts.
    y, _, a, b = _random_three_participants(seed=4, n=600)
    pids = np.repeat(np.arange(12), 50)
    ref = paired_cluster_bootstrap(y, pids, [(a, b)], n_boot=300, seed=11)
    out = cluster_bootstrap_stat(
        y, pids, lambda yy, rows: _mmr(yy, a[rows]) - _mmr(yy, b[rows]),
        n_boot=300, seed=11)
    assert out["point"] == ref["delta"]
    assert out["ci95"] == ref["ci95"]
    assert out["undefined_fraction"] == ref["undefined_fraction"]


# --- censoring, confusion profile, level bands -------------------------------


def test_censor_shares_counts():
    windows = np.full((4, 12), 120.0, dtype=np.float32)
    windows[0, 3] = 401.0
    windows[2, 11] = 401.0
    labels = np.array([0, 2, 2, 4])
    out = censor_shares(windows, labels)
    assert out["inputs"]["window_fraction"] == 0.5
    assert out["inputs"]["by_true_class"]["2"] == 0.5
    assert "endpoint" not in out
    out = censor_shares(windows, labels, endpoint=np.array([39.0, 39.0, 100.0, 100.0]))
    assert out["endpoint"]["window_fraction"] == 0.5
    assert out["inputs_or_endpoint"]["window_fraction"] == 0.75


def test_confusion_summary_reversal():
    # Three true falling_fast windows, one predicted rising_fast.
    y_true = np.array([0, 0, 0, 1, 2, 3, 4])
    y_pred = np.array([4, 0, 0, 1, 2, 3, 4])
    out = confusion_summary(y_true, y_pred)
    assert out["matrix"][0][4] == 1
    assert out["reversal_share"]["falling_fast"] == pytest.approx(1 / 3)
    assert out["reversal_share"]["rising_fast"] == 0.0
    assert out["recall_sensitivity"]["falling_fast"] == pytest.approx(2 / 3)
    assert out["precision_ppv"]["rising_fast"] == pytest.approx(1 / 2)
    assert out["adjacent_error_share"] == 0.0  # the one error skips 3 classes


def test_level_bands():
    out = level_stratified_recall(np.array([0, 1, 3, 4]), np.array([0, 1, 3, 4]),
                                  np.array([60.0, 100.0, 200.0, 300.0]))
    assert list(out) == ["<70", "70-180", "180-250", ">=250"]
    assert [b["n_windows"] for b in out.values()] == [1, 1, 1, 1]
    assert out["<70"]["support"][0] == 1 and out[">=250"]["support"][4] == 1
