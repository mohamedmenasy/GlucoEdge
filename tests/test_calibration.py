"""Sampler contracts for the calibration-coverage study (plan Section 3).

Everything here runs on synthetic pools; no GlucoBench data and no LiteRT
toolchain import."""
import numpy as np
import pandas as pd
import pytest

from conversion.calibration import build_pool, pool_hash, select
from training.dataset import GlucoseTrendDataset
from training.labeling import FIVE_CLASSES

STRATEGIES = ("sequential", "uniform_random",
              "participant_balanced_random", "participant_balanced_range_coverage")


def _pool(n_participants=8, windows_each=40, seed=7):
    """Synthetic pool dict: participants with varied input ranges."""
    rng = np.random.default_rng(seed)
    parts, mins, maxs, times, segs = [], [], [], [], []
    for p in range(n_participants):
        base = 80.0 + 30.0 * p
        for w in range(windows_each):
            lo = base + rng.uniform(0, 40)
            parts.append(float(p))
            mins.append(lo)
            maxs.append(lo + rng.uniform(5, 120))
            times.append(np.datetime64("2024-01-01") + np.timedelta64(5 * w, "m"))
            segs.append(0)
    return {
        "participant": np.array(parts),
        "segment": np.array(segs),
        "anchor_time": np.array(times),
        "input_min": np.array(mins, dtype=np.float32),
        "input_max": np.array(maxs, dtype=np.float32),
        "window_index": np.arange(len(parts)),
    }


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_selection_is_repeatable_unique_and_exact(strategy):
    pool = _pool()
    a = select(pool, strategy, budget=50, seed=101)
    b = select(pool, strategy, budget=50, seed=101)
    assert a["indices"] == b["indices"]
    assert len(a["indices"]) == 50
    assert len(set(a["indices"])) == 50
    assert a["strategy"] == strategy
    assert a["seed"] == 101
    assert a["actual_count"] == 50
    assert a["pool_sha256"] == pool_hash(pool)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_small_budget_is_nested_in_large_budget(strategy):
    pool = _pool()
    small = select(pool, strategy, budget=50, seed=101)
    large = select(pool, strategy, budget=200, seed=101)
    assert large["indices"][:50] == small["indices"]


def test_different_seeds_change_stochastic_selections_but_not_sequential():
    pool = _pool()
    for strategy in STRATEGIES[1:]:
        assert (select(pool, strategy, 50, seed=101)["indices"]
                != select(pool, strategy, 50, seed=102)["indices"])
    assert (select(pool, "sequential", 50, seed=101)["indices"]
            == select(pool, "sequential", 50, seed=102)["indices"])


def test_sequential_follows_participant_segment_time_order():
    pool = _pool(n_participants=2, windows_each=5)
    sel = select(pool, "sequential", budget=7, seed=101)["indices"]
    order = np.lexsort((pool["anchor_time"], pool["segment"], pool["participant"]))
    assert sel == [int(pool["window_index"][i]) for i in order[:7]]


def test_balanced_arms_share_participant_allocation():
    pool = _pool(n_participants=5, windows_each=30)
    for budget in (20, 60):
        rand = select(pool, "participant_balanced_random", budget, seed=101)
        cover = select(pool, "participant_balanced_range_coverage", budget, seed=101)
        by_part = lambda sel: dict(zip(*np.unique(
            pool["participant"][np.searchsorted(pool["window_index"], sel["indices"])],
            return_counts=True)))
        assert by_part(rand) == by_part(cover)


def test_round_robin_skips_exhausted_participants():
    # Participant 0 has only 2 windows; a budget of 12 over 3 participants
    # must still fill, taking the shortfall from the others.
    rng = np.random.default_rng(0)
    counts = {0.0: 2, 1.0: 10, 2.0: 10}
    parts = np.concatenate([np.full(c, p) for p, c in counts.items()])
    n = len(parts)
    pool = {
        "participant": parts,
        "segment": np.zeros(n, dtype=int),
        "anchor_time": np.array([np.datetime64("2024-01-01") + np.timedelta64(i, "m") for i in range(n)]),
        "input_min": rng.uniform(80, 200, n).astype(np.float32),
        "input_max": rng.uniform(200, 400, n).astype(np.float32),
        "window_index": np.arange(n),
    }
    for strategy in ("participant_balanced_random", "participant_balanced_range_coverage"):
        sel = select(pool, strategy, budget=12, seed=101)
        got = pool["participant"][np.searchsorted(pool["window_index"], sel["indices"])]
        by_part = dict(zip(*np.unique(got, return_counts=True)))
        assert by_part[0.0] == 2  # exhausted, not revisited
        assert by_part[1.0] == by_part[2.0] == 5
        assert len(set(sel["indices"])) == 12


def test_degenerate_quantile_edges_collapse_to_fewer_cells():
    # Every window has identical min and max: all range cells coincide.
    n = 30
    pool = {
        "participant": np.repeat(np.arange(3, dtype=float), 10),
        "segment": np.zeros(n, dtype=int),
        "anchor_time": np.array([np.datetime64("2024-01-01") + np.timedelta64(i, "m") for i in range(n)]),
        "input_min": np.full(n, 100.0, dtype=np.float32),
        "input_max": np.full(n, 150.0, dtype=np.float32),
        "window_index": np.arange(n),
    }
    sel = select(pool, "participant_balanced_range_coverage", budget=9, seed=101)
    assert len(set(sel["indices"])) == 9
    edges = sel["range_cells"]
    assert len(set(edges["input_min_edges"])) == len(edges["input_min_edges"])  # deduplicated


def test_budget_exceeding_pool_raises_infeasibility_error():
    pool = _pool(n_participants=2, windows_each=5)
    for strategy in STRATEGIES:
        with pytest.raises(ValueError, match="[Ii]nfeasible"):
            select(pool, strategy, budget=11, seed=101)


def test_selection_ignores_future_labels():
    # build_pool reads inputs and provenance metadata only: shuffling the
    # dataset's labels must not change any selection.
    rows = [{"id": f"p{j}", "id_segment": 0,
             "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
             "gl": 80.0 + 7.0 * ((i * (j + 3)) % 11)}
            for j in range(3) for i in range(20)]
    ds = GlucoseTrendDataset(pd.DataFrame(rows), classes=FIVE_CLASSES,
                             input_length=3, horizon=2, horizon_minutes=10.0)
    eligible = list(range(len(ds)))
    pool = build_pool(ds, eligible)
    before = {s: select(pool, s, 10, seed=101)["indices"] for s in STRATEGIES}

    ds.labels = list(np.random.default_rng(0).permutation(ds.labels))
    pool_shuffled = build_pool(ds, eligible)
    assert pool_hash(pool_shuffled) == pool_hash(pool)
    for s in STRATEGIES:
        assert select(pool_shuffled, s, 10, seed=101)["indices"] == before[s]


def test_build_pool_computes_input_extremes_and_window_keys():
    rows = [{"id": "p1", "id_segment": 0,
             "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
             "gl": [100, 90, 120, 80, 130, 95, 101][i]}
            for i in range(7)]
    ds = GlucoseTrendDataset(pd.DataFrame(rows), classes=FIVE_CLASSES,
                             input_length=3, horizon=2, horizon_minutes=10.0)
    pool = build_pool(ds, [0, 2])
    assert list(pool["window_index"]) == [0, 2]
    # Window 0 inputs: 100, 90, 120; window 2 inputs: 120, 80, 130.
    assert list(pool["input_min"]) == [90.0, 80.0]
    assert list(pool["input_max"]) == [120.0, 130.0]
    assert list(pool["participant"]) == ["p1", "p1"]
