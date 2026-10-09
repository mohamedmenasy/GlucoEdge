import numpy as np
import pandas as pd

from experiments.split_audit import observation_status, row_owner, window_provenance


def test_window_provenance_assigns_context_rows_to_their_earliest_split():
    # GlucoBench-style layout: val repeats training rows 5-9 as context and
    # test repeats val rows 15-19; rows 30-39 belong to a held-out participant.
    owner = row_owner(
        40,
        train_idx=range(0, 10),
        val_idx=range(5, 20),
        test_idx=range(15, 30),
        held_out_idx=range(30, 40),
    )
    windows = [np.arange(a, a + 5) for a in (0, 3, 8, 16, 22, 31)]

    assert window_provenance(windows, owner) == {
        "windows": 6,
        "target_owner": {"train": 2, "val": 1, "test": 2, "held_out": 1},
        # [8..12] ends in a val row, so only the first two are training copies.
        "training_copies": 2,
    }


def test_observation_status_separates_regridding_from_gap_filling():
    t = lambda minutes: pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minutes)
    readings = pd.DataFrame({
        "id": ["a", "a", "a", "a", "b"],
        # a: 00:05 -> 00:11 is clock drift (no reading missing);
        # 00:11 -> 00:25 skips at least one reading.
        "time": [t(0), t(5), t(11), t(25), t(10)],
    })
    grid = pd.DataFrame({"id": ["a"] * 6, "time": [t(m) for m in range(0, 30, 5)]})

    on_reading, no_gap_fill = observation_status(grid, readings)

    # b's 00:10 reading must not mark a's 00:10 grid row as observed.
    assert on_reading.tolist() == [True, True, False, False, False, True]
    assert no_gap_fill.tolist() == [True, True, True, False, False, True]
