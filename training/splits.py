"""Split-role and observation-mask rules for provenance-aware windows
(calibration-coverage plan, Section 2 + Amendment 1).

Pure functions over GlucoBench-formatter-style frames; GlucoBench itself is
never imported here. `row_owner` and `observation_status` moved here from
`experiments.split_audit`, which re-exports them.
"""
import numpy as np

OWNERS = ("train", "val", "test", "held_out")
# Minute-rounded readings more than 1.5 sensor intervals apart have at least
# one missing reading between them (Amendment 1's 7.5-minute threshold).
MAX_ADJACENT_GAP = np.timedelta64(450, "s")


def row_owner(n_rows, train_idx, val_idx, test_idx, held_out_idx):
    """Split that scores each formatter row. GlucoBench prefixes val/test
    frames with context rows from the preceding split, so a row listed in
    several splits belongs to the earliest one."""
    owner = np.full(n_rows, "", dtype=object)
    for name, idx in (("held_out", held_out_idx), ("test", test_idx), ("val", val_idx), ("train", train_idx)):
        owner[np.asarray(idx, dtype=int)] = name
    return owner


def observation_status(grid, readings):
    """Per grid row (by position): is it an original reading, and is it free
    of gap filling - an original reading, or between two consecutive readings
    of the same participant at most MAX_ADJACENT_GAP apart? Both frames hold
    `id` and minute-rounded `time`."""
    grid = grid.reset_index(drop=True)
    on_reading = np.zeros(len(grid), dtype=bool)
    no_gap_fill = np.zeros(len(grid), dtype=bool)
    reading_times = {pid: np.sort(g["time"].to_numpy()) for pid, g in readings.groupby("id")}
    for pid, g in grid.groupby("id"):
        times, t = reading_times[pid], g["time"].to_numpy()
        pos = np.searchsorted(times, t)  # first reading at or after t
        nxt = times[np.minimum(pos, len(times) - 1)]
        prv = times[np.maximum(pos - 1, 0)]
        hit = (pos < len(times)) & (nxt == t)
        bracketed = (pos > 0) & (pos < len(times)) & (nxt - prv <= MAX_ADJACENT_GAP)
        on_reading[g.index] = hit
        no_gap_fill[g.index] = hit | bracketed
    return on_reading, no_gap_fill


def scored_window_indices(ds, owner, split):
    """Indices of `ds` windows scored by `split`: the anchor (last input row)
    and every horizon row must be owned by `split`. Earlier-split rows may
    appear among the inputs as causal history; a window whose target region
    crosses an ownership boundary is scored nowhere."""
    scored = []
    for i, rows in enumerate(ds.row_ids):
        if (owner[rows[ds.input_length - 1:]] == split).all():
            scored.append(i)
    return scored


def gap_free_window_indices(ds, no_gap_fill):
    """Indices of `ds` windows whose 12 input rows and future endpoint are
    all free of gap filling. Intermediate horizon steps are not label
    inputs and are not checked."""
    kept = []
    for i, rows in enumerate(ds.row_ids):
        checked = np.r_[rows[:ds.input_length], rows[-1]]
        if no_gap_fill[checked].all():
            kept.append(i)
    return kept
