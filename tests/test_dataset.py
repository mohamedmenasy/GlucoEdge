import pandas as pd
import torch

from training.dataset import GlucoseTrendDataset
from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP


def _make_df(segment_lengths, start_value=100.0, step=0.0):
    """Synthetic single-patient dataframe. Each entry in segment_lengths
    becomes its own id_segment; values increase by `step` per row within
    a segment, restarting at start_value at the top of each segment."""
    rows = []
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    for seg_id, length in enumerate(segment_lengths):
        for i in range(length):
            rows.append({
                "id": "p1",
                "id_segment": seg_id,
                "time": t0 + pd.Timedelta(minutes=5 * i),
                "gl": start_value + step * i,
            })
    return pd.DataFrame(rows)


def test_windows_never_cross_segment_boundary():
    # Two 10-point segments. span = input_length(12) + horizon(3) = 15, so
    # neither segment alone can produce a window. If the boundary were
    # ignored and this were treated as one 20-point run, it would produce
    # 20 - 15 + 1 = 6 windows instead.
    df = _make_df([10, 10])
    dataset = GlucoseTrendDataset(df, classes=FIVE_CLASSES)
    assert len(dataset) == 0


def test_sample_count_for_known_segment_length():
    # One 17-point segment: span = 15, so valid windows = 17 - 15 + 1 = 3.
    df = _make_df([17])
    dataset = GlucoseTrendDataset(df, classes=FIVE_CLASSES)
    assert len(dataset) == 3


def test_window_and_label_alignment():
    # One 15-point segment (exactly span), so exactly 1 window.
    # values[k] = 100 + 20*k -> last window point (idx 11) = 320,
    # future point (idx 14, 3 steps/15min later) = 380,
    # rate = (380 - 320) / 15 = 4.0 mg/dL/min -> rising_fast.
    df = _make_df([15], start_value=100.0, step=20.0)
    dataset = GlucoseTrendDataset(df, classes=FIVE_CLASSES)
    assert len(dataset) == 1

    x, y = dataset[0]
    assert x.shape == (1, 12)
    expected_window = [100.0 + 20.0 * k for k in range(12)]
    assert torch.allclose(x.squeeze(0), torch.tensor(expected_window, dtype=torch.float32))
    assert FIVE_CLASSES[y] == "rising_fast"


def test_row_ids_trace_windows_to_time_ordered_source_rows():
    # Shuffled rows with offset index labels, like a GlucoBench split frame:
    # row_ids must follow time order and keep the frame's own index labels.
    df = _make_df([16], step=1.0)
    df.index = df.index + 1000
    df = df.sample(frac=1.0, random_state=0)
    dataset = GlucoseTrendDataset(df, classes=FIVE_CLASSES)

    assert [list(rows) for rows in dataset.row_ids] == [
        list(range(1000, 1015)),
        list(range(1001, 1016)),
    ]
    for i, rows in enumerate(dataset.row_ids):
        x, _ = dataset[i]
        expected = torch.tensor(df.loc[rows[:12], "gl"].to_numpy(), dtype=torch.float32)
        assert torch.equal(x.squeeze(0), expected)


def test_collapse_map_reduces_to_three_classes():
    df = _make_df([15], start_value=100.0, step=20.0)  # same rising_fast case
    dataset = GlucoseTrendDataset(df, classes=THREE_CLASSES, collapse_map=THREE_CLASS_MAP)
    assert len(dataset) == 1

    _, y = dataset[0]
    assert THREE_CLASSES[y] == "rising"


# --- Provenance metadata and split-role rules (calibration-coverage plan,
# --- Section 2 + Amendment 1) ------------------------------------------------

import numpy as np

from training.splits import gap_free_window_indices, scored_window_indices


def _global_df(participants):
    """Formatter-style global frame with a RangeIndex. `participants` maps
    pid -> (segment lengths, start, step); rows are emitted participant by
    participant so row positions double as GlucoBench-style row ids."""
    rows = []
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    for pid, (segment_lengths, start, step) in participants.items():
        for seg_id, length in enumerate(segment_lengths):
            for i in range(length):
                rows.append({
                    "id": pid,
                    "id_segment": seg_id,
                    "time": t0 + pd.Timedelta(minutes=5 * i),
                    "gl": start + step * i,
                })
    return pd.DataFrame(rows)


def _small_ds(df):
    # span = 3 inputs + 2 horizon = 5 rows, to keep fixtures small.
    return GlucoseTrendDataset(df, classes=FIVE_CLASSES, input_length=3, horizon=2,
                               horizon_minutes=10.0)


def test_window_metadata_aligned_with_windows():
    df = _global_df({"p1": ([7], 100.0, 1.0)})
    ds = _small_ds(df)
    assert len(ds) == 3
    assert len(ds.participants) == len(ds.segments) == 3
    assert len(ds.anchor_times) == len(ds.target_times) == 3
    # First window: rows 0-4, inputs 0-2 (anchor row 2), target row 4.
    assert ds.participants[0] == "p1"
    assert ds.segments[0] == 0
    assert ds.anchor_times[0] == df.loc[2, "time"]
    assert ds.target_times[0] == df.loc[4, "time"]


def test_scored_windows_reject_context_only_targets_and_keep_causal_history():
    # One participant, 10 rows; rows 0-4 train-owned, rows 5-9 val-owned.
    # The val frame repeats rows 2-4 as context, like GlucoBench prefixes.
    df = _global_df({"p1": ([10], 100.0, 1.0)})
    owner = np.array(["train"] * 5 + ["val"] * 5, dtype=object)
    train_ds = _small_ds(df.loc[0:4])
    val_ds = _small_ds(df.loc[2:9])

    # Train frame: one window (rows 0-4), anchor row 2 + horizon rows 3,4
    # all train-owned -> scored in train.
    assert scored_window_indices(train_ds, owner, "train") == [0]

    # Val frame windows start at rows 2,3,4,5. Window at 2 has a train-owned
    # anchor (row 4): context-only target region, scored nowhere. Window at 3
    # has anchor row 5 (val) with train rows 3,4 as causal history -> scored.
    scored_val = scored_window_indices(val_ds, owner, "val")
    assert scored_val == [1, 2, 3]
    assert scored_window_indices(val_ds, owner, "train") == []

    # Scored (participant, segment, target_time) keys stay disjoint.
    train_keys = {(train_ds.participants[i], train_ds.segments[i], train_ds.target_times[i])
                  for i in scored_window_indices(train_ds, owner, "train")}
    val_keys = {(val_ds.participants[i], val_ds.segments[i], val_ds.target_times[i])
                for i in scored_val}
    assert train_keys.isdisjoint(val_keys)


def test_anchor_and_every_horizon_row_must_be_owned_by_the_scoring_split():
    # Rows 0-5 val-owned, row 6+ test-owned: the window at rows 2-6 has a
    # val anchor (row 4) but its last horizon row (6) is test-owned, so it
    # is scored nowhere.
    df = _global_df({"p1": ([8], 100.0, 1.0)})
    owner = np.array(["val"] * 6 + ["test"] * 2, dtype=object)
    ds = _small_ds(df)
    boundary = 2  # window index whose rows are 2..6
    assert boundary not in scored_window_indices(ds, owner, "val")
    assert boundary not in scored_window_indices(ds, owner, "test")


def test_participants_sharing_timestamps_and_values_stay_distinct():
    # Identical glucose arrays and identical clock times; p1 train, p2 val.
    df = _global_df({"p1": ([5], 100.0, 1.0), "p2": ([5], 100.0, 1.0)})
    owner = np.array(["train"] * 5 + ["val"] * 5, dtype=object)
    ds = _small_ds(df)
    assert len(ds) == 2
    assert scored_window_indices(ds, owner, "train") == [0]
    assert scored_window_indices(ds, owner, "val") == [1]
    assert ds.participants[0] != ds.participants[1]
    assert ds.target_times[0] == ds.target_times[1]  # same clock, both valid


def test_short_segments_produce_no_scored_windows():
    df = _global_df({"p1": ([4, 4], 100.0, 1.0)})  # each segment < span
    owner = np.array(["train"] * 8, dtype=object)
    ds = _small_ds(df)
    assert len(ds) == 0
    assert scored_window_indices(ds, owner, "train") == []


def test_gap_free_filter_drops_masked_windows_and_keeps_labels_aligned():
    df = _global_df({"p1": ([7], 100.0, 1.0)})
    ds = _small_ds(df)  # 3 windows: rows 0-4, 1-5, 2-6
    no_gap_fill = np.ones(len(df), dtype=bool)
    no_gap_fill[3] = False  # gap-filled row
    # Window 0 (rows 0-4): row 3 is only an intermediate horizon step, which
    # the mask rule does not check -> kept. Windows 1 and 2 use row 3 as an
    # input -> dropped.
    kept = gap_free_window_indices(ds, no_gap_fill)
    assert kept == [0]
    for i in kept:
        x, y = ds[i]
        rows = ds.row_ids[i]
        expected = torch.tensor(df.loc[rows[:3], "gl"].to_numpy(), dtype=torch.float32)
        assert torch.equal(x.squeeze(0), expected)
        assert ds.labels[i] == y


def test_sensitivity_tiers_partition_scored_windows_disjointly():
    from training.splits import sensitivity_tier_indices

    df = _global_df({"p1": ([9], 100.0, 1.0)})
    ds = _small_ds(df)  # 5 windows: inputs {i,i+1,i+2}, endpoint i+4
    owner = np.full(9, "test", dtype=object)
    no_gap_fill = np.ones(9, dtype=bool)
    no_gap_fill[6] = False          # row 6 fills a sensor gap
    on_reading = no_gap_fill.copy()
    on_reading[[1, 8]] = False      # rows 1, 8 sit between readings

    tiers = sensitivity_tier_indices(ds, owner, "test", on_reading, no_gap_fill)
    # w3 (rows 3,4,5 + endpoint 7) is the only all-on-reading window.
    assert tiers["exact_observed"] == [3]
    # w4: input row 6 is gap-filled, endpoint 8 is clean.
    assert tiers["gap_filled_inputs"] == [4]
    # w2: endpoint row 6 is gap-filled (wins over any input status).
    assert tiers["gap_filled_endpoint"] == [2]
    # Disjoint, and together with the primary mask they cover every scored
    # window: w0/w1 are primary-but-not-exact (between-readings input row 1).
    primary = set(gap_free_window_indices(ds, no_gap_fill))
    assert set(tiers["exact_observed"]) <= primary
    assert primary - set(tiers["exact_observed"]) == {0, 1}
    all_tiered = sum(tiers.values(), [])
    assert len(all_tiered) == len(set(all_tiered))


def test_sensitivity_tiers_only_include_scored_windows():
    from training.splits import sensitivity_tier_indices

    df = _global_df({"p1": ([9], 100.0, 1.0)})
    ds = _small_ds(df)
    owner = np.full(9, "test", dtype=object)
    owner[7:] = "held_out"  # w3/w4 horizons cross ownership: scored nowhere
    masks = np.ones(9, dtype=bool)
    tiers = sensitivity_tier_indices(ds, owner, "test", masks, masks)
    assert tiers["exact_observed"] == [0, 1, 2]
    assert tiers["gap_filled_inputs"] == tiers["gap_filled_endpoint"] == []


def test_gap_free_windows_exclude_a_window_masked_only_at_its_endpoint():
    from training.splits import sensitivity_tier_indices

    df = _global_df({"p1": ([7], 100.0, 1.0)})
    ds = _small_ds(df)  # 3 windows; w2 = inputs 2,3,4 + endpoint 6
    no_gap_fill = np.ones(7, dtype=bool)
    no_gap_fill[6] = False  # only the future endpoint row is gap-filled
    assert gap_free_window_indices(ds, no_gap_fill) == [0, 1]
    owner = np.full(7, "test", dtype=object)
    tiers = sensitivity_tier_indices(ds, owner, "test", no_gap_fill, no_gap_fill)
    assert tiers["gap_filled_endpoint"] == [2]
    assert tiers["gap_filled_inputs"] == []
