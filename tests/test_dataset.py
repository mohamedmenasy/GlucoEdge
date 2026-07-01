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


def test_collapse_map_reduces_to_three_classes():
    df = _make_df([15], start_value=100.0, step=20.0)  # same rising_fast case
    dataset = GlucoseTrendDataset(df, classes=THREE_CLASSES, collapse_map=THREE_CLASS_MAP)
    assert len(dataset) == 1

    _, y = dataset[0]
    assert THREE_CLASSES[y] == "rising"
