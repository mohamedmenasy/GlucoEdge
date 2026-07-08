from datetime import timedelta

import numpy as np

from conversion.synthetic_trace import generate_trace
from training.labeling import label_trend


def test_trace_shape_and_cadence():
    trace = generate_trace()
    assert len(trace) == 288  # 24h at 5-min cadence
    deltas = {(b[0] - a[0]) for a, b in zip(trace, trace[1:])}
    assert deltas == {timedelta(minutes=5)}


def test_trace_values_plausible_and_cover_int8_ceiling():
    values = np.array([v for _, v in generate_trace()])
    assert values.min() >= 40.0
    assert values.max() <= 400.0
    assert values.max() > 240.0  # must exercise INT8 saturation region


def test_trace_produces_all_five_labels():
    values = [v for _, v in generate_trace()]
    labels = set()
    for i in range(len(values) - 15):
        # label_trend takes the two raw mg/dL readings 15 minutes apart, not
        # a precomputed rate - it derives rate = (future - last) / horizon
        # internally (see training/labeling.py).
        labels.add(label_trend(values[i + 11], values[i + 14], horizon_minutes=15.0))
    assert labels == {"falling_fast", "falling", "stable", "rising", "rising_fast"}


def test_trace_is_deterministic():
    assert generate_trace() == generate_trace()
