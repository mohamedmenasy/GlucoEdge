import numpy as np
import pytest

from training.deltas import tod_channels


def test_values_hand_computed_at_noon():
    out = tod_channels(np.array([12.0]), length=12)
    assert out.shape == (1, 2, 12)
    # last step is the anchor itself: h=12 -> sin ~ 0, cos = -1
    assert out[0, 0, -1] == pytest.approx(np.sin(2 * np.pi * 12 / 24), abs=1e-6)
    assert out[0, 1, -1] == pytest.approx(-1.0, abs=1e-6)
    # first step is 55 minutes earlier
    h0 = 12.0 - 11 * 5 / 60
    assert out[0, 0, 0] == pytest.approx(np.sin(2 * np.pi * h0 / 24), abs=1e-6)


def test_midnight_wrap_is_continuous():
    # anchor 00:10 -> early steps fall before midnight; sin/cos stay bounded
    out = tod_channels(np.array([10 / 60]), length=12)
    assert np.all(np.isfinite(out)) and np.all(np.abs(out) <= 1.0)
    h0 = (10 / 60 - 55 / 60) % 24       # 23.25
    assert out[0, 1, 0] == pytest.approx(np.cos(2 * np.pi * h0 / 24), abs=1e-6)


def test_float32_output():
    assert tod_channels(np.array([3.5])).dtype == np.float32
