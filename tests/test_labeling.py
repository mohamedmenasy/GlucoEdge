import numpy as np
import pytest

from training.labeling import (
    FIVE_CLASSES,
    THREE_CLASSES,
    THREE_CLASS_MAP,
    classify_rate,
    label_trend,
    lsq_slope,
)


@pytest.mark.parametrize("last_value,future_value,expected", [
    (100.0, 100.0 - 2 * 15, "falling_fast"),   # rate == -2 (boundary)
    (100.0, 100.0 - 3 * 15, "falling_fast"),   # rate == -3 (interior)
    (100.0, 100.0 - 1 * 15, "falling"),        # rate == -1 (boundary)
    (100.0, 100.0 - 1.5 * 15, "falling"),      # rate == -1.5 (interior)
    (100.0, 100.0 - 0.5 * 15, "stable"),       # rate == -0.5 (interior)
    (100.0, 100.0, "stable"),                  # rate == 0
    (100.0, 100.0 + 0.99 * 15, "stable"),      # rate == 0.99 (just under boundary)
    (100.0, 100.0 + 1 * 15, "rising"),         # rate == 1 (boundary)
    (100.0, 100.0 + 1.5 * 15, "rising"),       # rate == 1.5 (interior)
    (100.0, 100.0 + 2 * 15, "rising_fast"),    # rate == 2 (boundary)
    (100.0, 100.0 + 3 * 15, "rising_fast"),    # rate == 3 (interior)
])
def test_label_trend_thresholds(last_value, future_value, expected):
    assert label_trend(last_value, future_value, horizon_minutes=15.0) == expected


def test_five_classes_order():
    assert FIVE_CLASSES == ["falling_fast", "falling", "stable", "rising", "rising_fast"]


def test_three_class_map_covers_all_five_classes():
    assert set(THREE_CLASS_MAP.keys()) == set(FIVE_CLASSES)
    assert set(THREE_CLASS_MAP.values()) == set(THREE_CLASSES)


def test_three_class_map_collapses_fast_tiers():
    assert THREE_CLASS_MAP["falling_fast"] == "falling"
    assert THREE_CLASS_MAP["falling"] == "falling"
    assert THREE_CLASS_MAP["stable"] == "stable"
    assert THREE_CLASS_MAP["rising"] == "rising"
    assert THREE_CLASS_MAP["rising_fast"] == "rising"


def test_classify_rate_boundaries():
    # asymmetric rules: <= on the falling side, < on the rising side
    assert classify_rate(-2.0) == "falling_fast"
    assert classify_rate(-1.0) == "falling"
    assert classify_rate(-0.999) == "stable"
    assert classify_rate(0.999) == "stable"
    assert classify_rate(1.0) == "rising"
    assert classify_rate(1.999) == "rising"
    assert classify_rate(2.0) == "rising_fast"


def test_classify_rate_matches_label_trend():
    for last, future in [(100, 130), (100, 70), (100, 101), (100, 85), (100, 115)]:
        rate = (future - last) / 15.0
        assert classify_rate(rate) == label_trend(last, future)


def test_lsq_slope_hand_computed():
    # exact line: 100, 105, 110, 115 over 0/5/10/15 min -> slope 1.0 mg/dL/min
    pts = np.array([[100.0, 105.0, 110.0, 115.0],
                    [100.0, 100.0, 100.0, 100.0],
                    [120.0, 110.0, 100.0, 90.0]], dtype=np.float32)
    s = lsq_slope(pts)
    assert np.allclose(s, [1.0, 0.0, -2.0], atol=1e-6)


def test_lsq_slope_noisy_vs_endpoint():
    # noise on an interior point moves the endpoint rate, not the LSQ rate sign
    pts = np.array([[100.0, 112.0, 101.0, 102.0]], dtype=np.float32)
    endpoint = (pts[0, -1] - pts[0, 0]) / 15.0
    assert abs(lsq_slope(pts)[0] - endpoint) > 0.0  # estimators differ
