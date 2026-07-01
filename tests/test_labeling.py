import pytest

from training.labeling import (
    FIVE_CLASSES,
    THREE_CLASSES,
    THREE_CLASS_MAP,
    label_trend,
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
