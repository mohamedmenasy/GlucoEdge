"""Audit pure-function tests — no DiaData needed."""
import numpy as np
import pandas as pd
import pytest

from experiments.audit_diadata import (SENSOR_TABLE, collinear_fraction,
                                       label_window_counts, verdict)


def test_sensor_table_covers_known_cohorts_with_verdict_fields():
    for name, row in SENSOR_TABLE.items():
        assert row["interval_min"] in (1, 5, 15, None)
        assert row["base_verdict"] in ("included", "excluded_interval", "excluded_overlap")
        assert row["citation"]


def test_collinear_fraction_flags_interpolated_runs():
    seg = np.zeros(6, dtype=int)
    linear = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])
    assert collinear_fraction(linear, seg) == 1.0
    noisy = np.array([100.0, 104.0, 111.0, 113.0, 121.0, 124.0])
    assert collinear_fraction(noisy, seg) < 0.5


def test_collinear_fraction_respects_segment_boundaries():
    gl = np.array([100.0, 105.0, 110.0, 200.0, 100.0, 50.0])
    segs = np.array([0, 0, 0, 1, 1, 1])
    # only triples fully inside one segment count: (100,105,110) collinear,
    # (200,100,50) not; cross-boundary triples ignored
    assert collinear_fraction(gl, segs) == pytest.approx(0.5)


def test_verdict_unknown_cohort_and_flagging():
    assert verdict("NOT_A_REAL_COHORT", 0.0) == "excluded_unknown"
    included = next(k for k, v in SENSOR_TABLE.items() if v["base_verdict"] == "included")
    assert verdict(included, 0.1) == "included"
    assert verdict(included, 0.9) == "excluded_flagged"
    excluded = next(k for k, v in SENSOR_TABLE.items() if v["base_verdict"] != "included")
    assert verdict(excluded, 0.0) == SENSOR_TABLE[excluded]["base_verdict"]


def test_label_window_counts_on_synthetic_frame():
    n = 40
    t = pd.date_range("2024-01-01", periods=n, freq="5min")
    frame = pd.DataFrame({"id": "X:1", "id_segment": 0, "time": t,
                          "gl": np.full(n, 110.0)})  # flat -> all stable
    counts = label_window_counts(frame)
    assert counts["stable"] > 0
    assert counts["rising_fast"] == 0 and counts["falling_fast"] == 0
