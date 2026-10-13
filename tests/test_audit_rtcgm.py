import numpy as np
import pandas as pd

import experiments.audit_rtcgm as audit
from experiments.audit_rtcgm import subject_verdicts


def _frame(rows):
    return pd.DataFrame(rows, columns=["id", "id_segment", "time", "gl"])


def _times(start, step_s, n):
    return pd.date_range(start, periods=n, freq=f"{step_s}s")


def test_clean_5min_admitted_1min_rejected():
    rows = [("a", 0, t, 100.0) for t in _times("2020-01-01", 300, 50)]
    rows += [("b", 0, t, 100.0) for t in _times("2020-01-01", 60, 50)]
    v = subject_verdicts(_frame(rows))
    assert v["a"]["admitted"] is True and v["a"]["modal_s"] == 300.0
    assert v["b"]["admitted"] is False


def test_jitter_beyond_tolerance_rejected():
    # modal 300s but 10% of intervals at 600s -> within_tol = 0.9 < 0.95
    t, rows = pd.Timestamp("2020-01-01"), []
    for i in range(50):
        t += pd.Timedelta(seconds=600 if i % 10 == 0 else 300)
        rows.append(("j", 0, t, 100.0))
    v = subject_verdicts(_frame(rows))
    assert v["j"]["admitted"] is False and v["j"]["modal_s"] == 300.0


def test_single_reading_subject_rejected_not_crash():
    v = subject_verdicts(_frame([("solo", 0, pd.Timestamp("2020-01-01"), 100.0)]))
    assert v["solo"]["admitted"] is False and v["solo"]["modal_s"] is None


def test_intervals_never_cross_segments():
    # two segments far apart: the 1-day gap must not count as an interval
    rows = [("s", 0, t, 100.0) for t in _times("2020-01-01", 300, 20)]
    rows += [("s", 1, t, 100.0) for t in _times("2020-01-02", 300, 20)]
    v = subject_verdicts(_frame(rows))
    assert v["s"]["admitted"] is True and v["s"]["within_tol"] == 1.0


def test_main_aggregates_from_verdicts_and_frame():
    import pandas as pd
    t = pd.date_range("2020-01-01", periods=5, freq="300s")
    frame = pd.DataFrame({"id": ["a"] * 5, "id_segment": [0] * 5,
                          "time": t, "gl": [100.0] * 5})
    agg = audit.aggregate_stats(audit.subject_verdicts(frame), frame)
    assert agg["intervals_total"] == 4
    assert agg["intervals_within_270_330"] == 4
    assert agg["intervals_below_270"] == 0 and agg["intervals_above_330"] == 0
    assert agg["within_tol_min"] == 1.0 and agg["within_tol_median"] == 1.0
    assert agg["admitted_readings"] == 5
