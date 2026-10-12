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


def test_main_handles_included_cohort_filtered_to_empty(tmp_path, monkeypatch):
    # An included cohort whose rows all fail the glucose filter must flow
    # through main() with zero counts, not crash on a missing label_windows.
    import experiments.audit_diadata as audit
    root = tmp_path / "DiaData"
    root.mkdir()
    f = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="5min"),
                      "PtID": 1, "GlucoseCGM": 0.0, "Database": "CITY"})
    f.to_csv(root / "MDB_5min_sampling_raw.csv", index=False)
    monkeypatch.setattr(audit, "DIADATA_ROOT", root)
    monkeypatch.setattr(audit, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(audit, "OUT_LOCAL", tmp_path / "local.json")
    monkeypatch.setattr(audit, "OUT_PUBLISHED", tmp_path / "published.json")
    monkeypatch.setattr(audit, "_weinstock_fast_count",
                        lambda: {c: 10 for c in ("falling_fast", "falling", "stable", "rising", "rising_fast")})
    monkeypatch.setattr("sys.argv", ["audit_diadata"])
    audit.main()
    import json
    out = json.loads((tmp_path / "published.json").read_text())
    city = next(r for r in out["cohorts"] if r["cohort"] == "CITY")
    assert city["verdict"] == "included" and sum(city["label_windows"].values()) == 0
    assert out["summary"]["included_fast_class_windows"] == 0


def test_dedupe_splits_removes_context_overlap_and_sorts():
    # GlucoBench prepends context rows from train to val/test; the
    # concatenated frame holds duplicate (id, id_segment, time) rows out
    # of order, which would fabricate 15-minute-rate windows.
    from experiments.audit_diadata import dedupe_splits
    t = pd.date_range("2024-01-01", periods=6, freq="5min")
    train = pd.DataFrame({"id": "w1", "id_segment": 0, "time": t[:4], "gl": 100.0})
    val = pd.DataFrame({"id": "w1", "id_segment": 0, "time": t[2:], "gl": 100.0})  # 2 context rows overlap
    out = dedupe_splits([train, val])
    assert len(out) == 6
    assert out["time"].is_monotonic_increasing
    assert not out.duplicated(["id", "id_segment", "time"]).any()


def test_label_window_counts_on_synthetic_frame():
    n = 40
    t = pd.date_range("2024-01-01", periods=n, freq="5min")
    frame = pd.DataFrame({"id": "X:1", "id_segment": 0, "time": t,
                          "gl": np.full(n, 110.0)})  # flat -> all stable
    counts = label_window_counts(frame)
    assert counts["stable"] > 0
    assert counts["rising_fast"] == 0 and counts["falling_fast"] == 0
