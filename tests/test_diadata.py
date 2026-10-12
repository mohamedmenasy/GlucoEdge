"""Loader tests — synthetic frames only; no DiaData download needed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.diadata import GAP_SECONDS, load_diadata, segment_ids, split_to_cache


def _raw(cohort, pid, start, n, step_min=5):
    ts = pd.date_range(start, periods=n, freq=f"{step_min}min")
    return pd.DataFrame({"ts": ts, "PtID": pid, "GlucoseCGM": np.linspace(100, 120, n), "Database": cohort})


def _write_main_csv(tmp_path, frames):
    root = tmp_path / "DiaData"
    root.mkdir()
    pd.concat(frames).to_csv(root / "main.csv", index=False)
    return root


def test_segment_ids_split_on_gap():
    t = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:16", "2024-01-01 00:21"]))
    assert segment_ids(t).tolist() == [0, 0, 1, 1]  # 11 min > 330 s


def test_load_columns_id_format_and_filter(tmp_path):
    root = _write_main_csv(tmp_path, [_raw("CITY", 7, "2024-01-01", 5), _raw("HUPA", 3, "2024-01-01", 5)])
    df = load_diadata(root, ["CITY"], cache_dir=tmp_path / "cache")
    assert list(df.columns) == ["id", "id_segment", "time", "gl"]
    assert set(df["id"]) == {"CITY:7"}


def test_load_sorts_dedupes_and_drops_bad_glucose(tmp_path):
    f = _raw("CITY", 1, "2024-01-01", 4)
    f = pd.concat([f.iloc[[2, 0, 1, 3]], f.iloc[[1]]]).reset_index(drop=True)  # shuffled + duplicate ts
    f.loc[4, "GlucoseCGM"] = 999.0                               # dup carries a bogus value
    nan_row = _raw("CITY", 1, "2024-03-01", 1)
    nan_row["GlucoseCGM"] = np.nan
    bad = _raw("CITY", 1, "2024-02-01", 1)
    bad["GlucoseCGM"] = 0.0
    root = _write_main_csv(tmp_path, [f, nan_row, bad])
    df = load_diadata(root, ["CITY"], cache_dir=tmp_path / "cache")
    assert len(df) == 4                                          # dedup keeps first; NaN + nonpositive dropped
    assert 999.0 not in df["gl"].values
    assert df["time"].is_monotonic_increasing


def test_empty_cohort_returns_empty_frame(tmp_path):
    root = _write_main_csv(tmp_path, [_raw("CITY", 1, "2024-01-01", 3)])
    df = load_diadata(root, ["WISDM"], cache_dir=tmp_path / "cache")
    assert df.empty and list(df.columns) == ["id", "id_segment", "time", "gl"]


def test_split_to_cache_sanitizes_hostile_cohort_names(tmp_path):
    # Database values come from a downloaded CSV: untrusted. A traversal
    # attempt must not escape cache_dir.
    root = _write_main_csv(tmp_path, [_raw("../evil", 1, "2024-01-01", 3)])
    cache = tmp_path / "cache"
    files = split_to_cache(root, cache)
    assert len(files) == 1
    assert files[0].parent == cache
    assert not (tmp_path / "evil.csv").exists()


def test_main_csvs_prefer_mdb_and_skip_appledouble(tmp_path):
    from training.diadata import _main_csvs
    root = tmp_path / "DiaData"
    (root / "__MACOSX").mkdir(parents=True)
    (root / "MDB_5min_sampling_raw.csv").write_text("ts,PtID,GlucoseCGM,Database\n")
    (root / "SDBI_5min_sampling_raw.csv").write_text("ts,PtID,GlucoseCGM,Database,Sex\n")
    (root / "__MACOSX" / "._MDB_5min_sampling_raw.csv").write_text("junk")
    got = _main_csvs(root)
    assert [p.name for p in got] == ["MDB_5min_sampling_raw.csv"]  # SDB* are MDB subsets; AppleDouble junk skipped


def test_split_to_cache_one_file_per_cohort(tmp_path):
    root = _write_main_csv(tmp_path, [_raw("CITY", 1, "2024-01-01", 3), _raw("WISDM", 2, "2024-01-01", 3)])
    files = split_to_cache(root, tmp_path / "cache")
    assert sorted(p.stem for p in files) == ["CITY", "WISDM"]
