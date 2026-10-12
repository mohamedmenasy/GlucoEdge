"""DiaData v3 loader: emits the frame shape GlucoseTrendDataset consumes.

DiaData is an external, read-only sibling directory (like GlucoBench).
Research use of a public benchmark collection — not a medical device.
"""
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

COLUMNS = {"time": "ts", "subject": "PtID", "glucose": "GlucoseCGM", "cohort": "Database"}
GAP_SECONDS = 330  # 5 min + 30 s tolerance: beyond this, a new segment starts


def segment_ids(times: pd.Series) -> np.ndarray:
    gaps = times.diff().dt.total_seconds().fillna(0.0) > GAP_SECONDS
    return gaps.cumsum().to_numpy()


def _default_cache_dir() -> Path:
    # Must live under the repo's gitignored results/, never outside it.
    return Path(__file__).resolve().parents[1] / "results" / "diadata_cache"


def _main_csvs(root: Path) -> list[Path]:
    """Exactly the raw main-database CSV. The preprocessed variant
    interpolates gaps (label contamination) and the SDB* files are
    subsets of MDB; matching the exact raw filename excludes both, plus
    AppleDouble junk."""
    return sorted(
        p for p in root.rglob("MDB_5min_sampling_raw.csv")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )


def _safe_name(cohort) -> str:
    """Cohort names come from a downloaded CSV (untrusted); they become
    cache filenames, so strip anything that could traverse paths."""
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(cohort))
    cleaned = cleaned.lstrip(".")
    return cleaned or "_"


def split_to_cache(root: Path, cache_dir: Path) -> list[Path]:
    """One pass over the (large) main CSVs, split into one CSV per cohort.
    Cache lives under gitignored results/; DiaData itself is never written."""
    if cache_dir.exists() and any(cache_dir.glob("*.csv")):
        return sorted(cache_dir.glob("*.csv"))  # cache hit; never append twice
    sources = _main_csvs(root)
    if not sources:
        raise FileNotFoundError(f"no MDB_5min_sampling_raw.csv under {root}")
    # Build into a .partial dir and rename at the end: an interrupted
    # split must never be mistaken for a complete cache.
    partial = cache_dir.with_name(cache_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    seen: set[Path] = set()
    for csv in sources:
        for chunk in pd.read_csv(csv, chunksize=2_000_000):
            for cohort, rows in chunk.groupby(COLUMNS["cohort"]):
                out = partial / f"{_safe_name(cohort)}.csv"
                rows.to_csv(out, mode="a", header=out not in seen, index=False)
                seen.add(out)
    if cache_dir.exists():
        cache_dir.rmdir()  # empty leftover; a non-empty dir returned above
    partial.rename(cache_dir)
    return sorted(cache_dir.glob("*.csv"))


def _cohort_frame(root: Path, cohort: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{_safe_name(cohort)}.csv"
    if not path.exists():
        split_to_cache(root, cache_dir)
    if not path.exists():  # cohort absent from the data
        return pd.DataFrame(columns=list(COLUMNS.values()))
    return pd.read_csv(path)


def load_diadata(root: Path, include_cohorts: Sequence[str], cache_dir: Path | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or _default_cache_dir()
    parts = []
    for cohort in include_cohorts:
        raw = _cohort_frame(root, cohort, cache_dir)
        if raw.empty:
            continue
        df = pd.DataFrame({
            "id": cohort + ":" + raw[COLUMNS["subject"]].astype(str),
            "time": pd.to_datetime(raw[COLUMNS["time"]]),
            "gl": pd.to_numeric(raw[COLUMNS["glucose"]], errors="coerce"),
        })
        df = df.dropna(subset=["gl"])
        df = df[df["gl"] > 0]
        df = df.sort_values(["id", "time"]).drop_duplicates(["id", "time"], keep="first")
        df["id_segment"] = df.groupby("id", sort=False)["time"].transform(
            lambda t: segment_ids(t)
        )
        parts.append(df[["id", "id_segment", "time", "gl"]])
    if not parts:
        return pd.DataFrame(columns=["id", "id_segment", "time", "gl"])
    return pd.concat(parts, ignore_index=True)
