# DiaData Ingest + Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring DiaData v3 (Zenodo 17285631) in as an external, never-committed CGM data source and publish an aggregate audit that decides whether sweep r2 is worth running.

**Architecture:** A loader (`training/diadata.py`) emits the exact DataFrame shape `GlucoseTrendDataset` already consumes (`id, id_segment, time, gl`), with cohort filtering and our own gap segmentation; an audit CLI (`experiments/audit_diadata.py`) holds a citation-backed sensor table, computes per-cohort verdicts and label distributions, and writes a full local JSON plus an aggregates-only committed JSON. No new Dataset class, no training.

**Tech Stack:** Python 3.12, pandas, numpy, existing `training/` package, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-diadata-ingest-audit-design.md`

## Global Constraints

- Not a medical device: no code, comment, or output string implies clinical validity or treatment guidance.
- DiaData read from, never modified, never committed; `.gitignore` entry `DiaData` (no trailing slash).
- No subject-level rows, IDs, or timestamps in committed artifacts — aggregates only in `paper/results/diadata_audit_r1.json`.
- Test suite runs without DiaData or GlucoBench (synthetic frames; lazy path access).
- Only `5_min_sampling.zip` is extracted and used.
- Decision rule: included fast-class length-12 window count ≥ 2× weinstock's own, else stop before r2.
- Work on branch `feat/diadata-audit` off `main`.
- All test commands run as `venv/bin/python -m pytest ...` (bare `pytest` lacks sklearn).

## Review Focus

1. Duplicate timestamps for one subject (integration artifact) — loader must dedupe, keeping the first, or windows double-count. Test in Task 2.
2. Non-monotonic timestamps within a subject — loader must sort by `(id, time)` before segmentation or segments are garbage. Test in Task 2.
3. NaN / non-positive glucose rows — must be dropped before segmentation, not turned into labels. Test in Task 2.
4. A `Database` value absent from the sensor table — must land in the audit as `excluded_unknown`, never silently included. Test in Task 3.
5. A cohort that is empty after filtering — loader and audit must return empty frames / zero counts, not crash. Test in Task 2.

---

### Task 1: Acquisition, docs, and layout probe

**Files:**
- Modify: `.gitignore` (add `DiaData` line next to `GlucoBench`)
- Modify: `CLAUDE.md` (fetch block in Commands)
- Modify: `README.md` (data-sources line)

**Interfaces:**
- Produces: `../DiaData/` containing extracted `5_min_sampling/` contents plus `README.md`; recorded sha256 of the zip. No code.

- [ ] **Step 1: Branch**

```bash
git switch -c feat/diadata-audit main
```

- [ ] **Step 2: gitignore + docs edits**

`.gitignore`: after the `GlucoBench` line add:

```
DiaData
```

`CLAUDE.md`, in the one-time setup block after the GlucoBench clone lines, add:

```bash
# DiaData v3 (external T1D CGM collection, CC BY-NC 4.0 — sibling dir,
# never vendored). ~2.3GB download; only the 5-minute archive is used.
mkdir -p ../DiaData && cd ../DiaData
curl -LO https://zenodo.org/records/17285631/files/5_min_sampling.zip
curl -LO https://zenodo.org/records/17285631/files/README.md
shasum -a 256 5_min_sampling.zip   # recorded in the audit output
unzip 5_min_sampling.zip && cd -
```

`README.md`: in the data section next to GlucoBench, add one sentence:

```
Optional second source: [DiaData v3](https://zenodo.org/records/17285631)
(integrated T1D CGM collection, CC BY-NC 4.0) — external sibling
directory like GlucoBench, never committed; see the audit record in
paper/results/diadata_audit_r1.json.
```

- [ ] **Step 3: Download + extract (background; resumable)**

Run the CLAUDE.md commands verbatim from the repo root. Record the sha256.
Expected: `../DiaData/` contains the extracted CSV(s) and `README.md`.

- [ ] **Step 4: Layout probe (read-only)**

```bash
ls -la ../DiaData/ && find ../DiaData -name '*.csv' | head -20
head -3 "$(find ../DiaData -name '*.csv' | head -1)"
```

Expected: main CSV(s) with header `ts,PtID,GlucoseCGM,Database` (order may differ; extra columns tolerated). If the archive splits per cohort or names columns differently, ledger a Ruling with the observed layout and adjust the `COLUMNS` constant in Task 2 — the loader contract (`id, id_segment, time, gl` out) does not change.

- [ ] **Step 5: Read ../DiaData/README.md; commit docs**

```bash
git add .gitignore CLAUDE.md README.md
git commit -m "docs: DiaData v3 external data source - fetch instructions and gitignore"
```

### Task 2: Loader — cohort cache + `load_diadata`

**Files:**
- Create: `training/diadata.py`
- Test: `tests/test_diadata.py`

**Interfaces:**
- Consumes: `../DiaData/` layout from Task 1.
- Produces: `load_diadata(root: Path, include_cohorts: Sequence[str], cache_dir: Path | None = None) -> pd.DataFrame` with columns exactly `["id", "id_segment", "time", "gl"]`; `segment_ids(times: pd.Series) -> np.ndarray`; `split_to_cache(root: Path, cache_dir: Path) -> list[Path]`; constants `COLUMNS = {"time": "ts", "subject": "PtID", "glucose": "GlucoseCGM", "cohort": "Database"}`, `GAP_SECONDS = 330`.

- [ ] **Step 1: Write the failing tests**

```python
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
    f = pd.concat([f.iloc[[2, 0, 1, 3]], f.iloc[[1]]])          # shuffled + duplicate ts
    f.loc[f.index[-1], "GlucoseCGM"] = np.nan                    # dup row also NaN
    bad = _raw("CITY", 1, "2024-02-01", 1); bad["GlucoseCGM"] = 0.0
    root = _write_main_csv(tmp_path, [f, bad])
    df = load_diadata(root, ["CITY"], cache_dir=tmp_path / "cache")
    assert len(df) == 4                                          # dedup + NaN + nonpositive dropped
    assert df["time"].is_monotonic_increasing


def test_empty_cohort_returns_empty_frame(tmp_path):
    root = _write_main_csv(tmp_path, [_raw("CITY", 1, "2024-01-01", 3)])
    df = load_diadata(root, ["WISDM"], cache_dir=tmp_path / "cache")
    assert df.empty and list(df.columns) == ["id", "id_segment", "time", "gl"]


def test_split_to_cache_one_file_per_cohort(tmp_path):
    root = _write_main_csv(tmp_path, [_raw("CITY", 1, "2024-01-01", 3), _raw("WISDM", 2, "2024-01-01", 3)])
    files = split_to_cache(root, tmp_path / "cache")
    assert sorted(p.stem for p in files) == ["CITY", "WISDM"]
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_diadata.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.diadata'`

- [ ] **Step 3: Implement**

```python
"""DiaData v3 loader: emits the frame shape GlucoseTrendDataset consumes.

DiaData is an external, read-only sibling directory (like GlucoBench).
Research use of a public benchmark collection — not a medical device.
"""
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

COLUMNS = {"time": "ts", "subject": "PtID", "glucose": "GlucoseCGM", "cohort": "Database"}
GAP_SECONDS = 330  # 5 min + 30 s tolerance: beyond this, a new segment starts


def segment_ids(times: pd.Series) -> np.ndarray:
    gaps = times.diff().dt.total_seconds().fillna(0.0) > GAP_SECONDS
    return gaps.cumsum().to_numpy()


def _main_csvs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.csv") if "HeartRate" not in p.name)


def split_to_cache(root: Path, cache_dir: Path) -> list[Path]:
    """One pass over the (large) main CSVs, split into one CSV per cohort.
    Cache lives under gitignored results/; DiaData itself is never written."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob("*.csv"))
    if existing:  # cache hit; a rerun must not append duplicates
        return existing
    seen: set[Path] = set()
    for csv in _main_csvs(root):
        for chunk in pd.read_csv(csv, chunksize=2_000_000):
            for cohort, rows in chunk.groupby(COLUMNS["cohort"]):
                out = cache_dir / f"{cohort}.csv"
                rows.to_csv(out, mode="a", header=out not in seen, index=False)
                seen.add(out)
    return sorted(seen)


def _cohort_frame(root: Path, cohort: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{cohort}.csv"
    if not path.exists():
        split_to_cache(root, cache_dir)
    if not path.exists():  # cohort absent from the data
        return pd.DataFrame(columns=list(COLUMNS.values()))
    return pd.read_csv(path)


def load_diadata(root: Path, include_cohorts: Sequence[str], cache_dir: Path | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or root.parent / "results" / "diadata_cache"
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
```

Note: `load_diadata`'s default cache dir is wrong for callers outside the repo root — the audit CLI (Task 3) always passes `cache_dir=GLUCOEDGE_ROOT / "results" / "diadata_cache"` explicitly.

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_diadata.py -v`
Expected: 5 PASS

- [ ] **Step 5: Full suite + commit**

Run: `venv/bin/python -m pytest -q`
Expected: all pass (97 + 5).

```bash
git add training/diadata.py tests/test_diadata.py
git commit -m "feat(training): DiaData loader - cohort cache, gap segmentation, GlucoseTrendDataset frame contract"
```

### Task 3: Audit functions — sensor table, collinearity, verdicts, label counts

**Files:**
- Create: `experiments/audit_diadata.py` (functions + CLI `main()`; the run itself is Task 4)
- Test: `tests/test_audit_diadata.py`

**Interfaces:**
- Consumes: `load_diadata`, `segment_ids`, `GAP_SECONDS`, `COLUMNS` from Task 2; `GlucoseTrendDataset`, `FIVE_CLASSES` from `training/`.
- Produces: `SENSOR_TABLE: dict[str, dict]`, `COLLINEAR_FLAG_THRESHOLD = 0.5`, `collinear_fraction(gl: np.ndarray, segments: np.ndarray) -> float`, `verdict(cohort: str, collinear_frac: float) -> str`, `label_window_counts(frame: pd.DataFrame) -> dict[str, int]`, `main()`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_audit_diadata.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.audit_diadata'`

- [ ] **Step 3: Implement**

```python
"""DiaData cohort audit: sensor provenance, interpolation heuristic,
overlap exclusion, and label distributions. Read-only; publishes
aggregates only. Research artifact for a public benchmark collection —
not a medical device."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.diadata import COLUMNS, load_diadata, split_to_cache
from training.labeling import FIVE_CLASSES
from training.dataset import GlucoseTrendDataset

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
DIADATA_ROOT = GLUCOEDGE_ROOT.parent / "DiaData"
CACHE_DIR = GLUCOEDGE_ROOT / "results" / "diadata_cache"
OUT_LOCAL = GLUCOEDGE_ROOT / "results" / "diadata_audit" / "audit_r1.json"
OUT_PUBLISHED = GLUCOEDGE_ROOT / "paper" / "results" / "diadata_audit_r1.json"

COLLINEAR_FLAG_THRESHOLD = 0.5

# Native sensor interval per DiaData source study. base_verdict encodes the
# spec's two cohort-level exclusions; collinearity can still downgrade an
# "included" cohort to excluded_flagged at audit time. Citations are the
# study's primary publication or registry entry.
SENSOR_TABLE = {
    "CITY":      {"sensor": "Dexcom G5", "interval_min": 5, "base_verdict": "included",
                  "citation": "Laffel et al. 2020, JAMA 323(23); JAEB CITY study"},
    "DCLP3":     {"sensor": "Dexcom G6", "interval_min": 5, "base_verdict": "included",
                  "citation": "Brown et al. 2019, NEJM 381(18); iDCL protocol 3"},
    "SENCE":     {"sensor": "Dexcom G5", "interval_min": 5, "base_verdict": "included",
                  "citation": "Strategies to Enhance New CGM use in Early childhood, JAEB 2021"},
    "WISDM":     {"sensor": "Dexcom G5", "interval_min": 5, "base_verdict": "included",
                  "citation": "Pratley et al. 2020, JAMA 323(23); JAEB WISDM"},
    "PEDAP":     {"sensor": "Dexcom G6", "interval_min": 5, "base_verdict": "included",
                  "citation": "Wadwa et al. 2023, NEJM 388(11); PEDAP trial"},
    "ReplaceBG": {"sensor": "Dexcom G4 505", "interval_min": 5, "base_verdict": "included",
                  "citation": "Aleppo et al. 2017, Diabetes Care 40(4); JAEB REPLACE-BG"},
    "D1NAMO":    {"sensor": "Medtronic iPro2", "interval_min": 5, "base_verdict": "excluded_overlap",
                  "citation": "Dubosson et al. 2018 - same acquisition as GlucoBench's dubosson2018 cohort"},
    "RT-CGM":    {"sensor": "mixed (DexCom SEVEN / Paradigm / FreeStyle Navigator)",
                  "interval_min": None, "base_verdict": "excluded_interval",
                  "citation": "JDRF CGM RCT, NEJM 2008 359(14) - per-subject sensor not recoverable"},
    "HUPA-UCM":  {"sensor": "FreeStyle Libre", "interval_min": 15, "base_verdict": "excluded_interval",
                  "citation": "Hidalgo et al. 2024, HUPA-UCM dataset"},
    "ShanghaiT1DM": {"sensor": "FreeStyle Libre H", "interval_min": 15, "base_verdict": "excluded_interval",
                  "citation": "Zhao et al. 2023, Sci Data 10:35"},
    "SevereHypoOlderT1D": {"sensor": "blinded CGM", "interval_min": 5, "base_verdict": "excluded_overlap",
                  "citation": "Weinstock et al. 2016 - same study as GlucoBench weinstock2016"},
    "CGM_HR_Adolescents": {"sensor": "unknown", "interval_min": None, "base_verdict": "excluded_interval",
                  "citation": "Diabetes adolescents time-series w/ HR - sensor undocumented"},
    "T1GDUJA":   {"sensor": "unknown", "interval_min": None, "base_verdict": "excluded_interval",
                  "citation": "T1GDUJA - sensor undocumented"},
}
# The Database column's literal spellings may differ (e.g. "Replace-BG",
# "Severe Hypoglycemia..."). Task 4 Step 1 reconciles observed names to
# these keys via ALIASES and ledgers any addition.
ALIASES: dict[str, str] = {}


def _canon(name: str) -> str:
    return ALIASES.get(name, name)


def collinear_fraction(gl: np.ndarray, segments: np.ndarray) -> float:
    """Fraction of consecutive triples (same segment) that are exactly
    arithmetic — the signature of linear interpolation."""
    same = (segments[2:] == segments[:-2]) & (segments[1:-1] == segments[:-2])
    if not same.any():
        return 0.0
    mid = np.abs(gl[:-2] + gl[2:] - 2.0 * gl[1:-1]) < 1e-6
    return float(mid[same].mean())


def verdict(cohort: str, collinear_frac: float) -> str:
    row = SENSOR_TABLE.get(_canon(cohort))
    if row is None:
        return "excluded_unknown"
    if row["base_verdict"] != "included":
        return row["base_verdict"]
    if collinear_frac > COLLINEAR_FLAG_THRESHOLD:
        return "excluded_flagged"
    return "included"


def label_window_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {c: 0 for c in FIVE_CLASSES}
    ds = GlucoseTrendDataset(frame, classes=FIVE_CLASSES)
    labels = np.array(ds.labels)
    return {c: int((labels == i).sum()) for i, c in enumerate(FIVE_CLASSES)}


def _cohort_report(name: str) -> dict:
    frame = load_diadata(DIADATA_ROOT, [name], cache_dir=CACHE_DIR)
    row = SENSOR_TABLE.get(_canon(name), {})
    if frame.empty:
        return {"cohort": name, "subjects": 0, "hours": 0.0, "segments": 0,
                "collinear_fraction": 0.0, "verdict": verdict(name, 0.0),
                "sensor": row.get("sensor", "unknown"), "citation": row.get("citation", "")}
    cf = collinear_fraction(frame["gl"].to_numpy(), frame["id_segment"].to_numpy()
                            + pd.factorize(frame["id"])[0] * 10**9)
    v = verdict(name, cf)
    report = {
        "cohort": name,
        "sensor": row.get("sensor", "unknown"),
        "interval_min": row.get("interval_min"),
        "citation": row.get("citation", ""),
        "subjects": int(frame["id"].nunique()),
        "hours": round(len(frame) * 5 / 60.0, 1),
        "segments": int(frame.groupby("id")["id_segment"].nunique().sum()),
        "collinear_fraction": round(cf, 4),
        "verdict": v,
    }
    if v == "included":
        report["label_windows"] = label_window_counts(frame)
    return report


def _weinstock_fast_count() -> dict:
    """Weinstock fast-class window count under the same rule (full frame,
    no split provenance — this is a data-volume comparison, not an eval)."""
    import sys
    sys.path.insert(0, str(GLUCOEDGE_ROOT / "GlucoBench"))
    from training.train import load_formatter
    formatter = load_formatter("weinstock")
    frame = pd.concat([formatter.train_data, formatter.val_data, formatter.test_data])
    counts = label_window_counts(frame)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-sha256", default="", help="sha256 of 5_min_sampling.zip (provenance)")
    args = parser.parse_args()

    observed = sorted(p.stem for p in split_to_cache(DIADATA_ROOT, CACHE_DIR))
    reports = [_cohort_report(name) for name in observed]
    included = [r for r in reports if r["verdict"] == "included"]
    fast = sum(r["label_windows"]["falling_fast"] + r["label_windows"]["rising_fast"] for r in included)
    weinstock = _weinstock_fast_count()
    w_fast = weinstock["falling_fast"] + weinstock["rising_fast"]

    audit = {
        "study": "diadata-audit",
        "run_id": "diadata-audit-r1",
        "diadata_version": "v3 (Zenodo 17285631)",
        "zip_sha256": args.zip_sha256,
        "observed_cohorts": observed,
        "cohorts": reports,
        "overlap_checks": [
            {"diadata_source": "SevereHypoOlderT1D", "glucobench_cohort": "weinstock2016",
             "evidence": "Weinstock et al. 2016 is the primary publication of both; "
                         "same T1D Exchange severe-hypoglycemia older-adults study",
             "verdict": "excluded_overlap"},
            {"diadata_source": "D1NAMO", "glucobench_cohort": "dubosson2018",
             "evidence": "Dubosson et al. 2018 is D1NAMO's own paper; GlucoBench's "
                         "dubosson dataset is the same acquisition",
             "verdict": "excluded_overlap"},
        ],
        "summary": {
            "included_cohorts": [r["cohort"] for r in included],
            "included_subjects": sum(r["subjects"] for r in included),
            "included_hours": round(sum(r["hours"] for r in included), 1),
            "included_fast_class_windows": int(fast),
            "weinstock_fast_class_windows": int(w_fast),
            "fast_class_multiplier": round(fast / w_fast, 2) if w_fast else None,
            "decision_rule_2x_passed": bool(w_fast and fast >= 2 * w_fast),
        },
    }
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(json.dumps(audit, indent=2) + "\n")
    OUT_PUBLISHED.write_text(json.dumps(audit, indent=2) + "\n")  # already aggregate-only
    print(json.dumps(audit["summary"], indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_audit_diadata.py -v`
Expected: 5 PASS

- [ ] **Step 5: Full suite + commit**

Run: `venv/bin/python -m pytest -q`
Expected: all pass.

```bash
git add experiments/audit_diadata.py tests/test_audit_diadata.py
git commit -m "feat(experiments): DiaData audit - sensor table, collinearity heuristic, verdicts, label counts"
```

### Task 4: Audit run, published JSON, decision note

**Files:**
- Create: `paper/results/diadata_audit_r1.json` (by running the CLI)
- Modify: `docs/superpowers/specs/2026-10-12-diadata-ingest-audit-design.md` (decision note appended)

**Interfaces:**
- Consumes: Task 3's `main()`; Task 1's extracted `../DiaData/`.

- [ ] **Step 1: Reconcile cohort names**

```bash
venv/bin/python -c "from experiments.audit_diadata import *; print(sorted(p.stem for p in split_to_cache(DIADATA_ROOT, CACHE_DIR)))"
```

Expected: 13 names. Map any spelling that differs from a `SENSOR_TABLE` key via the `ALIASES` dict (e.g. `"Replace-BG": "ReplaceBG"`); ledger each alias as a Ruling. A name with no plausible table row stays unmapped → `excluded_unknown`.

- [ ] **Step 2: Run the audit (background; the cache split reads ~2.3GB once)**

```bash
venv/bin/python -m experiments.audit_diadata --zip-sha256 <sha from Task 1>
```

Expected: summary JSON printed; both output files written. Sanity-check subject totals against the DiaData paper (1,720 overall) and weinstock fast counts against sweep-r1 scale; ledger discrepancies.

- [ ] **Step 3: Verify the published file is aggregate-only**

```bash
venv/bin/python -c "import json; d=json.load(open('paper/results/diadata_audit_r1.json')); s=json.dumps(d); assert 'PtID' not in s and 'ts\"' not in s; print('keys ok:', list(d.keys()))"
```

Expected: no subject-level keys; lists are cohort-level only.

- [ ] **Step 4: Append the decision note to the spec**

Add at the bottom of the spec file, with the real numbers:

```markdown
## Audit decision note (2026-10-12, run diadata-audit-r1)

Included cohorts: <list>. Included subjects: <n>, hours: <n>.
Fast-class windows: <n> vs weinstock <n> — multiplier <x>.
Decision rule (>= 2x): <PASSED/FAILED>. <One sentence: proceed to r2 spec / stop and reassess.>
Excluded: <cohort: verdict> for each exclusion.
```

- [ ] **Step 5: Full suite, commit, ledger**

Run: `venv/bin/python -m pytest -q`
Expected: all pass.

```bash
git add paper/results/diadata_audit_r1.json docs/superpowers/specs/2026-10-12-diadata-ingest-audit-design.md
git commit -m "docs(audit): DiaData cohort audit r1 - verdicts, label distributions, 2x decision rule"
```
