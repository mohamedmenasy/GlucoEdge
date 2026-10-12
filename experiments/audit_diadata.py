"""DiaData cohort audit: sensor provenance, interpolation heuristic,
overlap exclusion, and label distributions. Read-only; publishes
aggregates only. Research artifact for a public benchmark collection —
not a medical device."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.dataset import GlucoseTrendDataset
from training.diadata import COLUMNS, load_diadata, split_to_cache
from training.labeling import FIVE_CLASSES

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
# "Severe Hypoglycemia..."). The audit run reconciles observed names to
# these keys via ALIASES; each addition is ledgered.
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
    from training.train import load_formatter
    formatter = load_formatter("weinstock")
    frame = pd.concat([formatter.train_data, formatter.val_data, formatter.test_data])
    return label_window_counts(frame)


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
