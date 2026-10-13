"""RT-CGM reopen audit (sweep r3 spec section 2): admit per-subject on cadence.

Rule: modal consecutive-interval within segments == 300s AND >= 95% of
intervals in [270, 330]s. Sensor brand is unrecoverable and irrelevant.
Participant ids go only to gitignored results/; the published JSON holds
counts and a binned histogram.
"""
import json
from collections import Counter

import numpy as np

from experiments.calibration_coverage import GLUCOEDGE_ROOT, _write_json

TOL_LOW, TOL_HIGH, MODAL = 270.0, 330.0, 300.0
ADMITTED_PATH = GLUCOEDGE_ROOT / "results" / "rtcgm_admitted.json"
AUDIT_PATH = GLUCOEDGE_ROOT / "paper" / "results" / "diadata_audit_r2.json"
RULE = ("admit iff modal within-segment interval == 300s and >=95% of "
        "intervals in [270s, 330s]")


def subject_verdicts(frame):
    out = {}
    for sid, g in frame.groupby("id"):
        ivals = []
        for _, seg in g.groupby("id_segment"):
            d = seg["time"].sort_values().diff().dt.total_seconds().dropna()
            ivals.extend(float(x) for x in d)
        if not ivals:
            out[str(sid)] = {"admitted": False, "modal_s": None, "within_tol": 0.0}
            continue
        modal = float(Counter(ivals).most_common(1)[0][0])
        within = float(np.mean([(TOL_LOW <= x <= TOL_HIGH) for x in ivals]))
        out[str(sid)] = {"admitted": bool(modal == MODAL and within >= 0.95),
                         "modal_s": modal, "within_tol": within}
    return out


def aggregate_stats(verdicts, frame):
    """Published aggregates over the whole cohort: interval distribution,
    tolerance-fraction spread, and the admitted pool's reading count."""
    ivals = []
    for _, g in frame.groupby("id"):
        for _, seg in g.groupby("id_segment"):
            d = seg["time"].sort_values().diff().dt.total_seconds().dropna()
            ivals.extend(float(x) for x in d)
    ivals = np.asarray(ivals)
    tols = [r["within_tol"] for r in verdicts.values()]
    admitted_ids = {s for s, r in verdicts.items() if r["admitted"]}
    return {
        "intervals_total": int(ivals.size),
        "intervals_below_270": int((ivals < TOL_LOW).sum()),
        "intervals_within_270_330": int(((ivals >= TOL_LOW) & (ivals <= TOL_HIGH)).sum()),
        "intervals_above_330": int((ivals > TOL_HIGH).sum()),
        "within_tol_min": float(min(tols)) if tols else 0.0,
        "within_tol_median": float(np.median(tols)) if tols else 0.0,
        "admitted_readings": int(frame["id"].astype(str).isin(admitted_ids).sum()),
    }


def main():
    from training.diadata import load_diadata
    frame = load_diadata(GLUCOEDGE_ROOT.parent / "DiaData", ["RT-CGM"],
                         cache_dir=GLUCOEDGE_ROOT / "results" / "diadata_cache")
    v = subject_verdicts(frame)
    admitted = sorted(s for s, r in v.items() if r["admitted"])
    _write_json(ADMITTED_PATH, {"admitted": admitted, "rule": RULE})
    hist = Counter()
    for r in v.values():
        if r["modal_s"] is not None:
            hist[int(r["modal_s"])] += 1
    _write_json(AUDIT_PATH, {
        "cohort": "RT-CGM", "rule": RULE,
        "subjects_total": len(v), "subjects_admitted": len(admitted),
        "modal_interval_histogram_s": {str(k): n for k, n in sorted(hist.items())},
        **aggregate_stats(v, frame),
    })
    print(json.dumps({"total": len(v), "admitted": len(admitted)}))


if __name__ == "__main__":
    main()
