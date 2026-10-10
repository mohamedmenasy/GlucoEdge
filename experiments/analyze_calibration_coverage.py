"""Aggregate metrics and paired participant-cluster bootstrap for the
calibration-coverage study (plan Section 4).

Pure numpy; the CLI at the bottom turns a run directory's cached
predictions into the published aggregate JSON and tables. A per-class
recall is undefined (NaN) on zero support, and any mean over an undefined
recall is undefined; zero is never substituted.
"""
import numpy as np

N_CLASSES = 5
STABLE = 2  # index of `stable` in FIVE_CLASSES
MINORITY = tuple(c for c in range(N_CLASSES) if c != STABLE)
ANALYSIS_SEED = 20261010


def confusion_counts(y_true, y_pred, n_classes=N_CLASSES):
    """Confusion matrix via bincount; rows = true class, cols = predicted."""
    idx = np.asarray(y_true) * n_classes + np.asarray(y_pred)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def recalls(cm):
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore"):
        r = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return r


def accuracy(cm):
    return float(np.diag(cm).sum() / cm.sum())


def macro_recall(cm):
    return float(np.mean(recalls(cm)))


def minority_macro_recall(cm):
    return float(np.mean(recalls(cm)[list(MINORITY)]))


def _clusters(participant_ids):
    participant_ids = np.asarray(participant_ids)
    participants = np.unique(participant_ids)
    return [np.flatnonzero(participant_ids == p) for p in participants]


def resample_rows(participant_ids, rng, clusters=None):
    """One cluster-bootstrap draw: participants sampled with replacement,
    each contributing every one of its windows (plan: never bootstrap
    individual overlapping windows)."""
    if clusters is None:
        clusters = _clusters(participant_ids)
    drawn = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[k] for k in drawn])


def paired_cluster_bootstrap(y_true, participant_ids, pred_pairs,
                             metric=minority_macro_recall,
                             n_boot=2000, seed=ANALYSIS_SEED):
    """Percentile CI for the mean paired metric difference (arm A - arm B),
    averaged over `pred_pairs` (one pair per model seed x calibration draw)
    inside each participant-cluster bootstrap draw."""
    y_true = np.asarray(y_true)
    pred_pairs = [(np.asarray(a), np.asarray(b)) for a, b in pred_pairs]

    def mean_delta(rows):
        deltas = [metric(confusion_counts(y_true[rows], a[rows]))
                  - metric(confusion_counts(y_true[rows], b[rows]))
                  for a, b in pred_pairs]
        return float(np.mean(deltas))  # NaN if any pair is undefined

    clusters = _clusters(participant_ids)
    rng = np.random.default_rng(seed)
    boot = np.array([mean_delta(resample_rows(participant_ids, rng, clusters))
                     for _ in range(n_boot)])
    undefined = float(np.mean(np.isnan(boot)))
    defined = boot[~np.isnan(boot)]
    ci = None
    if undefined <= 0.05 and len(defined):
        ci = [float(v) for v in np.percentile(defined, [2.5, 97.5])]
    point = mean_delta(np.arange(len(y_true)))
    return {
        "delta": None if np.isnan(point) else point,
        "ci95": ci,
        "undefined_fraction": undefined,
        "n_boot": n_boot,
        "seed": seed,
    }


# --- aggregation CLI ----------------------------------------------------------


def _metrics(cm):
    r = recalls(cm)
    return {
        "accuracy": accuracy(cm),
        "macro_recall": None if np.isnan(macro_recall(cm)) else macro_recall(cm),
        "minority_macro_recall": (None if np.isnan(minority_macro_recall(cm))
                                  else minority_macro_recall(cm)),
        "per_class_recall": [None if np.isnan(v) else float(v) for v in r],
        "support": [int(s) for s in cm.sum(axis=1)],
        "predicted_share": [float(s) for s in cm.sum(axis=0) / cm.sum()],
    }


def main():
    import argparse
    import json
    from pathlib import Path

    from experiments.calibration_coverage import (BUDGETS, DRAWS, RESULTS,
                                                  STOCHASTIC, TRAINING_SEEDS)

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    run = RESULTS / args.run_id
    ev = run / "eval"
    # Own run's cache; plain numeric arrays, no pickled objects.
    comp_npz = np.load(ev / "components.npz")

    def preds(component, name):
        p = ev / component / f"{name}.npy"
        return np.load(p) if p.exists() else None

    out = {"run_id": args.run_id, "n_boot": args.n_boot, "analysis_seed": ANALYSIS_SEED,
           "components": {}, "contrasts": {}, "mechanism": {}, "cells": []}

    sel_cells = [("sequential", b, 0) for b in BUDGETS] + \
                [(s, b, d) for s in STOCHASTIC for b in BUDGETS for d in DRAWS]

    for comp in ("held_out", "temporal"):
        y = comp_npz[f"{comp}_labels"]
        pids = comp_npz[f"{comp}_participants"]
        rows = {}
        for name in (["slope", "logistic"]
                     + [f"torch_seed{s}" for s in TRAINING_SEEDS]
                     + [f"seed{s}_float" for s in TRAINING_SEEDS]):
            p = preds(comp, name)
            if p is not None:
                rows[name] = _metrics(confusion_counts(y, p))
        rows["constant_stable"] = _metrics(confusion_counts(y, np.full(len(y), STABLE)))
        for strat, budget, draw in sel_cells:
            for seed in TRAINING_SEEDS:
                name = f"seed{seed}_int8_{strat}_{budget}_d{draw}"
                p = preds(comp, name)
                status = "ok" if p is not None else "missing"
                if comp == "held_out":
                    out["cells"].append({"seed": seed, "strategy": strat,
                                         "budget": budget, "draw": draw, "status": status})
                if p is not None:
                    rows[name] = _metrics(confusion_counts(y, p))
        # Observer arm (Amendment 3), descriptive only.
        for cell in ("sequential_200_d0",
                     "participant_balanced_range_coverage_200_d101"):
            for seed in TRAINING_SEEDS:
                name = f"seed{seed}_int8_{cell}_obsminmax"
                p = preds(comp, name)
                if p is not None:
                    rows[name] = _metrics(confusion_counts(y, p))
        out["components"][comp] = rows

        def pairs(name_a, name_b):
            got = []
            for seed in TRAINING_SEEDS:
                for draw in DRAWS:
                    a = preds(comp, name_a.format(seed=seed, draw=draw))
                    b = preds(comp, name_b.format(seed=seed, draw=draw))
                    if a is not None and b is not None:
                        got.append((a, b))
            return got

        contrasts = {
            "primary_range200_vs_balanced200": pairs(
                "seed{seed}_int8_participant_balanced_range_coverage_200_d{draw}",
                "seed{seed}_int8_participant_balanced_random_200_d{draw}"),
            "range200_vs_uniform200": pairs(
                "seed{seed}_int8_participant_balanced_range_coverage_200_d{draw}",
                "seed{seed}_int8_uniform_random_200_d{draw}"),
            "uniform200_vs_sequential200": pairs(
                "seed{seed}_int8_uniform_random_200_d{draw}",
                "seed{seed}_int8_sequential_200_d0"),
            "uniform1000_vs_sequential1000": pairs(
                "seed{seed}_int8_uniform_random_1000_d{draw}",
                "seed{seed}_int8_sequential_1000_d0"),
        }
        for strat in STOCHASTIC:
            contrasts[f"{strat}_1000_vs_200"] = pairs(
                f"seed{{seed}}_int8_{strat}_1000_d{{draw}}",
                f"seed{{seed}}_int8_{strat}_200_d{{draw}}")
        out["contrasts"][comp] = {
            k: (paired_cluster_bootstrap(y, pids, v, n_boot=args.n_boot)
                if v else {"delta": None, "status": "no pairs"})
            for k, v in contrasts.items()}

        # Mechanism controls: float vs QDQ-grid vs clip-only vs INT8.
        mech = {}
        for strat, budget, draw in sel_cells:
            deltas = {"float_minus_qdq": [], "float_minus_clip": [], "qdq_minus_int8": []}
            for seed in TRAINING_SEEDS:
                stem = f"seed{seed}_int8_{strat}_{budget}_d{draw}"
                f = preds(comp, f"seed{seed}_float")
                q = preds(comp, f"{stem}_qdqfloat")
                c = preds(comp, f"{stem}_cliponly")
                i8 = preds(comp, stem)
                if any(v is None for v in (f, q, c, i8)):
                    continue
                mm = lambda p: minority_macro_recall(confusion_counts(y, p))
                deltas["float_minus_qdq"].append(mm(f) - mm(q))
                deltas["float_minus_clip"].append(mm(f) - mm(c))
                deltas["qdq_minus_int8"].append(mm(q) - mm(i8))
            if deltas["float_minus_qdq"]:
                mech[f"{strat}_{budget}_d{draw}"] = {
                    k: {"mean": float(np.nanmean(v)),
                        "values": [None if np.isnan(x) else float(x) for x in v]}
                    for k, v in deltas.items()}
        out["mechanism"][comp] = mech

    out["class_support"] = json.loads((ev / "class_support.json").read_text())
    coverage = {}
    for sel_path in sorted((run / "selections").glob("*.json")):
        sel = json.loads(sel_path.read_text())
        cov = dict(sel["coverage"])
        cov.pop("windows_per_participant", None)  # participant-level stays local
        cov["sampler_time_s"] = sel.get("sampler_time_s")
        coverage[sel_path.stem] = cov
    out["selection_coverage"] = coverage

    out_path = run / "analysis" / "aggregate.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
