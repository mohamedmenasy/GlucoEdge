"""Round-three panel re-analyses (paper/analysis/panel-review-2026-10-18,
required items R6, R7, R9 and R10) from the published per-participant
confusion counts alone.

No cached prediction, checkpoint or label file is read: every number comes
from paper/results/calibration-coverage/participant_counts_*_r1.json.gz,
so anyone with the repository can rerun `python -m experiments.panel_r3`.
Intervals use paired_cluster_bootstrap_counts with each round's frozen
bootstrap seed. Differences are in points (x100); levels stay fractions.
"""
import gzip
import json

import numpy as np

from experiments.analyze_calibration_coverage import (
    ANALYSIS_SEED, MINORITY, STABLE, accuracy, macro_f1, minority_macro_recall,
    paired_cluster_bootstrap_counts, recalls)
from experiments.participant_counts import PUBLISHED
from training.labeling import FIVE_CLASSES

SWEEP_SEEDS = {"r1": ANALYSIS_SEED, "r2": 20261012, "r3": 20261013, "r5": 20261015}
CANDIDATE = "winner_seed{s}_int8_uniform_random_200_d101"


# --- metrics on 5x5 confusion counts (rows true, columns predicted) ----------


def qwk(cm):
    """Quadratic weighted Cohen's kappa of the ordinal five-class labels;
    NaN when chance agreement is perfect."""
    cm = np.asarray(cm, dtype=float)
    k = len(cm)
    i, j = np.indices(cm.shape)
    w = (i - j) ** 2 / (k - 1) ** 2
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / cm.sum()
    denom = (w * expected).sum()
    return float(1 - (w * cm).sum() / denom) if denom else float("nan")


def precisions(cm):
    """Per-class positive predictive value; NaN for a class never predicted."""
    cm = np.asarray(cm)
    predicted = cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(predicted > 0, np.diag(cm) / np.maximum(predicted, 1), np.nan)


def stable_recall(cm):
    return float(recalls(cm)[STABLE])


def profile(cm):
    """Summary of one pooled confusion matrix."""
    cm = np.asarray(cm)
    named = lambda v: {c: _num(x) for c, x in zip(FIVE_CLASSES, v)}
    return {"directional_recall": minority_macro_recall(cm), "stable_recall": stable_recall(cm),
            "macro_f1": macro_f1(cm), "qwk": qwk(cm), "accuracy": accuracy(cm),
            "recall": named(recalls(cm)), "precision": named(precisions(cm)),
            "predicted_share": named(cm.sum(axis=0) / cm.sum()),
            "prevalence": named(cm.sum(axis=1) / cm.sum())}


def _num(x):
    return None if x is None or np.isnan(x) else float(x)


# --- runner --------------------------------------------------------------------


def _counts():
    arms = {}
    for fam in ("glucobench", "diadata"):
        doc = json.loads(gzip.open(PUBLISHED / f"participant_counts_{fam}_r1.json.gz").read())
        for name, rec in doc["records"].items():
            arms[name] = {a: np.asarray(rows).reshape(len(rows), 5, 5)
                          for a, rows in rec["arms"].items()}
    return arms


def _pts(res):
    return {"delta": 100 * res["delta"],
            "ci95": None if res["ci95"] is None else [100 * v for v in res["ci95"]],
            "undefined_fraction": res["undefined_fraction"], "n_boot": res["n_boot"],
            "seed": res["seed"]}


def _contrast(arms, pairs, seed, metric=minority_macro_recall):
    return _pts(paired_cluster_bootstrap_counts([(arms[a], arms[b]) for a, b in pairs],
                                                metric=metric, seed=seed))


def _best_seed(r):
    stage = "D" if r == "r1" else "S"
    return json.loads((PUBLISHED / f"accuracy_sweep_{r}.json").read_text())[
        "selection_trail"][stage]["best_seed"]


def _gates(cand_int8, cand_float, champ_int8, contrast):
    """The frozen round-two/three gates on pooled counts (points)."""
    stable_change = 100 * (stable_recall(cand_int8) - stable_recall(champ_int8))
    fidelity = 100 * (minority_macro_recall(cand_int8) - minority_macro_recall(cand_float))
    return {"gate1_delta": contrast, "gate1": contrast["delta"] > 0 and contrast["ci95"][0] > -0.5,
            "stable_change": stable_change, "gate2": stable_change >= -2.0,
            "int8_minus_float": fidelity, "gate3": abs(fidelity) <= 0.5}


def per_seed_gates(counts):
    """R6: every candidate seed of rounds two and three against the champion
    artifact, and the five-seed means of the same quantities."""
    out = {}
    for r in ("r2", "r3"):
        seed = SWEEP_SEEDS[r]
        out[r] = {"chosen_seed": _best_seed(r)}
        for comp in ("held_out", "temporal"):
            arms = counts[f"sweep-{r}/{comp}"]
            pool = lambda a: arms[a].sum(axis=0)
            champ = pool("champion_int8")
            seeds = {}
            for s in range(5):
                c = CANDIDATE.format(s=s)
                seeds[str(s)] = _gates(pool(c), pool(f"winner_seed{s}_float"), champ,
                                       _contrast(arms, [(c, "champion_int8")], seed))
            mean = {"gate1_delta": _contrast(
                        arms, [(CANDIDATE.format(s=s), "champion_int8") for s in range(5)], seed)}
            for k in ("stable_change", "int8_minus_float"):
                mean[k] = float(np.mean([v[k] for v in seeds.values()]))
            mean["gate1"] = mean["gate1_delta"]["delta"] > 0 and mean["gate1_delta"]["ci95"][0] > -0.5
            mean["gate2"] = mean["stable_change"] >= -2.0
            mean["gate3"] = abs(mean["int8_minus_float"]) <= 0.5
            mean["all_gates"] = mean["gate1"] and mean["gate2"] and mean["gate3"]
            for v in seeds.values():
                v["all_gates"] = v["gate1"] and v["gate2"] and v["gate3"]
            out[r][comp] = {"per_seed": seeds, "five_seed_mean": mean,
                            "seeds_passing_all": [s for s, v in seeds.items() if v["all_gates"]]}
    return out


def _reads():
    """Every confirmatory read as (round, component record, candidate arm)."""
    reads = []
    for r in ("r1", "r2", "r3"):
        cand = CANDIDATE.format(s=_best_seed(r))
        for comp in ("held_out", "temporal"):
            reads.append((r, f"sweep-{r}/{comp}", cand))
        if r != "r1":  # the PEDAP probe cached only the chosen artifact
            reads.append((r, f"sweep-{r}/pedap", "winner_int8"))
    for comp in ("rtcgm", "held_out", "temporal", "pedap"):
        reads.append(("r5", f"sweep-r5/{comp}", "winner_int8"))
    return reads


def confirmatory_reads(counts):
    """R7 and R9: per-class recall and precision, QWK and macro-F1 of the
    candidate and champion INT8 artifacts in every confirmatory read, with
    paired intervals except on the descriptive PEDAP probe."""
    out = {}
    for r, name, cand in _reads():
        arms = counts[name]
        row = {"candidate_arm": cand,
               "candidate": profile(arms[cand].sum(axis=0)),
               "champion": profile(arms["champion_int8"].sum(axis=0))}
        if not name.endswith("/pedap"):
            for key, metric in (("qwk", qwk), ("macro_f1", macro_f1),
                                ("directional_recall", minority_macro_recall)):
                row[f"{key}_contrast"] = _contrast(arms, [(cand, "champion_int8")],
                                                   SWEEP_SEEDS[r], metric)
        out[f"{r}/{name.split('/')[1]}"] = row
    return out


def bundled_float_vs_int8(counts):
    """R10: the bundled round-five pair, float against INT8, on every
    component where both were evaluated."""
    out = {}
    for comp in ("held_out", "temporal", "rtcgm"):
        arms = counts[f"sweep-r5/{comp}"]
        f, q = arms["winner_float"].sum(axis=0), arms["winner_int8"].sum(axis=0)
        out[comp] = {"float": profile(f), "int8": profile(q),
                     "float_confusion": f.tolist(), "int8_confusion": q.tolist(),
                     "qwk_int8_minus_float": _contrast(arms, [("winner_int8", "winner_float")],
                                                       SWEEP_SEEDS["r5"], qwk)}
    return out


# Class counts of round two's DiaData-only training mixture (stage M, arm M3:
# 15,135,091 windows), read once from the local training tensor cache
# results/calibration-coverage/sweep-r2/tensors/M_M3_s20261012.npz.
DIADATA_TRAIN_COUNTS = dict(zip(FIVE_CLASSES, (429907, 1403869, 11324382, 1371132, 605801)))


def _ratios(counts):
    n = np.array([counts[c] for c in FIVE_CLASSES], dtype=float)
    return {c: float(n[STABLE] / v) for c, v in zip(FIVE_CLASSES, n)}


def main():
    counts = _counts()
    train = json.loads((PUBLISHED / "data_availability.json").read_text())["training"]["class_counts"]
    out = {
        "generated_by": "python -m experiments.panel_r3",
        "review": "paper/analysis/panel-review-2026-10-18/synthesis.md (R6, R7, R9, R10)",
        "source": "paper/results/calibration-coverage/participant_counts_*_r1.json.gz only",
        "conventions": {
            "differences": "points (x100)", "levels": "fractions",
            "directional_recall": "mean recall of the four non-stable classes",
            "qwk": "quadratic weighted Cohen's kappa over the ordered five classes",
            "bootstrap": "paired_cluster_bootstrap_counts, 2,000 draws, each round's frozen seed",
            "classes": list(FIVE_CLASSES)},
        "per_seed_gates": per_seed_gates(counts),
        "confirmatory_reads": confirmatory_reads(counts),
        "bundled_float_vs_int8": bundled_float_vs_int8(counts),
        "class_weight_ratio_to_stable": {
            "weinstock_eligible_training": _ratios(train),
            "diadata_round_two_mixture": _ratios(DIADATA_TRAIN_COUNTS),
            "note": ("CE weight of each class divided by the stable weight; balanced weights are "
                     "N/(5 n_c), so the ratio is n_stable / n_c")},
    }
    path = PUBLISHED / "panel_r3_r1.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {path.relative_to(PUBLISHED.parents[2])}")


if __name__ == "__main__":
    main()
