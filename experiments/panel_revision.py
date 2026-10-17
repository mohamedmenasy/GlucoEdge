"""Panel-revision re-analyses from cached results (plan
docs/superpowers/plans/2026-10-16-paper-panel-revision.md, Task 1).

No new inference: every prediction is a cached .npy from an earlier frozen
evaluation. The pure numpy helpers at the top are unit-tested on synthetic
arrays; `main()` reads the local caches under results/ and writes
paper/results/calibration-coverage/panel_revision_r1.json and
artifact_hashes_r1.json. Differences are in recall points (x100); recall
levels stay fractions, as in the paper.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.analyze_calibration_coverage import (
    ANALYSIS_SEED, MINORITY, _clusters, confusion_counts, minority_macro_recall,
    paired_cluster_bootstrap, recalls, resample_rows)
from experiments.calibration_coverage import (DRAWS, GLUCOEDGE_ROOT, RESULTS,
                                              TRAINING_SEEDS)
from training.labeling import FIVE_CLASSES

FALLING = (0, 1)
RISING = (3, 4)
CODES = (39.0, 401.0)  # candidate sensor LOW/HIGH censoring codes (roadmap S20)
PUBLISHED = GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage"
UNITS = "recall points (x100)"
RANGE, BALANCED = "participant_balanced_range_coverage", "participant_balanced_random"


def _num(x):
    """JSON-safe float: NaN (undefined) becomes None, never zero."""
    return None if x is None or np.isnan(x) else float(x)


# --- per-participant contributions and exact sign-flip test ------------------


def participant_contributions(y_true, participant_ids, pred_a, pred_b, classes=MINORITY):
    """Additive per-participant shares (recall points x100) of the pooled
    mean-recall difference A - B over `classes`. Each class term divides by
    the pooled class support, so the shares sum to the pooled difference."""
    y, pids = np.asarray(y_true), np.asarray(participant_ids)
    hit_a = np.asarray(pred_a) == y
    hit_b = np.asarray(pred_b) == y
    support = np.array([(y == c).sum() for c in classes], dtype=float)
    out = {}
    for p in np.unique(pids):
        m = pids == p
        gained = np.array([(hit_a & m & (y == c)).sum() - (hit_b & m & (y == c)).sum()
                           for c in classes], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[str(p)] = float(100 * np.mean(gained / support))  # NaN on zero support
    return out


def mean_contributions(y_true, participant_ids, pred_pairs, classes=MINORITY):
    """participant_contributions averaged over (seed x draw) pairs, as
    paired_cluster_bootstrap averages its point estimate."""
    per = [participant_contributions(y_true, participant_ids, a, b, classes)
           for a, b in pred_pairs]
    return {p: float(np.mean([d[p] for d in per])) for p in per[0]}


def sign_flip_test(contribs):
    """Exact two-sided sign-flip randomization test of the summed
    per-participant contributions over all 2^G sign patterns (G <= 20)."""
    c = np.asarray(contribs, dtype=float)
    g = len(c)
    if g > 20:
        raise ValueError(f"exact enumeration needs G <= 20, got {g}")
    signs = 1 - 2 * ((np.arange(2 ** g)[:, None] >> np.arange(g)) & 1)
    observed = float(c.sum())
    p = float(np.mean(np.abs(signs @ c) >= abs(observed) - 1e-9))  # ties count
    return {"observed": observed, "p_two_sided": p, "n_patterns": 2 ** g}


# --- participant-cluster bootstrap for any statistic --------------------------


def cluster_bootstrap_stat(y_true, participant_ids, stat_fn, n_boot=2000,
                           seed=ANALYSIS_SEED):
    """paired_cluster_bootstrap's draws, percentile 95% interval and
    >5%-undefined rule for any statistic. stat_fn(y, rows) -> float, NaN
    when undefined; rows index the original windows and y = y_true[rows]."""
    y_true = np.asarray(y_true)
    clusters = _clusters(participant_ids)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        rows = resample_rows(participant_ids, rng, clusters)
        boot.append(stat_fn(y_true[rows], rows))
    boot = np.array(boot, dtype=float)
    undefined = float(np.mean(np.isnan(boot)))
    defined = boot[~np.isnan(boot)]
    ci = None
    if undefined <= 0.05 and len(defined):
        ci = [float(v) for v in np.percentile(defined, [2.5, 97.5])]
    rows = np.arange(len(y_true))
    return {"point": _num(stat_fn(y_true, rows)), "ci95": ci,
            "undefined_fraction": undefined, "n_boot": n_boot, "seed": seed}


# --- censoring codes, error profile, level bands ------------------------------


def _shares(mask, labels):
    return {"window_fraction": float(mask.mean()),
            "by_true_class": {str(c): float(mask[labels == c].mean())
                              for c in np.unique(labels)}}


def censor_shares(windows, labels, endpoint=None, codes=(39.0, 401.0)):
    """Fraction of windows with any input reading (and, if given, the label
    endpoint) at a sensor censoring code, overall and by true class."""
    windows, labels = np.asarray(windows), np.asarray(labels)
    inputs = np.isin(windows, codes).any(axis=1)
    out = {"codes": [float(c) for c in codes], "n_windows": int(len(labels)),
           "inputs": _shares(inputs, labels)}
    out["inputs"]["per_code"] = {str(c): float((windows == c).any(axis=1).mean())
                                 for c in codes}
    if endpoint is not None:
        e = np.isin(np.asarray(endpoint).reshape(len(labels), -1), codes).any(axis=1)
        out["endpoint"] = _shares(e, labels)
        out["inputs_or_endpoint"] = _shares(inputs | e, labels)
    return out


def confusion_summary(y_true, y_pred):
    """5x5 confusion matrix (rows true, columns predicted) with per-class
    recall (sensitivity), precision (positive predictive value) and F1, the
    share of errors landing in an adjacent class, and direction-reversal
    shares: true falling/falling_fast predicted rising/rising_fast, and the
    mirror, per true class and pooled per side."""
    cm = confusion_counts(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1_den = support + predicted  # 2tp + fp + fn
        f1 = np.where(f1_den > 0, 2 * tp / f1_den, 0.0)  # macro_f1's convention
    i, j = np.indices(cm.shape)
    errors = cm.sum() - tp.sum()
    reversal = {}
    for c in FALLING:
        reversal[FIVE_CLASSES[c]] = _num(cm[c, list(RISING)].sum() / support[c]) if support[c] else None
    for c in RISING:
        reversal[FIVE_CLASSES[c]] = _num(cm[c, list(FALLING)].sum() / support[c]) if support[c] else None
    for side, src, dst in (("falling_side", FALLING, RISING), ("rising_side", RISING, FALLING)):
        n = support[list(src)].sum()
        reversal[side] = _num(cm[np.ix_(src, dst)].sum() / n) if n else None
    named = lambda v: {c: _num(x) for c, x in zip(FIVE_CLASSES, v)}
    return {
        "classes": list(FIVE_CLASSES),
        "matrix": cm.tolist(),
        "support": [int(s) for s in support],
        "recall_sensitivity": named(recalls(cm)),
        "precision_ppv": named(precision),
        "f1": named(f1),
        "adjacent_error_share": _num(cm[np.abs(i - j) == 1].sum() / errors) if errors else None,
        "reversal_share": reversal,
    }


def level_stratified_recall(y_true, y_pred, anchor_mgdl, edges=(70, 180, 250)):
    """Directional recall, per-class recall and support within bands of the
    anchor (last input) glucose level. Bands are left-closed: [70, 180) etc."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    band = np.digitize(np.asarray(anchor_mgdl), edges)
    names = ([f"<{edges[0]}"] + [f"{lo}-{hi}" for lo, hi in zip(edges, edges[1:])]
             + [f">={edges[-1]}"])
    out = {}
    for k, name in enumerate(names):
        m = band == k
        cm = confusion_counts(y[m], p[m])
        out[name] = {"n_windows": int(m.sum()),
                     "support": [int(s) for s in cm.sum(axis=1)],
                     "per_class_recall": [_num(r) for r in recalls(cm)],
                     "directional_recall": _num(minority_macro_recall(cm))}
    return out


def levels_from_deltas(w, anchor_scale=16.0):
    """Inverse of training.deltas.delta_windows(anchor=True): (N, L) deltas
    plus last level / anchor_scale -> (N, L) absolute mg/dL."""
    w = np.asarray(w, dtype=np.float32)
    last = w[:, -1:] * np.float32(anchor_scale)
    back = np.cumsum(w[:, -2::-1], axis=1)[:, ::-1]  # sum of deltas after each step
    return np.concatenate([last - back, last], axis=1)


# --- runner (local caches only; not unit-tested) ------------------------------


def _rel(path):
    return str(Path(path).relative_to(GLUCOEDGE_ROOT))


def _read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=None)
def _npy(path):
    """Cached predictions; one object per file, so stats can memoize by id."""
    return np.load(path)


def _mmr(y, pred):
    return minority_macro_recall(confusion_counts(y, pred))


def _pts(res):
    """A bootstrap result restated in recall points (x100)."""
    point = res["point"] if "point" in res else res["delta"]
    return {"point": None if point is None else 100 * point,
            "ci95": None if res["ci95"] is None else [100 * v for v in res["ci95"]],
            "undefined_fraction": res["undefined_fraction"],
            "n_boot": res["n_boot"], "seed": res["seed"], "units": UNITS}


def _linear_contrast(y, pids, terms, n_boot):
    """Cluster bootstrap of the (seed x draw) mean of sum(sign * directional
    recall); `terms` holds one [(sign, preds), ...] list per pair."""
    def stat(yy, rows):
        memo = {}

        def mmr(pred):
            if id(pred) not in memo:
                memo[id(pred)] = _mmr(yy, pred[rows])
            return memo[id(pred)]
        return float(np.mean([sum(sign * mmr(p) for sign, p in t) for t in terms]))
    return cluster_bootstrap_stat(y, pids, stat, n_boot=n_boot)


def rebuild_eval_components():
    """Weinstock held-out/temporal eligible windows rebuilt with the study
    loader, with label endpoints (the reading 15 minutes ahead) and unique
    input readings. Refuses unless the windows equal the cached evaluation
    inputs and the keys equal the round-five read's keys, in order."""
    from experiments.calibration_coverage import _eligible, _load_study_data
    datasets, owner, no_gap_fill = _load_study_data("weinstock")
    t = datasets["test"]
    cached = np.load(RESULTS / "r1" / "eval" / "components.npz")
    out = {}
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        idx = _eligible(t, owner, no_gap_fill, split)
        windows = np.stack([t.windows[i] for i in idx])
        keys = [[str(t.participants[i]), str(t.anchor_times[i])] for i in idx]
        r5_keys = _read(RESULTS / "sweep-r5" / "eval" / comp / "meta.json")["keys"]
        if not np.array_equal(windows, cached[f"{comp}_windows"]) or keys != r5_keys:
            raise SystemExit(f"REFUSED: rebuilt {comp} windows differ from the cached evaluation")
        rows = np.concatenate([t.row_ids[i][:t.input_length] for i in idx])
        _, first = np.unique(rows, return_index=True)
        out[comp] = {"windows": windows,
                     "labels": cached[f"{comp}_labels"],
                     "participants": cached[f"{comp}_participants"],
                     "endpoint": np.array([t.horizons[i][-1] for i in idx], dtype=np.float32),
                     "unique_input_readings": windows.reshape(-1)[first]}
    return out


def _weinstock(n_boot):
    ev_root = RESULTS / "r1" / "eval"
    comps = np.load(ev_root / "components.npz")
    preds_src = "results/calibration-coverage/r1/eval/<component>/seed*_*.npy"
    out = {"seed_draw_variation": {}, "interaction": {}, "sequential_1000_vs_200": {}}
    for comp in ("held_out", "temporal"):
        ev = ev_root / comp
        y, pids = comps[f"{comp}_labels"], comps[f"{comp}_participants"]

        def arm(strategy, budget, ev=ev):
            if strategy == "sequential":
                return lambda s, d: _npy(ev / f"seed{s}_int8_sequential_{budget}_d0.npy")
            return lambda s, d: _npy(ev / f"seed{s}_int8_{strategy}_{budget}_d{d}.npy")

        seq, uni = arm("sequential", 200), arm("uniform_random", 200)
        per_seed = {}
        for s in TRAINING_SEEDS:
            u = [_mmr(y, uni(s, d)) for d in DRAWS]
            per_seed[str(s)] = {
                "float_directional_recall": _mmr(y, _npy(ev / f"seed{s}_float.npy")),
                "uniform200_directional_recall_by_draw": dict(zip(map(str, DRAWS), u)),
                "uniform200_minus_sequential200_pts": 100 * (np.mean(u) - _mmr(y, seq(s, 0))),
                "uniform200_within_seed_draw_sd_pts": 100 * float(np.std(u, ddof=1)),
            }
        eff = [v["uniform200_minus_sequential200_pts"] for v in per_seed.values()]
        flt = [v["float_directional_recall"] for v in per_seed.values()]
        out["seed_draw_variation"][comp] = {
            "per_seed": per_seed,
            "effect_mean_pts": float(np.mean(eff)),
            "effect_range_pts": [float(min(eff)), float(max(eff))],
            "effect_across_seed_sd_pts": float(np.std(eff, ddof=1)),
            "float_across_seed_sd_pts": 100 * float(np.std(flt, ddof=1)),
            "sd": "sample SD (ddof=1); the effect mean equals the published contrast point",
            "sources": [preds_src, "results/calibration-coverage/r1/eval/components.npz"]}

        def budget_by_strategy(a, b, arm=arm, y=y, pids=pids):
            """(A1000 - A200) - (B1000 - B200), mean over (seed x draw)."""
            return _pts(_linear_contrast(y, pids, [
                [(1, arm(a, 1000)(s, d)), (-1, arm(a, 200)(s, d)),
                 (-1, arm(b, 1000)(s, d)), (1, arm(b, 200)(s, d))]
                for s in TRAINING_SEEDS for d in DRAWS], n_boot))

        out["interaction"][comp] = {
            "uniform_vs_sequential": budget_by_strategy("uniform_random", "sequential"),
            "randomized_pairs": {
                "range_minus_balanced": budget_by_strategy(RANGE, BALANCED),
                "range_minus_uniform": budget_by_strategy(RANGE, "uniform_random"),
                "balanced_minus_uniform": budget_by_strategy(BALANCED, "uniform_random")}}
        out["sequential_1000_vs_200"][comp] = _pts(paired_cluster_bootstrap(
            y, pids, [(arm("sequential", 1000)(s, 0), arm("sequential", 200)(s, 0))
                      for s in TRAINING_SEEDS], n_boot=n_boot))
        print(f"weinstock {comp} done", flush=True)
    out["interaction"]["definition"] = (
        "(A1000 - A200) - (B1000 - B200) in directional recall, averaged over the "
        "15 (seed x draw) pairs; sequential has one order per seed, reused for "
        "every draw as in the published contrasts, so uniform_vs_sequential "
        "equals (U1000 - S1000) - (U200 - S200)")
    out["interaction"]["protocol_note"] = (
        "protocol.json lists 'budget-by-strategy interaction' among the secondary "
        "contrasts without a formula; the definition above was fixed here, after "
        "the primary analysis was published")
    out["interaction"]["sources"] = [preds_src, "paper/results/calibration-coverage/protocol.json"]
    out["sequential_1000_vs_200"]["note"] = "five seed pairs (sequential has one order); same bootstrap as the published contrasts"
    out["sequential_1000_vs_200"]["sources"] = [preds_src]
    return out


def _value_counts(values, lo, hi):
    v, c = np.unique(values[(values >= lo) & (values <= hi)], return_counts=True)
    return {f"{x:g}": int(n) for x, n in zip(v, c)}


def _censoring(n_boot, rebuilt):
    ev = RESULTS / "r1" / "eval" / "held_out"
    out = {"code_evidence": {}, "shares": {}}
    for comp, r in rebuilt.items():
        v = r["unique_input_readings"]
        out["code_evidence"][comp] = {
            "unique_input_readings": int(len(v)), "min_mgdl": float(v.min()),
            "max_mgdl": float(v.max()), "below_39": int((v < 39).sum()),
            "above_401": int((v > 401).sum()),
            "counts_38_to_41_mgdl": _value_counts(v, 38, 41),
            "counts_399_to_402_mgdl": _value_counts(v, 399, 402)}
        out["shares"][comp] = censor_shares(r["windows"], r["labels"], r["endpoint"], CODES)
    r = rebuilt["held_out"]
    anchor, endpoint = np.isin(r["windows"][:, -1], CODES), np.isin(r["endpoint"], CODES)
    pairs = [(_npy(ev / f"seed{s}_int8_uniform_random_200_d{d}.npy"),
              _npy(ev / f"seed{s}_int8_sequential_200_d0.npy"))
             for s in TRAINING_SEEDS for d in DRAWS]
    published = _read(PUBLISHED / "aggregate_r1.json")["contrasts"]["held_out"]["uniform200_vs_sequential200"]
    contrast = {"published_all_windows": _pts(published)}
    for name, drop in (("excluding_coded_anchor", anchor),
                       ("excluding_coded_anchor_or_endpoint", anchor | endpoint)):
        keep = ~drop
        res = _pts(paired_cluster_bootstrap(r["labels"][keep], r["participants"][keep],
                                            [(a[keep], b[keep]) for a, b in pairs], n_boot=n_boot))
        contrast[name] = dict(res, n_windows=int(keep.sum()), n_excluded=int(drop.sum()))
    out["held_out_uniform200_vs_sequential200"] = contrast
    out["definitions"] = {
        "anchor": "last input reading g_t", "endpoint": "label endpoint g_{t+15 min}",
        "unique_input_readings": "distinct source rows among the eligible windows' inputs"}
    out["sources"] = ["GlucoBench/raw_data/weinstock.csv via experiments.calibration_coverage."
                      "_load_study_data (rebuilt windows verified equal to results/calibration-"
                      "coverage/r1/eval/components.npz and keys equal to sweep-r5/eval/*/meta.json)",
                      "results/calibration-coverage/r1/eval/held_out/seed*_int8_{uniform_random,sequential}_200_*.npy",
                      "paper/results/calibration-coverage/aggregate_r1.json"]
    print("censoring done", flush=True)
    return out


def _facts():
    ev_root = RESULTS / "r1" / "eval"
    comps = np.load(ev_root / "components.npz")
    observer, flat = {}, []
    for comp in ("held_out", "temporal"):
        y = comps[f"{comp}_labels"]
        observer[comp] = {}
        for cell in ("sequential_200_d0", f"{RANGE}_200_d101"):
            d = [100 * (_mmr(y, _npy(ev_root / comp / f"seed{s}_int8_{cell}_obsminmax.npy"))
                        - _mmr(y, _npy(ev_root / comp / f"seed{s}_int8_{cell}.npy")))
                 for s in TRAINING_SEEDS]
            observer[comp][cell] = {"seed_mean_minmax_minus_histogram_pts": float(np.mean(d)),
                                    "per_seed_pts": [float(x) for x in d]}
            flat += [(abs(x), comp, cell, s) for s, x in zip(TRAINING_SEEDS, d)]
    means = [(abs(v["seed_mean_minmax_minus_histogram_pts"]), comp, cell)
             for comp, cells in observer.items() for cell, v in cells.items()]
    top_mean, top_seed = max(means), max(flat)
    observer.update(
        max_abs_seed_mean_pts={"value": top_mean[0], "component": top_mean[1], "cell": top_mean[2]},
        max_abs_single_seed_pts={"value": top_seed[0], "component": top_seed[1],
                                 "cell": top_seed[2], "seed": top_seed[3]},
        sources=["results/calibration-coverage/r1/eval/<component>/seed*_int8_*_obsminmax.npy"])

    coverage = _read(PUBLISHED / "aggregate_r1.json")["selection_coverage"]
    rand200 = {k: v["participants"] for k, v in coverage.items()
               if "_200_" in k and not k.startswith("sequential")}
    manifest = _read(RESULTS / "r1" / "artifacts_manifest.json")
    input_ranges = {}
    for name, m in manifest.items():
        if "int8" in name and "obsminmax" not in name:
            cell = name.split("_int8_")[1].rsplit("_d", 1)[0]
            scale, zp = m["input_quantization"]
            input_ranges.setdefault(cell, set()).add(round(scale * (127 - zp), 1))
    tiers = _read(PUBLISHED / "sensitivity_tiers_r1.json")["tiers"]
    widest = {}
    for comp, ts in tiers.items():
        for contrast in ("uniform200_vs_sequential200", "float_vs_int8_uniform200"):
            name, e = max(ts.items(), key=lambda kv: np.diff(kv[1]["contrasts"][contrast]["ci95"])[0])
            c = e["contrasts"][contrast]
            widest.setdefault(comp, {})[contrast] = {
                "tier": name, "point_pts": 100 * c["delta"],
                "ci95_pts": [100 * v for v in c["ci95"]],
                "width_pts": 100 * (c["ci95"][1] - c["ci95"][0])}
    return {
        "observer": observer,
        "randomized_200_participants": {
            "min": min(rand200.values()), "max": max(rand200.values()), "per_selection": rand200,
            "sources": ["paper/results/calibration-coverage/aggregate_r1.json (selection_coverage)"]},
        "int8_input_ceiling_mgdl_by_cell": {
            "values": {k: sorted(v) for k, v in sorted(input_ranges.items())},
            "note": "top of the INT8 input range, scale * (127 - zero_point), across seeds and draws",
            "sources": ["results/calibration-coverage/r1/artifacts_manifest.json"]},
        "widest_single_tier_interval": dict(widest, sources=[
            "paper/results/calibration-coverage/sensitivity_tiers_r1.json"]),
    }


CLIP_NOTE = ("share of windows with at least one input slot whose rounded code falls "
             "outside the int8 range; every range starts at 0 mg/dL, so this is the "
             "share of windows with an input above the artifact's ceiling")


def _clip_fractions(ev):
    """Rounded-code clipped-window share per calibration cell, overall and
    for falling_fast, from the cached per-artifact meta (mean over seeds)."""
    cells = {}
    for cell in ["sequential_200_d0"] + [f"uniform_random_200_d{d}" for d in DRAWS]:
        metas = [_read(ev / f"seed{s}_int8_{cell}_meta.json") for s in TRAINING_SEEDS]
        overall = [m["clipping"]["window_fraction"] for m in metas]
        ff = [m["clipping"]["by_true_class"].get("0", {}).get("window_fraction") for m in metas]
        scale, zp = metas[0]["quantization"]
        cells[cell] = {"window_fraction": float(np.mean(overall)),
                       "falling_fast_window_fraction": None if None in ff else float(np.mean(ff)),
                       "identical_across_seeds": len(set(overall)) == 1 and len(set(ff)) == 1,
                       "input_ceiling_mgdl": round(scale * (127 - zp), 1)}
    seq = cells["sequential_200_d0"]
    uni, diff = {}, {}
    for k in ("window_fraction", "falling_fast_window_fraction"):
        vals = [cells[f"uniform_random_200_d{d}"][k] for d in DRAWS]
        uni[k] = None if None in vals else float(np.mean(vals))
        diff[k] = None if None in (uni[k], seq[k]) else seq[k] - uni[k]
    return {"by_cell": cells, "uniform_200_mean_over_draws": uni,
            "sequential_minus_uniform_200": diff, "definition": CLIP_NOTE}


def _confirmation(cohort):
    run = RESULTS / f"conf-{cohort}"
    pub_path = PUBLISHED / f"confirmation_{cohort}.json"
    pub = _read(pub_path)["contrast_uniform200_vs_sequential200"]
    comps = np.load(run / "eval" / "components.npz")
    out = {}
    for comp in ("temporal", "held_out"):
        ev = run / "eval" / comp
        y, pids = comps[f"{comp}_labels"], comps[f"{comp}_participants"]
        pairs = [(np.load(ev / f"seed{s}_int8_uniform_random_200_d{d}.npy"),
                  np.load(ev / f"seed{s}_int8_sequential_200_d0.npy"))
                 for s in TRAINING_SEEDS for d in DRAWS]
        ordered = sorted(mean_contributions(y, pids, pairs).values(), reverse=True)
        g, total = len(ordered), float(np.sum(ordered))
        if abs(total - 100 * pub[comp]["delta"]) > 1e-6:
            raise SystemExit(f"REFUSED: {cohort} {comp} contributions do not sum to the published delta")
        entry = {
            "n_participants": g,
            "per_participant_mean_contribution_pts": {f"P{k + 1}": v for k, v in enumerate(ordered)},
            "participant_labels": "anonymized, ordered by descending contribution",
            "sum_of_contributions_pts": total,
            "published_delta_pts": 100 * pub[comp]["delta"],
            "published_percentile_ci95_pts": (None if pub[comp]["ci95"] is None
                                              else [100 * v for v in pub[comp]["ci95"]]),
            "undefined_fraction": pub[comp]["undefined_fraction"],
            "class_support": dict(zip(FIVE_CLASSES, map(int, np.bincount(y, minlength=5)))),
            "clipping": _clip_fractions(ev),
        }
        if g <= 20:
            entry["sign_flip_test"] = dict(sign_flip_test(ordered), min_attainable_p=2 / 2 ** g, statistic=(
                "sum of per-participant mean contributions (points); exact over all sign patterns"))
            if g == 1:
                entry["sign_flip_test"]["note"] = "one participant: p = 1 by construction"
        else:
            entry["sign_flip_test"] = None
            entry["sign_flip_reason"] = (f"{g} clusters: exact enumeration is limited to G <= 20; "
                                         "the published percentile cluster bootstrap applies at this size")
        entry["sources"] = [f"results/calibration-coverage/conf-{cohort}/eval/{comp}/seed*_int8_*_200_*.npy",
                            f"results/calibration-coverage/conf-{cohort}/eval/{comp}/seed*_int8_*_meta.json",
                            f"results/calibration-coverage/conf-{cohort}/eval/components.npz", _rel(pub_path)]
        out[comp] = entry
    return out


def _shipped_model(rebuilt):
    from experiments.accuracy_sweep_r5 import (BOOT_SEED_R5, RTCGM_CACHE, _champion_paths,
                                            _sha256)
    r5 = RESULTS / "sweep-r5"
    best = _read(r5 / "selection" / "S.json")["best_seed"]
    art = r5 / "artifacts" / f"winner_seed{best}_int8_uniform_random_200_d101.tflite"
    bundled = GLUCOEDGE_ROOT / "android" / "app" / "src" / "main" / "assets" / "trend_int8.tflite"
    identity = {"artifact": _rel(art), "seed": best,
                "sha256_equals_bundled_trend_int8": _sha256(art) == _sha256(bundled)}

    meta = _read(r5 / "eval" / "held_out" / "meta.json")
    y = np.array(meta["labels"])
    pred = np.load(r5 / "eval" / "held_out" / "winner_int8_preds.npy")
    held_out = {"identity": identity, "confusion": confusion_summary(y, pred),
                "level_stratified": {
                    "bands": level_stratified_recall(y, pred, rebuilt["held_out"]["windows"][:, -1]),
                    "anchor": "last input reading (mg/dL), rebuilt with the study loader; "
                              "keys verified equal to the read's"},
                "sources": ["results/calibration-coverage/sweep-r5/eval/held_out/{meta.json,winner_int8_preds.npy}",
                            "GlucoBench weinstock via experiments.calibration_coverage._load_study_data"]}

    meta = np.load(r5 / "eval" / "rtcgm" / "meta.npz")
    y = meta["y"]
    pred = np.load(r5 / "eval" / "rtcgm" / "winner_int8_preds.npy")
    cache = np.load(RTCGM_CACHE)
    if not (np.array_equal(cache["y"], y) and np.array_equal(cache["pids"], meta["pids"])):
        raise SystemExit("REFUSED: RT-CGM hours cache no longer aligns with the round-five read")
    anchor = cache["w"][:, -1] * np.float32(16.0)  # delta-anchor channel = g_t / 16, exact
    levels = levels_from_deltas(cache["w"])
    keep = ((levels >= 40) & (levels <= 400)).all(axis=1)
    champ = np.load(r5 / "eval" / "rtcgm" / "champion_int8_preds.npy")
    without = paired_cluster_bootstrap(y[keep], meta["pids"][keep], [(pred[keep], champ[keep])],
                                       n_boot=2000, seed=BOOT_SEED_R5)
    clipping = {}
    for name, path in (("winner_int8", art), ("champion_int8", _champion_paths()["int8"])):
        q = _input_qparams(path)
        lo, hi = q["range"]
        clipping[name] = {"input_qparams": q,
                          "window_fraction_any_channel_outside": float(
                              ((cache["w"] < lo) | (cache["w"] > hi)).any(axis=1).mean()),
                          "anchor_fraction_above_level_ceiling": float(
                              (anchor > q["level_ceiling_mgdl"]).mean())}
    rtcgm = {"identity": identity, "confusion": confusion_summary(y, pred),
             "level_stratified": {
                 "bands": level_stratified_recall(y, pred, anchor),
                 "anchor": "last input reading (mg/dL) = 16 x the delta-anchor channel of the cached "
                           "eval inputs; cache rows verified equal to the read's labels and participants"},
             "anchor_range_mgdl": [float(anchor.min()), float(anchor.max())],
             "anchor_fraction_below_40": float((anchor < 40).mean()),
             "anchor_fraction_above_400": float((anchor > 400).mean()),
             "input_clipping": clipping,
             "readings_outside_40_400": {
                 "window_fraction": float((~keep).mean()),
                 "participants_affected": int(len(np.unique(meta["pids"][~keep]))),
                 "gate1_contrast_without": _pts(without),
                 "note": "round five's gate-1 contrast (winner INT8 - champion INT8, directional recall) "
                         "recomputed on the windows whose 12 input readings all lie in 40-400 mg/dL; "
                         "draws and seed as in the read (BOOT_SEED_R5)"},
             "sources": ["results/calibration-coverage/sweep-r5/eval/rtcgm/"
                         "{meta.npz,winner_int8_preds.npy,champion_int8_preds.npy}",
                         _rel(RTCGM_CACHE)]}
    return {"held_out": held_out, "rtcgm": rtcgm}


def _round_one_f1():
    from experiments.accuracy_sweep import paired_rows
    run = RESULTS / "sweep-r1"
    sel = _read(run / "selection" / "D.json")
    if sel["winner"].get("ensemble"):
        raise SystemExit("REFUSED: round-one winner is an ensemble; per-seed logits do not apply")
    pub = _read(PUBLISHED / "accuracy_sweep_r1.json")
    out = {}
    for comp in ("held_out", "temporal"):
        base = run / "eval" / comp
        w_meta, c_meta = _read(base / "winner_meta.json"), _read(base / "champion_meta.json")
        ia, ib = paired_rows([tuple(k) for k in w_meta["keys"]], [tuple(k) for k in c_meta["keys"]])
        y = np.array(w_meta["labels"])[ia]
        if not np.array_equal(y, np.array(c_meta["labels"])[ib]):
            raise SystemExit(f"REFUSED: round-one {comp} labels disagree after pairing")
        win = np.load(base / f"winner_seed{sel['best_seed']}_int8_uniform_random_200_d101_logits.npy"
                      ).argmax(axis=1)[ia]
        champ = np.load(base / "champion_int8_preds.npy")[ib]
        sw, sc = confusion_summary(y, win), confusion_summary(y, champ)
        keep = ("recall_sensitivity", "precision_ppv", "f1")
        df1 = {c: 100 * (sw["f1"][c] - sc["f1"][c]) for c in FIVE_CLASSES}
        published = pub["post_hoc_descriptive_recall"][comp]["winner_int8"]
        out[comp] = {
            "paired_windows": len(ia), "candidate": f"seed {sel['best_seed']}, uniform INT8",
            "winner_int8": {k: sw[k] for k in keep}, "champion_int8": {k: sc[k] for k in keep},
            "macro_f1": {"winner_int8": float(np.mean(list(sw["f1"].values()))),
                         "champion_int8": float(np.mean(list(sc["f1"].values())))},
            "f1_change_pts": df1,
            "macro_f1_change_pts": float(np.mean(list(df1.values()))),
            "decomposition": "macro-F1 change = mean of the five per-class F1 changes",
            "recall_matches_published": all(round(sw["recall_sensitivity"][c], 4) == published[c]
                                            for c in FIVE_CLASSES),
            "sources": [f"results/calibration-coverage/sweep-r1/eval/{comp}/{{winner_meta.json,champion_meta.json,"
                        "champion_int8_preds.npy,winner_seed*_int8_uniform_random_200_d101_logits.npy}",
                        "results/calibration-coverage/sweep-r1/selection/D.json",
                        "paper/results/calibration-coverage/accuracy_sweep_r1.json"]}
    return out


def _pedap_per_class():
    base = RESULTS / "sweep-r5" / "eval" / "pedap"
    y = np.load(base / "meta.npz")["y"]
    sw = confusion_summary(y, np.load(base / "winner_int8_preds.npy"))
    sc = confusion_summary(y, np.load(base / "champion_int8_preds.npy"))
    pub = _read(PUBLISHED / "accuracy_sweep_r5.json")["pedap_zero_shot"]
    share = lambda s: dict(zip(FIVE_CLASSES, (np.sum(s["matrix"], axis=0) / len(y)).tolist()))
    directional = lambda s: float(np.mean([s["recall_sensitivity"][FIVE_CLASSES[c]] for c in MINORITY]))
    return {
        "windows": int(len(y)),
        "recall_winner_int8": sw["recall_sensitivity"], "recall_champion_int8": sc["recall_sensitivity"],
        "recall_change_pts": {c: 100 * (sw["recall_sensitivity"][c] - sc["recall_sensitivity"][c])
                              for c in FIVE_CLASSES},
        "precision_winner_int8": sw["precision_ppv"], "precision_champion_int8": sc["precision_ppv"],
        "predicted_share_winner_int8": share(sw), "predicted_share_champion_int8": share(sc),
        "directional_recall": {"winner_int8": directional(sw), "champion_int8": directional(sc)},
        "recall_matches_published": all(
            abs(sw["recall_sensitivity"][c] - pub["winner_int8"]["per_class"][c]) < 1e-12
            and abs(sc["recall_sensitivity"][c] - pub["champion_int8"]["per_class"][c]) < 1e-12
            for c in FIVE_CLASSES),
        "note": "descriptive; round-five winner (bundled) vs the previous bundled champion",
        "sources": ["results/calibration-coverage/sweep-r5/eval/pedap/{meta.npz,winner_int8_preds.npy,"
                    "champion_int8_preds.npy}", "paper/results/calibration-coverage/accuracy_sweep_r5.json"]}


def _interpreter(path):
    """Builtin resolver without the default XNNPACK delegate, whose DELEGATE
    node is a runtime rewrite and not part of the model graph."""
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(model_path=str(path), experimental_op_resolver_type=(
        OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES))
    it.allocate_tensors()
    return it


def _input_qparams(path):
    """INT8 input tensor's scale, zero point and representable float range;
    for a delta+anchor model, level_ceiling_mgdl = range max x ANCHOR_SCALE."""
    t = _interpreter(path).get_input_details()[0]
    scale, zp = t["quantization"]
    lo, hi = (-128 - zp) * scale, (127 - zp) * scale
    return {"scale": round(float(scale), 4), "zero_point": int(zp),
            "range": [round(float(lo), 2), round(float(hi), 2)],
            "level_ceiling_mgdl": round(float(hi) * 16.0, 1)}


def _ops(path):
    return sorted({o["op_name"] for o in _interpreter(path)._get_ops_details()})


def _artifact_group(art_dir):
    from experiments.accuracy_sweep_r5 import _sha256
    int8 = sorted(art_dir.glob("*int8*.tflite"))
    floats = sorted(art_dir.glob("*float.tflite"))
    inventories = sorted({tuple(_ops(p)) for p in int8})
    return {"int8_artifacts": len(int8), "float_artifacts": len(floats),
            "int8_sizes_bytes": sorted({p.stat().st_size for p in int8}),
            "float_sizes_bytes": sorted({p.stat().st_size for p in floats}),
            "int8_distinct_sha256": len({_sha256(p) for p in int8}),
            "int8_distinct_operator_inventories": len(inventories),
            "int8_operator_inventories": [list(i) for i in inventories],
            "sources": [f"{_rel(art_dir)}/*.tflite"]}


def _calibration_diff(path_a, path_b):
    """Tensors whose quantization parameters or constant data differ between
    two INT8 artifacts of one checkpoint calibrated on different windows."""
    a, b = _interpreter(path_a), _interpreter(path_b)
    ops = a._get_ops_details()
    producer = {t: o["op_name"] for o in ops for t in o["outputs"]}
    # Kernel scratch tensors are neither consumed nor produced; their
    # contents are uninitialized memory, so only graph tensors are compared.
    graph = {t for o in ops for t in o["inputs"] if t >= 0} | set(producer)
    graph_input = a.get_input_details()[0]["index"]
    changed, const_int8_same = [], []
    pairs = [(ta, tb) for ta, tb in zip(a.get_tensor_details(), b.get_tensor_details())
             if ta["index"] in graph]
    for ta, tb in pairs:
        if ta["name"] != tb["name"] or list(ta["shape"]) != list(tb["shape"]):
            raise SystemExit("REFUSED: artifacts do not share one graph")
        i, qa, qb = ta["index"], ta["quantization_parameters"], tb["quantization_parameters"]
        q_diff = not (np.array_equal(qa["scales"], qb["scales"])
                      and np.array_equal(qa["zero_points"], qb["zero_points"]))
        const = i not in producer and i != graph_input
        data_diff = const and not np.array_equal(a.get_tensor(i), b.get_tensor(i))
        dtype = np.dtype(ta["dtype"]).name
        if const and dtype == "int8":
            const_int8_same.append(not (q_diff or data_diff))
        if q_diff or data_diff:
            changed.append({
                "tensor": int(i), "dtype": dtype,
                "role": ("constant" if const else "graph input" if i == graph_input
                         else f"output of {producer[i]}"),
                "constant_data_differs": bool(data_diff), "n_scales": int(len(qa["scales"])),
                "scale": [float(qa["scales"][0]), float(qb["scales"][0])],
                "zero_point": [int(qa["zero_points"][0]), int(qb["zero_points"][0])]})
    return {"artifacts": [_rel(path_a), _rel(path_b)],
            "n_tensors": len(pairs), "n_changed": len(changed),
            "n_changed_outside_layout_ops": sum(  # RESHAPE/TRANSPOSE inherit their input's params
                c["role"] not in ("output of RESHAPE", "output of TRANSPOSE") for c in changed),
            "int8_constants_identical": all(const_int8_same), "n_int8_constants": len(const_int8_same),
            "changed": changed,
            "note": "scale/zero_point list [first artifact, second]; first channel shown for per-channel tensors"}


def _dtype_map(path):
    it = _interpreter(path)
    dt = {t["index"]: np.dtype(t["dtype"]).name for t in it.get_tensor_details()}
    return [{"op": o["op_name"], "inputs": [dt[i] for i in o["inputs"] if i >= 0],
             "outputs": [dt[i] for i in o["outputs"]]} for o in it._get_ops_details()]


def _artifacts():
    run = RESULTS / "r1"
    manifest = _read(run / "artifacts_manifest.json")
    bench = _read(PUBLISHED / "device_benchmark_ios_r1.json")["artifacts"]
    int8 = [k for k in manifest if "int8" in k]
    timed = [k for k in bench if "int8" in k]
    probe = run / "artifacts" / "seed0_int8_uniform_random_200_d101.tflite"
    return {
        "study_r1": {
            "int8_artifacts": len(int8), "float_artifacts": len(manifest) - len(int8),
            "device_timed_int8": len(timed), "device_timed_float": len(bench) - len(timed),
            "untimed_int8": len(int8) - len(timed), "device_timed": sorted(bench),
            "inventory_method_reproduces_manifest": _ops(probe) == manifest[probe.name]["ops"],
            "sources": ["results/calibration-coverage/r1/artifacts_manifest.json",
                        "paper/results/calibration-coverage/device_benchmark_ios_r1.json"]},
        "confirmation_dubosson": _artifact_group(RESULTS / "conf-dubosson" / "artifacts"),
        "confirmation_hall": _artifact_group(RESULTS / "conf-hall" / "artifacts"),
        "delta_r1": _artifact_group(RESULTS / "delta-r1" / "artifacts"),
        "delta_r1_anchor_input_qparams": {
            cell: [_input_qparams(RESULTS / "delta-r1" / "artifacts" / f"anchor_seed{s}_int8_{cell}.tflite")
                   for s in TRAINING_SEEDS]
            for cell in ["sequential_200_d0"] + [f"uniform_random_200_d{d}" for d in DRAWS]},
        "study_int8_dtype_map": {"artifact": _rel(probe), "ops": _dtype_map(probe),
                                 "sources": [_rel(probe)]},
        "calibration_changes_inside_int8": _calibration_diff(
            run / "artifacts" / "seed0_int8_sequential_200_d0.tflite", probe),
    }


def _sweep_settings():
    import inspect

    from training.train import train_model
    main_sel = {s: _read(RESULTS / "r1" / "selections" / f"{s}.json")
                for s in ("sequential_200_d0", "uniform_random_200_d101")}
    pool = _read(PUBLISHED / "data_availability.json")["calibration_pool"]
    rounds = {}
    for r in range(1, 6):
        run = RESULTS / f"sweep-r{r}"
        smoke = run / "smoke.json"
        if smoke.exists():
            batch = {"batch_size": _read(smoke)["frozen_batch"], "device": "mps",
                     "source": _rel(smoke) + " (frozen_batch)"}
        else:
            batch = {"batch_size": inspect.signature(train_model).parameters["batch_size"].default,
                     "device": "cpu",
                     "source": "no smoke record: experiments/accuracy_sweep.py stage_train calls "
                               "train_model without batch_size, so training/train.py's default applies"}
        sels = {s: _read(run / "selections" / f"{s}.json") for s in main_sel}
        rounds[f"r{r}"] = {
            "training": batch,
            "int8_calibration": {
                "selections": sorted(sels), "ship_candidate_calibration": "uniform_random_200_d101",
                "pool_sha256": sorted({v["pool_sha256"] for v in sels.values()}),
                "same_pool_as_main_study": all(v["pool_sha256"] == main_sel[k]["pool_sha256"]
                                               for k, v in sels.items()),
                "same_windows_as_main_study": all(v["indices"] == main_sel[k]["indices"]
                                                  for k, v in sels.items()),
                "sources": [f"{_rel(run)}/selections/{s}.json" for s in sorted(sels)]}}
    return {"rounds": rounds,
            "main_study_pool": {"description": "eligible Weinstock validation windows (primary mask)",
                                "windows": pool["eligible_no_gap_fill"], "participants": pool["participants"],
                                "sources": ["paper/results/calibration-coverage/data_availability.json",
                                            "results/calibration-coverage/r1/selections/*.json"]}}


def _flags(out):
    """Observed values that contradict a statement in the plan, the roadmap
    or a published record (reported, never silently reconciled)."""
    flags = []
    p5 = _read(PUBLISHED / "protocol_sweep_r5.json")["calibration"]
    if all(r["int8_calibration"]["same_pool_as_main_study"] for r in out["sweep_settings"]["rounds"].values()):
        flags.append({"item": "INT8 calibration pool of the sweeps (S2)",
                      "stated": f"protocol_sweep_r5.json calibration: '{p5}'",
                      "observed": "every round's selections carry the main-study pool_sha256: the eligible "
                                  "Weinstock validation pool, not training windows"})
    batches = {k: v["training"]["batch_size"] for k, v in out["sweep_settings"]["rounds"].items()}
    flags.append({"item": "sweep training batch size (S17; plan Task 5 'batch size 512')",
                  "stated": "roadmap S17 '512 vs the stated 64'; plan Task 5 'batch size 512'",
                  "observed": batches})
    diff = out["artifacts"]["calibration_changes_inside_int8"]
    roles = sorted({c["role"] for c in diff["changed"]} - {"output of RESHAPE", "output of TRANSPOSE"})
    flags.append({"item": "what calibration changes inside an INT8 artifact (S21)",
                  "stated": "main.tex Deployment costs: 'changes nothing the device sees except the two "
                            "quantization parameters'",
                  "observed": f"{diff['n_changed']} of {diff['n_tensors']} graph tensors change between two "
                              f"calibrations of one checkpoint; {diff['n_changed_outside_layout_ops']} outside "
                              f"layout ops ({', '.join(roles)}); int8 weights identical: "
                              f"{diff['int8_constants_identical']}"})
    part = out["weinstock"]["facts"]["randomized_200_participants"]
    flags.append({"item": "randomized 200-window participant range (S38)",
                  "stated": "main.tex: 96--127", "observed": [part["min"], part["max"]]})
    return flags


def main():
    import argparse
    import time

    from experiments.accuracy_sweep_r5 import _sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    t0 = time.time()
    rebuilt = rebuild_eval_components()
    weinstock = _weinstock(args.n_boot)
    weinstock["censoring"] = _censoring(args.n_boot, rebuilt)
    weinstock["facts"] = _facts()
    out = {
        "generated_by": "python -m experiments.panel_revision",
        "plan": "docs/superpowers/plans/2026-10-16-paper-panel-revision.md (Task 1)",
        "conventions": {
            "differences": UNITS, "levels": "fractions (recall, shares)",
            "directional_recall": "minority macro recall: mean recall over the four non-stable classes",
            "classes": list(FIVE_CLASSES),
            "bootstrap": ("paired participant-cluster bootstrap semantics of analyze_calibration_coverage: "
                          f"percentile 95%, ANALYSIS_SEED {ANALYSIS_SEED}, interval omitted when more "
                          "than 5% of draws are undefined"),
            "no_new_inference": "every prediction is a cached .npy from a frozen evaluation",
        },
        "weinstock": weinstock,
        "confirmation": {c: _confirmation(c) for c in ("dubosson", "hall")},
        "shipped_model": _shipped_model(rebuilt),
        "round_one_f1": _round_one_f1(),
        "pedap_per_class": _pedap_per_class(),
        "artifacts": _artifacts(),
        "sweep_settings": _sweep_settings(),
    }
    out["confirmation"]["weinstock_comparison"] = {
        comp: {"clipping": _clip_fractions(RESULTS / "r1" / "eval" / comp),
               "sources": [f"results/calibration-coverage/r1/eval/{comp}/seed*_int8_*_200_*_meta.json"]}
        for comp in ("held_out", "temporal")}
    out["flags"] = _flags(out)
    path = PUBLISHED / "panel_revision_r1.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")

    manifest = _read(RESULTS / "r1" / "artifacts_manifest.json")
    local = {n: _sha256(RESULTS / "r1" / "artifacts" / n) for n in manifest}
    hashes = {"generated_by": "python -m experiments.panel_revision",
              "study_r1": {"source": "results/calibration-coverage/r1/artifacts_manifest.json",
                           "verified_against_local_files": all(
                               local[n] == m["sha256"] for n, m in manifest.items()),
                           "sha256": {n: m["sha256"] for n, m in sorted(manifest.items())}}}
    for key, d in (("confirmation_dubosson", "conf-dubosson"), ("confirmation_hall", "conf-hall"),
                   ("delta_r1", "delta-r1")):
        files = sorted((RESULTS / d / "artifacts").glob("*.tflite"))
        hashes[key] = {"source": f"results/calibration-coverage/{d}/artifacts/*.tflite",
                       "sha256": {p.name: _sha256(p) for p in files}}
    (PUBLISHED / "artifact_hashes_r1.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"wrote {path} and artifact_hashes_r1.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
