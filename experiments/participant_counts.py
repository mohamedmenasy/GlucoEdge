"""Per-participant confusion counts for every cached evaluation arm (re-review
item S17), so the published participant-cluster bootstrap intervals can be
recomputed without retraining.

No new inference: counts come from the cached .npy predictions (logits are
argmaxed, as the sweep runners did). Participants are anonymous indices in
the order paired_cluster_bootstrap resamples them. `main()` writes one
gzipped JSON per licence family under paper/results/calibration-coverage/
and fails unless the counts reproduce every listed published interval
exactly.
"""
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.analyze_calibration_coverage import (
    ANALYSIS_SEED, macro_f1, minority_macro_recall, paired_cluster_bootstrap_counts,
    participant_counts)
from experiments.calibration_coverage import DRAWS, RESULTS, TRAINING_SEEDS
from training.labeling import FIVE_CLASSES

PUBLISHED = Path(__file__).resolve().parents[1] / "paper" / "results" / "calibration-coverage"
WEINSTOCK_SPLITS = ("held_out", "temporal")
FAMILIES = {
    "glucobench": {
        "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
        "attribution": (
            "Derived from the Weinstock (CC BY-SA 4.0), Dubosson/D1NAMO (CC BY-SA 4.0) "
            "and Hall (CC BY 4.0) datasets as distributed by GlucoBench (Sergazinov et al., "
            "ICLR 2024; licences as listed in the GlucoBench README), by GlucoEdge, "
            "Mohamed Menasy (https://github.com/mohamedmenasy/GlucoEdge)."),
    },
    "diadata": {
        "license": "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)",
        "attribution": (
            "Derived from the PEDAP and RT-CGM cohorts of DiaData v3 (Cinar and Maleshkova, "
            "Zenodo, doi:10.5281/zenodo.17285631, CC BY-NC 4.0), by GlucoEdge, Mohamed Menasy "
            "(https://github.com/mohamedmenasy/GlucoEdge). Some DiaData cohorts originate from "
            "the Jaeb Center for Health Research and other institutions; the analyses, content "
            "and conclusions are solely the responsibility of the author and have not been "
            "reviewed or approved by these institutions."),
    },
}
FORMAT = (
    "records[name]['arms'][arm] is a list of n_participants rows; row p is participant p's "
    "5x5 confusion matrix flattened row-major (rows = true class, columns = predicted, "
    "class order in 'classes'). Participant indices are anonymous and follow the sorted "
    "stored IDs, the order paired_cluster_bootstrap resamples (numeric for the study and "
    "second-cohort runs, lexicographic string order for the delta and sweep runs), so an "
    "index does not name the same participant across records. Recompute an interval with "
    "experiments.analyze_calibration_coverage.paired_cluster_bootstrap_counts. Subset "
    "analyses (sensitivity tiers, the RT-CGM 40-400 mg/dL check) are not covered.")


def _labels(split_dir, arm):
    """(labels, participant IDs) a split's predictions were scored against."""
    run, comp = split_dir.parents[1], split_dir.name
    if (run / "eval" / "components.npz").exists():
        c = np.load(run / "eval" / "components.npz")
        return c[f"{comp}_labels"], c[f"{comp}_participants"]
    if (split_dir / "meta.npz").exists():
        m = np.load(split_dir / "meta.npz")
        return m["y"], m["pids"]
    for name in ("component_meta.json", "meta.json", f"{arm.split('_')[0]}_meta.json"):
        if (split_dir / name).exists():
            m = json.loads((split_dir / name).read_text())
            return np.array(m["labels"]), np.array(m["participants"])
    raise FileNotFoundError(f"no labels for {split_dir}/{arm}")


def _record(split_dir):
    arms, n_windows, n_participants = {}, set(), set()
    for path in sorted(split_dir.glob("*.npy")):
        pred = np.load(path)
        pred = pred.argmax(axis=1) if pred.ndim == 2 else pred
        y, pids = _labels(split_dir, path.stem)
        counts = participant_counts(y, pids, pred)
        arms[path.stem.removesuffix("_logits").removesuffix("_preds")] = counts
        n_windows.add(len(y))
        n_participants.add(len(counts))
    assert len(n_participants) == 1, f"{split_dir}: arms disagree on participants"
    return {"n_participants": n_participants.pop(), "n_windows": sorted(n_windows),
            "arms": arms}


def _check(records, name, pairs, published, metric=minority_macro_recall, seed=ANALYSIS_SEED):
    arms = records[name]["arms"]
    got = paired_cluster_bootstrap_counts([(arms[a], arms[b]) for a, b in pairs],
                                          metric=metric, n_boot=published["n_boot"], seed=seed)
    assert (got["delta"], got["ci95"]) == (published["delta"], published["ci95"]), (
        name, got, published)
    return {"record": name, "delta": got["delta"], "ci95": got["ci95"]}


def _pub(name):
    return json.loads((PUBLISHED / f"{name}.json").read_text())


def _verify(records):
    """Recompute the published intervals from the counts alone."""
    out = []
    seeds_draws = [(s, d) for s in TRAINING_SEEDS for d in DRAWS]
    agg = _pub("aggregate_r1")
    named = {
        "primary_range200_vs_balanced200": ("participant_balanced_range_coverage_200_d{d}",
                                            "participant_balanced_random_200_d{d}"),
        "range200_vs_uniform200": ("participant_balanced_range_coverage_200_d{d}",
                                   "uniform_random_200_d{d}"),
        "uniform200_vs_sequential200": ("uniform_random_200_d{d}", "sequential_200_d0"),
        "uniform1000_vs_sequential1000": ("uniform_random_1000_d{d}", "sequential_1000_d0"),
    }
    for comp in WEINSTOCK_SPLITS:
        for key, (a, b) in named.items():
            pairs = [(f"seed{s}_int8_{a.format(d=d)}", f"seed{s}_int8_{b.format(d=d)}")
                     for s, d in seeds_draws]
            out.append(_check(records, f"r1/{comp}", pairs, agg["contrasts"][comp][key]) | {"key": key})
        headline = [(f"seed{s}_int8_uniform_random_200_d{d}", f"seed{s}_int8_sequential_200_d0")
                    for s, d in seeds_draws]
        for cohort in ("dubosson", "hall"):
            pub = _pub(f"confirmation_{cohort}")["contrast_uniform200_vs_sequential200"][comp]
            out.append(_check(records, f"conf-{cohort}/{comp}", headline, pub))
        pub = _pub("full_integer_r1")["contrasts"][comp]["uniform200_vs_sequential200"]
        out.append(_check(records, f"full-int-r1/{comp}", headline, pub["full_integer"])
                   | {"key": "uniform200_vs_sequential200"})
        delta = _pub("delta_inputs_r1")["int8_robustness"]["anchor"][comp]
        for key, arm in (("uniform_vs_sequential", "int8_uniform_random_200_d{d}"),
                         ("float_vs_sequential", "float")):
            pairs = [(f"anchor_seed{s}_{arm.format(d=d)}", f"anchor_seed{s}_int8_sequential_200_d0")
                     for s, d in seeds_draws]
            out.append(_check(records, f"delta-r1/{comp}", pairs, delta[key]) | {"key": key})
    sweeps = (("r1", "D", macro_f1, ANALYSIS_SEED), ("r2", "S", minority_macro_recall, 20261012),
              ("r3", "S", minority_macro_recall, 20261013))
    for r, stage, metric, seed in sweeps:
        rec = _pub(f"accuracy_sweep_{r}")
        best = rec["selection_trail"][stage]["best_seed"]
        for comp in WEINSTOCK_SPLITS:
            pub = rec["components"][comp]["gates"]["contrast_int8_vs_champion"]
            pairs = [(f"winner_seed{best}_int8_uniform_random_200_d101", "champion_int8")]
            out.append(_check(records, f"sweep-{r}/{comp}", pairs, pub, metric, seed))
    r5 = _pub("accuracy_sweep_r5")["components"]
    for comp, pub in (("held_out", r5["held_out"]["contrast"]),
                      ("temporal", r5["temporal"]["contrast"]),
                      ("rtcgm", r5["rtcgm_primary"]["gates"])):
        out.append(_check(records, f"sweep-r5/{comp}", [("winner_int8", "champion_int8")],
                          pub["contrast_int8_vs_champion"], seed=20261015))
    return out


def main():
    records = {}
    for split_dir in sorted(RESULTS.glob("*/eval/*/")):
        if any(split_dir.glob("*.npy")):
            records[f"{split_dir.parents[1].name}/{split_dir.name}"] = _record(split_dir)
            print(f"{split_dir.parents[1].name}/{split_dir.name}: "
                  f"{len(records[f'{split_dir.parents[1].name}/{split_dir.name}']['arms'])} arms")
    verified = _verify(records)
    print(f"{len(verified)} published intervals reproduced exactly")
    family = {name: "diadata" if name.endswith(("/pedap", "/rtcgm")) else "glucobench"
              for name in records}
    for fam, terms in FAMILIES.items():
        recs = {n: r | {"arms": {a: c.reshape(len(c), -1).tolist() for a, c in r["arms"].items()}}
                for n, r in records.items() if family[n] == fam}
        doc = {"description": __doc__.split("\n\n")[0].replace("\n", " "), **terms,
               "format": FORMAT, "classes": list(FIVE_CLASSES),
               "verified_published_intervals": [v for v in verified if family[v["record"]] == fam],
               "records": recs}
        path = PUBLISHED / f"participant_counts_{fam}_r1.json.gz"
        # mtime=0 keeps the gzip bytes reproducible across runs.
        with gzip.GzipFile(path, "wb", mtime=0) as fh:
            fh.write(json.dumps(doc, separators=(",", ":")).encode())
        print(f"wrote {path.relative_to(PUBLISHED.parents[2])} ({path.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
