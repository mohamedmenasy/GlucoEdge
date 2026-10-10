"""Evaluate the protocol's three frozen sensitivity tiers (plan Amendment 1).

The primary analysis scored only no-gap-fill windows. This stage evaluates the
already-frozen artifacts on the three prespecified disjoint tiers outside or
inside that mask (exact-observed subset; gap-filled inputs with no-gap
endpoints; any gap-filled endpoint). It runs after the primary analysis was
published, but the tiers, artifacts, metrics, and contrasts were all frozen in
protocol.json before any held-out label was read. Descriptive sensitivity
analysis only - no new model, selection, or protocol choice depends on it.
"""
import time

import numpy as np

from experiments.analyze_calibration_coverage import (
    _metrics, confusion_counts, minority_macro_recall, paired_cluster_bootstrap)
from experiments.calibration_coverage import (
    DRAWS, TRAINING_SEEDS, _predict_tflite_batch, _run_dir, _write_json,
    recent_slope_predict)


def _tier_predictions(run, comp, tier, windows):
    """Predictions for every primary-eval model on `windows`, cached under
    eval/<comp>/tier_<tier>/ with the primary stage's file names."""
    import joblib
    import torch

    from conversion.common import load_checkpoint

    base = run / "eval" / comp / f"tier_{tier}"
    base.mkdir(parents=True, exist_ok=True)

    def cached(name, fn):
        p = base / f"{name}.npy"
        if not p.exists():
            np.save(p, fn())
        return np.load(p)

    # Safe: joblib loads the model this run's own train stage wrote locally.
    out = {"slope": cached("slope", lambda: recent_slope_predict(windows)),
           "logistic": cached("logistic", lambda: joblib.load(
               run / "baselines" / "logistic.joblib").predict(windows))}
    for seed in TRAINING_SEEDS:
        def torch_preds(seed=seed):
            model = load_checkpoint(run / "checkpoints" / f"seed{seed}.pt", 5)
            with torch.no_grad():
                logits = model(torch.from_numpy(windows).float().unsqueeze(1))
            return logits.argmax(dim=1).numpy()
        out[f"torch_seed{seed}"] = cached(f"torch_seed{seed}", torch_preds)
    for a in sorted((run / "artifacts").glob("*.tflite")):
        out[a.stem] = cached(a.stem, lambda a=a: _predict_tflite_batch(a, windows)[0])
    return out


def main():
    import argparse

    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES
    from training.splits import sensitivity_tier_indices

    from experiments.data_availability import load_weinstock_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    run = _run_dir(args.run_id)

    formatter, owner, on_reading, no_gap_fill = load_weinstock_context()
    test_ds = GlucoseTrendDataset(formatter.test_data, classes=FIVE_CLASSES)

    out = {"run_id": args.run_id, "n_boot": args.n_boot,
           "note": ("Prespecified in protocol.json (observation_mask.sensitivity); "
                    "evaluated 2026-10-10 after the primary analysis was published. "
                    "Same frozen artifacts, metrics, and bootstrap as the primary; "
                    "labels on the gap_filled_endpoint tier derive from gap-filled "
                    "grid values and describe the formatter's grid, not sensor truth."),
           "tiers": {}}
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        tiers = sensitivity_tier_indices(test_ds, owner, split, on_reading, no_gap_fill)
        out["tiers"][comp] = {}
        for tier, idx in tiers.items():
            t0 = time.time()
            entry = {"n_windows": len(idx)}
            if not idx:
                out["tiers"][comp][tier] = entry
                continue
            y = np.array([test_ds.labels[i] for i in idx])
            pids = np.array([test_ds.participants[i] for i in idx])
            windows = np.stack([test_ds.windows[i] for i in idx])
            preds = _tier_predictions(run, comp, tier, windows)
            entry["models"] = {name: _metrics(confusion_counts(y, p))
                               for name, p in preds.items()}

            def pairs(name_a, name_b):
                return [(preds[name_a.format(seed=s, draw=d)],
                         preds[name_b.format(seed=s, draw=d)])
                        for s in TRAINING_SEEDS for d in DRAWS]

            entry["contrasts"] = {
                "uniform200_vs_sequential200": paired_cluster_bootstrap(
                    y, pids, pairs("seed{seed}_int8_uniform_random_200_d{draw}",
                                   "seed{seed}_int8_sequential_200_d0"),
                    metric=minority_macro_recall, n_boot=args.n_boot),
                "float_vs_int8_uniform200": paired_cluster_bootstrap(
                    y, pids, pairs("seed{seed}_float",
                                   "seed{seed}_int8_uniform_random_200_d{draw}"),
                    metric=minority_macro_recall, n_boot=args.n_boot),
            }
            out["tiers"][comp][tier] = entry
            print(f"{comp}/{tier}: n={len(idx)} in {time.time() - t0:.1f}s")

    path = run / "analysis" / "sensitivity_tiers.json"
    _write_json(path, out)
    print("wrote", path)


if __name__ == "__main__":
    main()
