"""POST_HOC cached-logit diagnostic; never a selection or deployment protocol.

Run from the repository with its existing environment::

    venv/bin/python paper/analysis/accuracy-review-2026-10-13/inspect_operating_points.py

Both evaluation splits have already been examined. Their labels choose the
reported oracle offsets, so these results cannot establish an improvement,
reopen a failed gate, or justify shipping a model. The fixed 0.01-logit grid
avoids optimizing insignificant floating-point differences between INT8 ties.
No checkpoints, prediction caches, or bundled artifacts are changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CLASSES = ["falling_fast", "falling", "stable", "rising", "rising_fast"]
MODELS = {"r2_cnn_seed1": ("sweep-r2", 1), "r3_tcn_seed2": ("sweep-r3", 2)}
OFFSETS = np.arange(-200, 201, dtype=np.float64) / 100


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def predict(logits, offset):
    adjusted = logits.copy()
    adjusted[:, 2] += offset
    return adjusted.argmax(axis=1)


def metrics(labels, predictions):
    cm = np.bincount(labels * 5 + predictions, minlength=25).reshape(5, 5)
    support = cm.sum(axis=1)
    assert (support > 0).all(), "All five labels must have support."
    recall = np.diag(cm) / support
    f1 = 2 * np.diag(cm) / (support + cm.sum(axis=0))
    return {
        "directional_recall": float(recall[[0, 1, 3, 4]].mean()),
        "stable_recall": float(recall[2]),
        "macro_f1": float(f1.mean()),
        "accuracy": float(np.diag(cm).sum() / cm.sum()),
    }


def error_diagnostics(labels, predictions):
    cm = np.bincount(labels * 5 + predictions, minlength=25).reshape(5, 5)
    distance = np.abs(labels - predictions)
    n_errors = int((distance > 0).sum())
    opposite = ((labels < 2) & (predictions > 2)) | (
        (labels > 2) & (predictions < 2)
    )
    return {
        "class_order": CLASSES,
        "confusion_true_rows_predicted_columns": cm.tolist(),
        "windows": len(labels),
        "errors": n_errors,
        "adjacent_errors": int((distance == 1).sum()),
        "adjacent_fraction_of_errors": float((distance == 1).sum() / n_errors),
        "two_or_more_band_errors": int((distance >= 2).sum()),
        "two_or_more_band_fraction_of_errors": float((distance >= 2).sum() / n_errors),
        "opposite_direction_errors": int(opposite.sum()),
        "opposite_direction_fraction_of_errors": float(opposite.sum() / n_errors),
        "opposite_direction_fraction_of_windows": float(opposite.mean()),
    }


def oracle_at_floor(curve, target):
    eligible = [row for row in curve if row["stable_recall"] >= target - 1e-12]
    assert eligible, "Fixed diagnostic grid does not reach the requested floor."
    # Increasing only the stable logit makes stable recall nondecreasing and
    # directional recall nonincreasing. The first feasible offset is optimal
    # for directional recall within this grid, conditional on these labels.
    return {"target_stable_recall": target, **eligible[0]}


def run(repo):
    source_root = repo / "results" / "calibration-coverage"
    sources = {}
    output = {
        "status": "POST_HOC_ORACLE_DESCRIPTIVE_ONLY",
        "warning": (
            "Evaluation labels select offsets in this diagnostic. These are "
            "optimistic operating-point descriptions, not validation-selected "
            "models, fresh confirmation, or deployment recommendations."
        ),
        "method": {
            "adjustment": "Add a scalar to the stable-class logit; other logits unchanged.",
            "grid": {"minimum": -2.0, "maximum": 2.0, "step": 0.01},
            "tie_rule": "NumPy argmax: lowest class index wins exact ties.",
            "arithmetic": "Cached float32 logits converted to float64 before adding offsets.",
            "matched_floor_note": (
                "INT8 margin ties make exact stable recall matching impossible in "
                "general. Reported stable recall is the first grid point at or "
                "above the requested floor; inspect realized values."
            ),
            "unchanged": "Five ground-truth classes and their label thresholds.",
        },
        "script_sha256": sha256(Path(__file__)),
        "sources_sha256": sources,
        "components": {},
    }
    held_out_offsets = {}
    for component in ["held_out", "temporal"]:
        data = {}
        reference_meta = None
        for name, (run_name, seed) in MODELS.items():
            folder = source_root / run_name / "eval" / component
            meta_path = folder / "meta.json"
            logits_path = folder / f"winner_seed{seed}_int8_uniform_random_200_d101_logits.npy"
            meta = json.loads(meta_path.read_text())
            if reference_meta is None:
                reference_meta = meta
            else:
                for field in ["labels", "participants", "keys"]:
                    assert meta[field] == reference_meta[field], f"Alignment failure: {field}"
            labels = np.asarray(meta["labels"], dtype=np.int64)
            logits = np.load(logits_path).astype(np.float64)
            assert logits.shape == (len(labels), 5)
            assert np.isfinite(logits).all()
            data[name] = (labels, logits)
            for path in [meta_path, logits_path]:
                sources[str(path.relative_to(repo))] = sha256(path)
        champion_y, champion_z = data["r2_cnn_seed1"]
        champion = metrics(champion_y, predict(champion_z, 0.0))
        result = {
            "alignment_asserted": ["labels", "participants", "keys"],
            "windows": len(champion_y),
            "champion_unmodified": champion,
            "champion_error_diagnostics": error_diagnostics(champion_y, predict(champion_z, 0.0)),
            "models": {},
        }
        for name, (labels, logits) in data.items():
            curve = [
                {"offset": float(offset), **metrics(labels, predict(logits, offset))}
                for offset in OFFSETS
            ]
            row = {
                "zero_offset": metrics(labels, predict(logits, 0.0)),
                "oracle_at_champion_stable": oracle_at_floor(curve, champion["stable_recall"]),
                "oracle_at_champion_stable_minus_2pp": oracle_at_floor(curve, champion["stable_recall"] - 0.02),
                "curve": curve,
            }
            selected_keys = ["oracle_at_champion_stable", "oracle_at_champion_stable_minus_2pp"]
            if component == "held_out":
                held_out_offsets[name] = {key: row[key]["offset"] for key in selected_keys}
            else:
                row["held_out_oracle_offsets_applied_to_temporal"] = {
                    key: {"offset": offset, **metrics(labels, predict(logits, offset))}
                    for key, offset in held_out_offsets[name].items()
                }
            result["models"][name] = row
        output["components"][component] = result
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("operating_points.json"))
    args = parser.parse_args()
    result = run(args.repo.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        component: {
            "champion": data["champion_unmodified"],
            "errors": data["champion_error_diagnostics"],
            "models": {
                name: {key: value for key, value in row.items() if key != "curve"}
                for name, row in data["models"].items()
            },
        }
        for component, data in result["components"].items()
    }
    print(json.dumps({"status": result["status"], "summary": summary}, indent=2))
