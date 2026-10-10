"""Calibration-coverage study runner (plan Task 4).

Stages (each resumable, writing under results/calibration-coverage/<run-id>/):
    python -m experiments.calibration_coverage train    --run-id r1
    python -m experiments.calibration_coverage select   --run-id r1
    python -m experiments.calibration_coverage convert  --run-id r1
    python -m experiments.calibration_coverage evaluate --run-id r1

`evaluate` is the locked-evaluation gate: it is the first code that reads
held-out labels. Everything upstream (protocol, samplers, masks, missing-
class rule) was frozen first; see paper/results/calibration-coverage/
protocol.json.

The quantization-control helpers at the top are pure and unit-tested; the
heavy stages lazily import torch/LiteRT/GlucoBench so the fast test suite
never needs them.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
RESULTS = GLUCOEDGE_ROOT / "results" / "calibration-coverage"

TRAINING_SEEDS = (0, 1, 2, 3, 4)
DRAWS = (101, 102, 103)
BUDGETS = (200, 1000)
STOCHASTIC = ("uniform_random", "participant_balanced_random",
              "participant_balanced_range_coverage")

# --- quantization controls (H3): pure, unit-tested ---------------------------


def qdq(x, scale, zero_point, qmin=-128, qmax=127):
    """Quantize-dequantize on the INT8 input grid: round half-to-even like
    all three clients, clip to the representable codes, map back."""
    q = np.clip(np.round(np.asarray(x, dtype=np.float32) / scale) + zero_point, qmin, qmax)
    return (scale * (q - zero_point)).astype(np.float32)


def clip_only(x, scale, zero_point, qmin=-128, qmax=127):
    """Range truncation without re-gridding: separates clipping loss from
    input-resolution loss."""
    lo = scale * (qmin - zero_point)
    hi = scale * (qmax - zero_point)
    return np.clip(np.asarray(x, dtype=np.float32), lo, hi)


def qdq_from_detail(x, input_detail):
    """QDQ using the artifact's own scale/zero point, read from its input
    detail - never hardcoded."""
    scale, zero_point = input_detail["quantization"]
    info = np.iinfo(input_detail["dtype"])
    return qdq(x, np.float32(scale), int(zero_point), info.min, info.max)


# --- shared stage plumbing ----------------------------------------------------


def _run_dir(run_id):
    d = RESULTS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _versions():
    import sklearn
    import torch
    return {"torch": torch.__version__, "numpy": np.__version__,
            "sklearn": sklearn.__version__}


def _eligible(ds, owner, no_gap_fill, split):
    from training.splits import gap_free_window_indices, scored_window_indices
    scored = scored_window_indices(ds, owner, split)
    gap_free = set(gap_free_window_indices(ds, no_gap_fill))
    return [i for i in scored if i in gap_free]


def _load_study_data():
    from experiments.data_availability import load_weinstock_context
    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES

    formatter, owner, _, no_gap_fill = load_weinstock_context()
    datasets = {
        "train": GlucoseTrendDataset(formatter.train_data, classes=FIVE_CLASSES),
        "val": GlucoseTrendDataset(formatter.val_data, classes=FIVE_CLASSES),
        "test": GlucoseTrendDataset(formatter.test_data, classes=FIVE_CLASSES),
    }
    return datasets, owner, no_gap_fill


# --- stage: train --------------------------------------------------------------


def stage_train(run_id, epochs=20):
    """Five fixed-seed TrendCNN checkpoints plus the logistic-regression
    baseline, all on the eligible (scored + no-gap-fill) windows."""
    import torch
    from torch.utils.data import Subset

    from training.train import train_model

    datasets, owner, no_gap_fill = _load_study_data()
    train_idx = _eligible(datasets["train"], owner, no_gap_fill, "train")
    val_idx = _eligible(datasets["val"], owner, no_gap_fill, "val")
    train_sub = Subset(datasets["train"], train_idx)
    val_sub = Subset(datasets["val"], val_idx)
    out = _run_dir(run_id) / "checkpoints"
    out.mkdir(exist_ok=True)

    for seed in TRAINING_SEEDS:
        ckpt = out / f"seed{seed}.pt"
        if ckpt.exists():
            print(f"skip seed {seed}: {ckpt} exists")
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_model(train_sub, val_sub, 5, epochs, torch.device("cpu"))
        torch.save(model.state_dict(), ckpt)
        _write_json(out / f"seed{seed}.json", {
            "command": f"python -m experiments.calibration_coverage train --run-id {run_id}",
            "seed": seed, "epochs": epochs, "device": "cpu",
            "rng": "torch.manual_seed + numpy.random.seed, no CUDA, "
                   "DataLoader shuffle from torch default generator",
            "eligible_train_windows": len(train_idx),
            "eligible_val_windows": len(val_idx),
            "duration_s": round(time.time() - t0, 1),
            "versions": _versions(),
        })
        print(f"seed {seed} done in {time.time() - t0:.0f}s")

    _fit_logistic(run_id, datasets["train"], train_idx)


def _train_matrix(ds, indices):
    return np.stack([ds.windows[i] for i in indices]), np.array([ds.labels[i] for i in indices])


def _fit_logistic(run_id, train_ds, train_idx):
    """Multinomial logistic regression on the same 12 readings:
    training-only standardization, balanced weights, fixed C=1."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = _run_dir(run_id) / "baselines"
    out.mkdir(exist_ok=True)
    if (out / "logistic.joblib").exists():
        print("skip logistic: exists")
        return
    x, y = _train_matrix(train_ds, train_idx)
    t0 = time.time()
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
    clf.fit(x, y)
    joblib.dump(clf, out / "logistic.joblib")
    _write_json(out / "logistic.json", {
        "model": "StandardScaler + LogisticRegression(C=1, class_weight=balanced, max_iter=1000)",
        "converged": int(clf[-1].n_iter_.max()) < 1000,
        "n_iter": int(clf[-1].n_iter_.max()),
        "train_windows": len(y),
        "duration_s": round(time.time() - t0, 1),
        "versions": _versions(),
    })
    print("logistic baseline fitted")


def recent_slope_predict(windows):
    """Recent-slope baseline: (g_t - g_{t-15}) / 15 through the same five
    thresholds, inputs only."""
    from training.labeling import FIVE_CLASSES, label_trend
    idx = {c: i for i, c in enumerate(FIVE_CLASSES)}
    return np.array([idx[label_trend(float(w[-4]), float(w[-1]), 15.0)] for w in windows])


# --- stage: select ---------------------------------------------------------------


def _coverage(pool, selection, ds):
    """Measured coverage of one calibration set (plan Section 3)."""
    chosen = selection["indices"]
    pos = np.searchsorted(pool["window_index"], chosen)
    parts = pool["participant"][pos]
    rows = np.unique(np.concatenate([ds.row_ids[i][:ds.input_length] for i in chosen]))
    mins, maxs = pool["input_min"][pos], pool["input_max"][pos]
    per_part = {str(k): int(v) for k, v in zip(*np.unique(parts, return_counts=True))}
    return {
        "participants": len(per_part),
        "segments": int(len({(pool["participant"][p], pool["segment"][p]) for p in pos})),
        "windows_per_participant": per_part,
        "unique_input_readings": int(len(rows)),
        "input_min": float(mins.min()), "input_max": float(maxs.max()),
        "input_quantiles_min_feature": [float(q) for q in np.percentile(mins, [10, 50, 90])],
        "input_quantiles_max_feature": [float(q) for q in np.percentile(maxs, [10, 50, 90])],
    }


def stage_select(run_id):
    from conversion.calibration import build_pool, select

    datasets, owner, no_gap_fill = _load_study_data()
    val_ds = datasets["val"]
    val_idx = _eligible(val_ds, owner, no_gap_fill, "val")
    pool = build_pool(val_ds, val_idx)
    out = _run_dir(run_id) / "selections"
    out.mkdir(exist_ok=True)

    cells = [("sequential", 0)] + [(s, d) for s in STOCHASTIC for d in DRAWS]
    for strategy, draw in cells:
        for budget in BUDGETS:
            name = f"{strategy}_{budget}_d{draw}.json"
            path = out / name
            if path.exists():
                continue
            t0 = time.time()
            sel = select(pool, strategy, budget, seed=draw)
            sel["draw"] = draw
            sel["sampler_time_s"] = round(time.time() - t0, 3)
            sel["coverage"] = _coverage(pool, sel, val_ds)
            _write_json(path, sel)
    print(f"selections ready: {len(list(out.glob('*.json')))} files, pool={len(val_idx)}")


# --- stage: convert ---------------------------------------------------------------


def stage_convert(run_id):
    import torch

    from conversion.common import load_checkpoint
    from conversion.convert import convert_float, convert_int8

    datasets, _, _ = _load_study_data()
    val_ds = datasets["val"]
    run = _run_dir(run_id)
    sels = sorted((run / "selections").glob("*.json"))
    art = run / "artifacts"
    art.mkdir(exist_ok=True)
    sample = torch.zeros(1, 1, 12, dtype=torch.float32)

    for seed in TRAINING_SEEDS:
        model = load_checkpoint(run / "checkpoints" / f"seed{seed}.pt", 5)
        fpath = art / f"seed{seed}_float.tflite"
        if not fpath.exists():
            t0 = time.time()
            convert_float(model, sample, fpath)
            _write_json(fpath.with_suffix(".json"), {"export_time_s": round(time.time() - t0, 2)})
        for sel_path in sels:
            sel = json.loads(sel_path.read_text())
            ipath = art / f"seed{seed}_int8_{sel_path.stem}.tflite"
            if ipath.exists():
                continue
            calib = [val_ds[i][0].unsqueeze(0) for i in sel["indices"]]
            t0 = time.time()
            convert_int8(model, sample, calib, ipath)
            _write_json(ipath.with_suffix(".json"), {
                "selection": sel_path.name,
                "observer_time_s_incl_export": round(time.time() - t0, 2),
                "export_time_s": round(time.time() - t0, 2),
            })
            print(f"converted {ipath.name}")

        # Observer arm (Amendment 3): min-max activation observer on two
        # prespecified cells, all five seeds, draw 101 / d0 for sequential.
        for cell in ("sequential_200_d0",
                     "participant_balanced_range_coverage_200_d101"):
            sel_path = run / "selections" / f"{cell}.json"
            ipath = art / f"seed{seed}_int8_{cell}_obsminmax.tflite"
            if ipath.exists() or not sel_path.exists():
                continue
            sel = json.loads(sel_path.read_text())
            calib = [val_ds[i][0].unsqueeze(0) for i in sel["indices"]]
            t0 = time.time()
            convert_int8(model, sample, calib, ipath, observer="minmax")
            _write_json(ipath.with_suffix(".json"), {
                "selection": sel_path.name, "observer": "minmax",
                "export_time_s": round(time.time() - t0, 2),
            })
            print(f"converted {ipath.name}")


# --- stage: evaluate (locked gate) -----------------------------------------------


def _predict_tflite_batch(path, windows):
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = np.empty(len(windows), dtype=np.int64)
    is_int8 = inp["dtype"] == np.int8
    scale, zp = inp["quantization"]
    for k, w in enumerate(windows):
        x = w.reshape(1, 1, 12).astype(np.float32)
        if is_int8:
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / scale) + zp, info.min, info.max).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        preds[k] = int(np.argmax(interp.get_tensor(out["index"])))
    detail = {"dtype": inp["dtype"], "quantization": inp["quantization"]}
    return preds, detail


def _clip_stats(windows, labels, scale, zp, dtype_info):
    # Rounded-code clipping, per the plan: a slot clips only when its
    # rounded code falls outside the dtype, not when the real value merely
    # exceeds the representable range (those within half a step round in).
    q_raw = np.round(windows / scale) + zp
    clipped = (q_raw < dtype_info.min) | (q_raw > dtype_info.max)
    by_class = {}
    for c in range(5):
        m = labels == c
        if m.any():
            by_class[str(c)] = {
                "slot_fraction": float(clipped[m].mean()),
                "window_fraction": float(clipped[m].any(axis=1).mean()),
            }
    return {"slot_fraction": float(clipped.mean()),
            "window_fraction": float(clipped.any(axis=1).mean()),
            "by_true_class": by_class}


def stage_evaluate(run_id):
    """LOCKED-EVALUATION GATE: first read of held-out labels."""
    import torch

    from conversion.common import load_checkpoint

    datasets, owner, no_gap_fill = _load_study_data()
    test_ds = datasets["test"]
    run = _run_dir(run_id)
    ev = run / "eval"
    ev.mkdir(exist_ok=True)

    components = {}
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        idx = _eligible(test_ds, owner, no_gap_fill, split)
        components[comp] = {
            "windows": np.stack([test_ds.windows[i] for i in idx]),
            "labels": np.array([test_ds.labels[i] for i in idx]),
            "participants": np.array([test_ds.participants[i] for i in idx]),
        }
    np.savez_compressed(ev / "components.npz", **{
        f"{c}_{k}": v for c, d in components.items() for k, v in d.items()})

    # Reveal aggregate held-out class support (published later via analysis).
    support = {c: {str(k): int(v) for k, v in
                   zip(*np.unique(d["labels"], return_counts=True))}
               for c, d in components.items()}
    _write_json(ev / "class_support.json", support)
    print("class support:", support)

    arts = sorted((run / "artifacts").glob("*.tflite"))
    for comp, d in components.items():
        windows, labels = d["windows"], d["labels"]

        base = ev / comp
        base.mkdir(exist_ok=True)
        # Torch checkpoints.
        for seed in TRAINING_SEEDS:
            p = base / f"torch_seed{seed}.npy"
            if p.exists():
                continue
            model = load_checkpoint(run / "checkpoints" / f"seed{seed}.pt", 5)
            with torch.no_grad():
                logits = model(torch.from_numpy(windows).float().unsqueeze(1))
            np.save(p, logits.argmax(dim=1).numpy())
        # Baselines.
        if not (base / "slope.npy").exists():
            np.save(base / "slope.npy", recent_slope_predict(windows))
        if not (base / "logistic.npy").exists():
            import joblib
            # Safe: loads the model this run's own train stage wrote locally.
            clf = joblib.load(run / "baselines" / "logistic.joblib")
            np.save(base / "logistic.npy", clf.predict(windows))
        # Artifacts + input-grid controls.
        for a in arts:
            p = base / f"{a.stem}.npy"
            if p.exists():
                continue
            t0 = time.time()
            preds, detail = _predict_tflite_batch(a, windows)
            np.save(p, preds)
            meta = {"inference_time_s": round(time.time() - t0, 2),
                    "input_dtype": np.dtype(detail["dtype"]).name,
                    "quantization": [float(detail["quantization"][0]),
                                     int(detail["quantization"][1])]}
            if "int8" in a.stem:
                scale, zp = detail["quantization"]
                info = np.iinfo(detail["dtype"])
                seed = a.stem.split("_")[0]
                fpath = run / "artifacts" / f"{seed}_float.tflite"
                qdq_preds, _ = _predict_tflite_batch(fpath, qdq_from_detail(windows, detail))
                np.save(base / f"{a.stem}_qdqfloat.npy", qdq_preds)
                clip_preds, _ = _predict_tflite_batch(
                    fpath, clip_only(windows, np.float32(scale), int(zp), info.min, info.max))
                np.save(base / f"{a.stem}_cliponly.npy", clip_preds)
                meta["clipping"] = _clip_stats(windows, labels, scale, zp, info)
                meta["real_range_exceedance_window_fraction"] = float(
                    ((windows > scale * (info.max - zp)) | (windows < scale * (info.min - zp)))
                    .any(axis=1).mean())
            _write_json(base / f"{a.stem}_meta.json", meta)
            print(f"{comp}/{a.stem} evaluated in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "select", "convert", "evaluate"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.stage == "train":
        stage_train(args.run_id, args.epochs)
    elif args.stage == "select":
        stage_select(args.run_id)
    elif args.stage == "convert":
        stage_convert(args.run_id)
    elif args.stage == "evaluate":
        stage_evaluate(args.run_id)


if __name__ == "__main__":
    main()
