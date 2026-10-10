"""D12: delta-encoded inputs (spec 2026-10-10, Part B).

Same eligible weinstock windows, splits, and seeds as the main study; the
only change is the input representation (training/deltas.delta_windows).
Post-hoc on previously touched held-out data - disclosed in the protocol
and the paper. Stages are resumable by run id, mirroring
experiments.calibration_coverage.
"""
import json
import time

import numpy as np

from experiments.calibration_coverage import (
    TRAINING_SEEDS, _eligible, _load_study_data, _predict_tflite_batch,
    _run_dir, _write_json)
from training.deltas import delta_windows

VARIANTS = ("plain", "anchor")
INT8_ARMS = ("sequential_200_d0", "uniform_random_200_d101",
             "uniform_random_200_d102", "uniform_random_200_d103")
FLOAT_GATE = 0.482  # five-seed held-out directional mean (spec: 0.487 - 0.5pp)


def variant_length(variant):
    return {"plain": 11, "anchor": 12}[variant]


def _is_anchor(variant):
    return variant == "anchor"


class DeltaView:
    """Torch-style dataset over selected indices of a GlucoseTrendDataset,
    with windows delta-transformed. Exposes .labels for class_weights."""

    def __init__(self, base, indices, anchor):
        import torch
        self._torch = torch
        self.base, self.indices, self.anchor = base, list(indices), anchor
        self.labels = [base.labels[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        x = delta_windows(self.base.windows[i][None, :], self.anchor)[0]
        return self._torch.from_numpy(x).unsqueeze(0), self.labels[k]


def stage_train(run_id, variant, epochs=20):
    import torch

    from training.train import train_model

    datasets, owner, no_gap_fill = _load_study_data()
    train_idx = _eligible(datasets["train"], owner, no_gap_fill, "train")
    val_idx = _eligible(datasets["val"], owner, no_gap_fill, "val")
    anchor = _is_anchor(variant)
    train_v = DeltaView(datasets["train"], train_idx, anchor)
    val_v = DeltaView(datasets["val"], val_idx, anchor)
    out = _run_dir(run_id) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    for seed in TRAINING_SEEDS:
        ckpt = out / f"{variant}_seed{seed}.pt"
        if ckpt.exists():
            print(f"skip {ckpt.name}: exists")
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_model(train_v, val_v, 5, epochs, torch.device("cpu"))
        torch.save(model.state_dict(), ckpt)
        _write_json(out / f"{variant}_seed{seed}.json", {
            "variant": variant, "seed": seed, "epochs": epochs,
            "input_length": variant_length(variant),
            "duration_s": round(time.time() - t0, 1)})
        print(f"{variant} seed {seed} done in {time.time() - t0:.0f}s")


def stage_convert(run_id, variant):
    import torch

    from conversion.common import load_checkpoint
    from conversion.convert import convert_float, convert_int8

    datasets, owner, no_gap_fill = _load_study_data()
    val_ds = datasets["val"]
    run = _run_dir(run_id)
    art = run / "artifacts"
    art.mkdir(exist_ok=True)
    anchor = _is_anchor(variant)
    sample = torch.zeros(1, 1, variant_length(variant), dtype=torch.float32)

    for seed in TRAINING_SEEDS:
        model = load_checkpoint(run / "checkpoints" / f"{variant}_seed{seed}.pt", 5)
        fpath = art / f"{variant}_seed{seed}_float.tflite"
        if not fpath.exists():
            convert_float(model, sample, fpath)
            print(f"converted {fpath.name}")
        for arm in INT8_ARMS:
            ipath = art / f"{variant}_seed{seed}_int8_{arm}.tflite"
            sel_path = run / "selections" / f"{arm}.json"
            if ipath.exists():
                continue
            sel = json.loads(sel_path.read_text())
            ws = np.stack([val_ds.windows[i] for i in sel["indices"]])
            dws = delta_windows(ws, anchor)
            calib = [torch.from_numpy(w[None, None, :]) for w in dws]
            timings = convert_int8(model, sample, calib, ipath)
            _write_json(ipath.with_suffix(".json"), {"selection": arm, **timings})
            print(f"converted {ipath.name}")


def stage_evaluate(run_id, variant):
    import torch

    from conversion.common import load_checkpoint

    datasets, owner, no_gap_fill = _load_study_data()
    test_ds = datasets["test"]
    run = _run_dir(run_id)
    anchor = _is_anchor(variant)
    for comp, split in (("held_out", "held_out"), ("temporal", "test")):
        idx = _eligible(test_ds, owner, no_gap_fill, split)
        windows = delta_windows(np.stack([test_ds.windows[i] for i in idx]), anchor)
        base = run / "eval" / comp
        base.mkdir(parents=True, exist_ok=True)
        meta = base / "component_meta.json"
        if not meta.exists():
            _write_json(meta, {
                "labels": [int(test_ds.labels[i]) for i in idx],
                "participants": [str(test_ds.participants[i]) for i in idx]})
        for seed in TRAINING_SEEDS:
            p = base / f"{variant}_seed{seed}_float_torch.npy"
            if not p.exists():
                model = load_checkpoint(run / "checkpoints" / f"{variant}_seed{seed}.pt", 5)
                with torch.no_grad():
                    logits = model(torch.from_numpy(windows).float().unsqueeze(1))
                np.save(p, logits.argmax(dim=1).numpy())
        for a in sorted((run / "artifacts").glob(f"{variant}_seed*.tflite")):
            p = base / f"{a.stem}.npy"
            if p.exists():
                continue
            preds, detail = _predict_tflite_batch(a, windows)
            np.save(p, preds)
            scale, zp = detail["quantization"]
            _write_json(base / f"{a.stem}_meta.json", {
                "input_scale": float(scale), "input_zero_point": int(zp)})
            print(f"{comp}/{a.stem} evaluated")


def stage_analyze(run_id, n_boot=2000):
    from experiments.analyze_calibration_coverage import (
        _metrics, confusion_counts, minority_macro_recall,
        paired_cluster_bootstrap)

    run = _run_dir(run_id)
    out = {"run_id": run_id, "float_gate": {"threshold": FLOAT_GATE},
           "variants": {}, "int8_robustness": {}}
    for comp in ("held_out", "temporal"):
        base = run / "eval" / comp
        meta = json.loads((base / "component_meta.json").read_text())
        y = np.array(meta["labels"])
        pids = np.array(meta["participants"])
        for variant in VARIANTS:
            v = out["variants"].setdefault(variant, {})
            rows = {}
            for seed in TRAINING_SEEDS:
                for kind in ("float",):
                    p = base / f"{variant}_seed{seed}_{kind}.npy"
                    if p.exists():
                        rows[f"seed{seed}_{kind}"] = _metrics(
                            confusion_counts(y, np.load(p)))
                for arm in INT8_ARMS:
                    p = base / f"{variant}_seed{seed}_int8_{arm}.npy"
                    if p.exists():
                        rows[f"seed{seed}_int8_{arm}"] = _metrics(
                            confusion_counts(y, np.load(p)))
            if rows:
                v[comp] = rows
                floats = [rows[f"seed{s}_float"]["minority_macro_recall"]
                          for s in TRAINING_SEEDS if f"seed{s}_float" in rows]
                if comp == "held_out" and floats:
                    out["float_gate"][variant] = {
                        "five_seed_mean": float(np.mean(floats)),
                        "passes": bool(np.mean(floats) >= FLOAT_GATE)}

        def pairs(variant, name_a, name_b):
            got = []
            for s in TRAINING_SEEDS:
                for arm_a, arm_b in [(a, name_b) for a in name_a]:
                    pa = base / f"{variant}_seed{s}_{arm_a}.npy"
                    pb = base / f"{variant}_seed{s}_{arm_b}.npy"
                    if pa.exists() and pb.exists():
                        got.append((np.load(pa), np.load(pb)))
            return got

        for variant in VARIANTS:
            uni = [f"int8_uniform_random_200_d{d}" for d in (101, 102, 103)]
            u_vs_s = pairs(variant, uni, "int8_sequential_200_d0")
            f_vs_s = pairs(variant, ["float"] * 3, "int8_sequential_200_d0")
            if u_vs_s:
                out["int8_robustness"].setdefault(variant, {})[comp] = {
                    "uniform_vs_sequential": paired_cluster_bootstrap(
                        y, pids, u_vs_s, metric=minority_macro_recall,
                        n_boot=n_boot),
                    "float_vs_sequential": paired_cluster_bootstrap(
                        y, pids, f_vs_s, metric=minority_macro_recall,
                        n_boot=n_boot),
                }
    path = run / "analysis"
    path.mkdir(exist_ok=True)
    _write_json(path / "delta_aggregate.json", out)
    print("wrote", path / "delta_aggregate.json")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "convert", "evaluate", "analyze"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.stage == "analyze":
        stage_analyze(args.run_id)
        return
    if not args.variant:
        parser.error("--variant required for this stage")
    {"train": stage_train, "convert": stage_convert,
     "evaluate": stage_evaluate}[args.stage](args.run_id, args.variant) \
        if args.stage != "train" else stage_train(args.run_id, args.variant, args.epochs)


if __name__ == "__main__":
    main()
