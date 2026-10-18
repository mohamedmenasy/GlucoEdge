"""Full-integer reconversion of the study checkpoints (round-three panel
items R1 and R5).

LiteRT Torch 0.9.1 lowers each Conv1d to a float convolution behind
DEQUANTIZE, with the bias added separately, so the study's INT8 artifacts
run only the final fully connected layer in int8. The same weights written
as Conv2d (1x3) kernels on a (B, C, 1, 12) view, with the global average
pool as AvgPool2d (1x12), convert to a graph whose every operator is int8.
This runner reconverts run r1's five checkpoints with r1's 20 selections
(histogram observer; the min-max observer arm is not repeated), scores them
on r1's cached evaluation windows and recomputes r1's contrasts. No
retraining. Post hoc: the r1 labels were read before this run existed.

Stages (resumable, under results/calibration-coverage/full-int-r1/):
    python -m experiments.full_integer convert
    python -m experiments.full_integer evaluate
    python -m experiments.analyze_calibration_coverage --run-id full-int-r1
    python -m experiments.full_integer summarize
"""
import argparse
import copy
import json
import shutil

import numpy as np
import torch
import torch.nn as nn

from experiments.analyze_calibration_coverage import (
    accuracy, confusion_counts, macro_f1, minority_macro_recall)
from experiments.calibration_coverage import (
    BUDGETS, DRAWS, GLUCOEDGE_ROOT, RESULTS, STOCHASTIC, TRAINING_SEEDS,
    _evaluate_artifacts, _load_study_data, _run_dir, _versions, _write_json)

RUN_ID = "full-int-r1"
SOURCE = RESULTS / "r1"
PUBLISHED = GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage"
COMPONENTS = ("held_out", "temporal")
COMMANDS = [
    "python -m experiments.full_integer convert",
    "python -m experiments.full_integer evaluate",
    f"python -m experiments.analyze_calibration_coverage --run-id {RUN_ID}",
    "python -m experiments.full_integer summarize",
]


def _as_conv2d(conv):
    out = nn.Conv2d(conv.in_channels, conv.out_channels, (1, conv.kernel_size[0]),
                    padding=(0, conv.padding[0]))
    with torch.no_grad():
        out.weight.copy_(conv.weight.unsqueeze(2))
        out.bias.copy_(conv.bias)
    return out


class TrendCNN2d(nn.Module):
    """A gap-head TrendCNN re-expressed for an integer graph: the same
    arithmetic and the same (B, 1, 12) float input."""

    def __init__(self, src, length=12):
        super().__init__()
        if src.head != "gap":
            raise ValueError("only the gap head is re-expressed")
        self.conv1 = _as_conv2d(src.conv1)
        self.conv2 = _as_conv2d(src.conv2)
        self.relu = nn.ReLU()
        # AdaptiveAvgPool2d or mean() would lower to a float SUM/MUL.
        self.pool = nn.AvgPool2d((1, length))
        self.fc = copy.deepcopy(src.fc)

    def forward(self, x):
        x = self.relu(self.conv1(x.unsqueeze(2)))
        x = self.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


def _study_model(seed):
    from conversion.common import load_checkpoint
    src = load_checkpoint(SOURCE / "checkpoints" / f"seed{seed}.pt", 5)
    return src, TrendCNN2d(src).eval()


def stage_convert():
    import time

    from conversion.convert import convert_float, convert_int8

    run = _run_dir(RUN_ID)
    if not (run / "selections").exists():  # r1's windows, byte-identical
        shutil.copytree(SOURCE / "selections", run / "selections")
    val_ds = _load_study_data("weinstock")[0]["val"]
    art = run / "artifacts"
    art.mkdir(exist_ok=True)
    sample = torch.zeros(1, 1, 12, dtype=torch.float32)
    for seed in TRAINING_SEEDS:
        model = _study_model(seed)[1]
        fpath = art / f"seed{seed}_float.tflite"
        if not fpath.exists():
            t0 = time.time()
            convert_float(model, sample, fpath)
            _write_json(fpath.with_suffix(".json"), {"export_time_s": round(time.time() - t0, 2)})
        for sel_path in sorted((run / "selections").glob("*.json")):
            ipath = art / f"seed{seed}_int8_{sel_path.stem}.tflite"
            if ipath.exists():
                continue
            sel = json.loads(sel_path.read_text())
            calib = [val_ds[i][0].unsqueeze(0) for i in sel["indices"]]
            timings = convert_int8(model, sample, calib, ipath)
            _write_json(ipath.with_suffix(".json"), {"selection": sel_path.name, **timings})
            print(f"converted {ipath.name}")


def stage_evaluate():
    run = _run_dir(RUN_ID)
    ev = run / "eval"
    ev.mkdir(exist_ok=True)
    for name in ("components.npz", "class_support.json"):
        if not (ev / name).exists():
            shutil.copy2(SOURCE / "eval" / name, ev / name)
    comps = np.load(ev / "components.npz")
    for comp in COMPONENTS:
        base = ev / comp
        base.mkdir(exist_ok=True)
        _evaluate_artifacts(run, base, comps[f"{comp}_windows"], comps[f"{comp}_labels"])


# --- summary ------------------------------------------------------------------


def _cell_names(strategy, budget):
    draws = (0,) if strategy == "sequential" else DRAWS
    return [f"seed{s}_int8_{strategy}_{budget}_d{d}" for s in TRAINING_SEEDS for d in draws]


def _graph(path):
    from experiments.panel_revision import _dtype_map
    ops = _dtype_map(path)
    return {"n_ops": len(ops), "ops": ops,
            "all_int8": all(t in ("int8", "int32") for o in ops for t in o["inputs"] + o["outputs"])}


def _qparams(path):
    from experiments.panel_revision import _interpreter
    it = _interpreter(path)
    (i,), (o,) = it.get_input_details(), it.get_output_details()
    return {"input": [float(i["quantization"][0]), int(i["quantization"][1])],
            "output": [float(o["quantization"][0]), int(o["quantization"][1])]}


def stage_summarize():
    import hashlib
    from importlib.metadata import version

    run = RESULTS / RUN_ID
    agg = json.loads((run / "analysis" / "aggregate.json").read_text())
    agg_r1 = json.loads((PUBLISHED / "aggregate_r1.json").read_text())
    comps = np.load(run / "eval" / "components.npz")
    runs = {"full_integer": run, "hybrid_r1": SOURCE}

    def pred(which, comp, name):
        return np.load(runs[which] / "eval" / comp / f"{name}.npy")

    # Re-expression check on the real evaluation windows.
    logit_diff = {}
    for seed in TRAINING_SEEDS:
        src, model = _study_model(seed)
        with torch.no_grad():
            d = max(float((model(w) - src(w)).abs().max()) for w in (
                torch.from_numpy(comps[f"{c}_windows"]).float().unsqueeze(1) for c in COMPONENTS))
        assert d < 1e-3, f"seed {seed}: Conv2d logits differ by {d}"
        logit_diff[str(seed)] = d

    int8 = sorted(p for p in (run / "artifacts").glob("*int8*.tflite"))
    graphs = {p.stem: _graph(p) for p in int8}
    inventories = sorted({json.dumps(g["ops"]) for g in graphs.values()})
    qp = {p.stem: {RUN_ID: _qparams(p), "r1": _qparams(SOURCE / "artifacts" / p.name)} for p in int8}

    out = {
        "generated_by": COMMANDS,
        "protocol": {
            "study": "full-integer reconversion of the r1 study checkpoints (panel items R1, R5)",
            "disclosure": ("post hoc: converted and scored after the r1 held-out and temporal "
                           "labels were read; checkpoints, the 20 selections, the analysis seed "
                           "and every contrast definition are r1's, unchanged"),
            "re_expression": ("Conv1d(k=3, padding 1) -> Conv2d((1,3), padding (0,1)) on a "
                              "(B,C,1,12) view; AdaptiveAvgPool1d(1) -> AvgPool2d((1,12)); "
                              "input interface (1,1,12) float, as r1"),
            "observer": "histogram (r1 default); the min-max observer arm is not repeated",
            "evaluation": "r1's cached eligible windows (r1/eval/components.npz), no data reload",
            "units": "fractions; keys ending in _pts are recall points (x100)",
        },
        "versions": _versions() | {k: version(k) for k in ("litert-torch", "ai-edge-litert", "torchao")},
        "re_expression_check": {
            "max_abs_logit_diff_torch": logit_diff,
            "float_tflite_argmax_agreement_with_r1_float": {
                c: {str(s): float(np.mean(pred("full_integer", c, f"seed{s}_float")
                                          == pred("hybrid_r1", c, f"seed{s}_float")))
                    for s in TRAINING_SEEDS} for c in COMPONENTS}},
        "graph": {
            "full_integer_int8_inventories": [json.loads(i) for i in inventories],
            "full_integer_int8_all_int8": all(g["all_int8"] for g in graphs.values()),
            "hybrid_r1_int8_example": {"artifact": "seed0_int8_uniform_random_200_d101",
                                       **_graph(SOURCE / "artifacts" / "seed0_int8_uniform_random_200_d101.tflite")},
            "float_n_ops": {w: _graph(runs[w] / "artifacts" / "seed0_float.tflite")["n_ops"]
                            for w in runs},
        },
        "quantization": {
            "input_params_equal_to_r1": sum(v[RUN_ID]["input"] == v["r1"]["input"] for v in qp.values()),
            "n_int8_artifacts": len(qp),
            "cliponly_preds_equal_to_r1": {
                c: sum(np.array_equal(pred("full_integer", c, f"{s}_cliponly"),
                                      pred("hybrid_r1", c, f"{s}_cliponly")) for s in qp)
                for c in COMPONENTS},
            "per_artifact": qp,
        },
        "contrasts": {c: {k: {"full_integer": v, "hybrid_r1": agg_r1["contrasts"][c][k]}
                          for k, v in agg["contrasts"][c].items()} for c in COMPONENTS},
        "mechanism": {c: {cell: {w: {k: m[c][cell][k]["mean"] for k in m[c][cell]}
                                 for w, m in (("full_integer", agg["mechanism"]),
                                              ("hybrid_r1", agg_r1["mechanism"]))}
                          for cell in agg["mechanism"][c] if "_200_" in cell}
                      for c in COMPONENTS},
        "arms": {}, "per_seed": {}, "agreement": {},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((run / "artifacts").glob("*.tflite"))},
        "serialized_bytes": {k: sorted({p.stat().st_size for p in (run / "artifacts").glob(f"*_{k}*.tflite")})
                             for k in ("float", "int8")},
    }

    mmr = lambda y, p: minority_macro_recall(confusion_counts(y, p))
    for c in COMPONENTS:
        y = comps[f"{c}_labels"]
        arms = {"float": {"directional_recall": float(np.mean([
            mmr(y, pred("hybrid_r1", c, f"seed{s}_float")) for s in TRAINING_SEEDS]))}}
        agree = {}
        for strategy in ("sequential",) + STOCHASTIC:
            for budget in BUDGETS:
                names = _cell_names(strategy, budget)
                cell = arms[f"{strategy}_{budget}"] = {}
                for w in runs:
                    cms = [confusion_counts(y, pred(w, c, n)) for n in names]
                    cell[w] = {"directional_recall": float(np.mean([minority_macro_recall(m) for m in cms])),
                               "macro_f1": float(np.mean([macro_f1(m) for m in cms])),
                               "accuracy": float(np.mean([accuracy(m) for m in cms])),
                               "n_artifacts": len(cms)}
                agree[f"{strategy}_{budget}"] = {
                    "full_integer_vs_float": float(np.mean([np.mean(
                        pred("full_integer", c, n) == pred("hybrid_r1", c, n.split("_")[0] + "_float"))
                        for n in names])),
                    "hybrid_vs_float": float(np.mean([np.mean(
                        pred("hybrid_r1", c, n) == pred("hybrid_r1", c, n.split("_")[0] + "_float"))
                        for n in names])),
                    "full_integer_vs_hybrid": float(np.mean([np.mean(
                        pred("full_integer", c, n) == pred("hybrid_r1", c, n)) for n in names]))}
        each = [np.mean(pred("full_integer", c, p.stem) == pred("hybrid_r1", c, p.stem)) for p in int8]
        agree["full_integer_vs_hybrid_range"] = [float(min(each)), float(max(each))]
        out["arms"][c], out["agreement"][c] = arms, agree

        per_seed = {}
        for w in runs:
            eff = {str(s): 100 * (np.mean([mmr(y, pred(w, c, f"seed{s}_int8_uniform_random_200_d{d}"))
                                           for d in DRAWS])
                                  - mmr(y, pred(w, c, f"seed{s}_int8_sequential_200_d0")))
                   for s in TRAINING_SEEDS}
            per_seed[w] = {"uniform200_minus_sequential200_pts": eff,
                           "mean_pts": float(np.mean(list(eff.values()))),
                           "range_pts": [float(min(eff.values())), float(max(eff.values()))],
                           "across_seed_sd_pts": float(np.std(list(eff.values()), ddof=1))}
        out["per_seed"][c] = per_seed

    path = PUBLISHED / "full_integer_r1.json"
    _write_json(path, out)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["convert", "evaluate", "summarize"])
    stage = parser.parse_args().stage
    {"convert": stage_convert, "evaluate": stage_evaluate, "summarize": stage_summarize}[stage]()


if __name__ == "__main__":
    main()
