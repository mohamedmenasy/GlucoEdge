"""Staged accuracy sweep (spec 2026-10-11). Run id sweep-r1.

Greedy stages A (loss) -> B (capacity x window) -> C (pooling) ->
D (ensemble), every selection on eligible weinstock val macro-F1 only.
held_out/temporal are touched by stage_evaluate alone, which refuses to
run until the sweep protocol file is committed and clean. All stages are
resumable: existing checkpoints/selections are skipped."""
import json
import subprocess
import time

import numpy as np

from experiments.calibration_coverage import (GLUCOEDGE_ROOT, _eligible,
                                              _run_dir, _write_json)
from training.deltas import delta_windows
from training.losses import LOSSES

ARM_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
POOL_COHORTS = ("weinstock", "dubosson", "hall")
CHANNEL_GRID = ((24, 36), (48, 72), (96, 144))
TIE_BAND = 0.001        # 0.1pp tie rule
ENSEMBLE_MARGIN = 0.005  # Stage D carry gate
EPOCHS = 20
PROTOCOL = GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage" / "protocol_sweep_r1.json"
CHAMPION = {  # bundled delta-anchor pair (D12 ship)
    "float": GLUCOEDGE_ROOT / "android/app/src/main/assets/trend_float.tflite",
    "int8": GLUCOEDGE_ROOT / "android/app/src/main/assets/trend_int8.tflite",
    "length": 12,
}


# --- arms -------------------------------------------------------------------


def arm(name, loss, channels, length, pooled=False):
    return {"name": name, "loss": loss, "channels": list(channels),
            "length": length, "pooled": pooled}


def stage_a_arms():
    return [arm(f"A_{l}", l, (24, 36), 12) for l in LOSSES]


def stage_b_arms(winner_loss, include_36):
    lengths = [12, 24] + ([36] if include_36 else [])
    return [arm(f"B_c{c1}x{c2}_L{n}", winner_loss, (c1, c2), n)
            for c1, c2 in CHANNEL_GRID for n in lengths
            if (c1, c2, n) != (24, 36, 12)]  # that cell is Stage A's winner, reused


def param_count(channels, num_classes=5):
    c1, c2 = channels
    return (3 * c1 + c1) + (3 * c1 * c2 + c2) + (c2 * num_classes + num_classes)


def cost_key(a):
    return (param_count(a["channels"]), a["length"], a["pooled"],
            LOSSES.index(a["loss"]), a.get("ensemble", False))


def pick_winner(arms, means):
    best = max(means[a["name"]] for a in arms)
    candidates = [a for a in arms if means[a["name"]] >= best - TIE_BAND]
    return min(candidates, key=cost_key)


# --- data views ---------------------------------------------------------------


class ConcatView:
    """Concatenation of DeltaView-like datasets; exposes .labels for
    class_weights."""

    def __init__(self, views):
        self.views = views
        self.labels = [l for v in views for l in v.labels]
        self._index = [(vi, k) for vi, v in enumerate(views)
                       for k in range(len(v))]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, k):
        vi, j = self._index[k]
        return self.views[vi][j]


_DATA_CACHE = {}


def _data(dataset, length):
    """(datasets dict, owner, no_gap_fill) for one cohort at one window
    length; cached, GlucoBench loaded lazily."""
    key = (dataset, length)
    if key not in _DATA_CACHE:
        from experiments.data_availability import load_cohort_context
        from training.dataset import GlucoseTrendDataset
        from training.labeling import FIVE_CLASSES
        formatter, owner, _, no_gap_fill = load_cohort_context(dataset)
        ds = {s: GlucoseTrendDataset(getattr(formatter, f"{s}_data"),
                                     classes=FIVE_CLASSES, input_length=length)
              for s in ("train", "val", "test")}
        _DATA_CACHE[key] = (ds, owner, no_gap_fill)
    return _DATA_CACHE[key]


def _train_views(a):
    from experiments.delta_inputs import DeltaView
    cohorts = POOL_COHORTS if a["pooled"] else ("weinstock",)
    views = []
    for c in cohorts:
        ds, owner, ngf = _data(c, a["length"])
        views.append(DeltaView(ds["train"], _eligible(ds["train"], owner, ngf, "train"), True))
    train_v = views[0] if len(views) == 1 else ConcatView(views)
    wds, wowner, wngf = _data("weinstock", a["length"])
    val_v = DeltaView(wds["val"], _eligible(wds["val"], wowner, wngf, "val"), True)
    return train_v, val_v


# --- pairing (used by analyze; pure) -----------------------------------------


def paired_rows(keys_a, keys_b):
    """Row indices pairing two window sets on (participant, anchor_time)
    keys, in keys_a order; windows only one side can score are dropped."""
    pos = {k: i for i, k in enumerate(keys_b)}
    ia = [i for i, k in enumerate(keys_a) if k in pos]
    ib = [pos[keys_a[i]] for i in ia]
    return ia, ib


# --- stage: precheck ----------------------------------------------------------


def stage_precheck(run_id):
    counts = {}
    for length in (12, 24, 36):
        ds, owner, ngf = _data("weinstock", length)
        counts[str(length)] = {s: len(_eligible(ds[s], owner, ngf, s))
                               for s in ("train", "val")}
    include_36 = counts["36"]["train"] >= 0.8 * counts["12"]["train"]
    out = {"counts": counts, "include_36": include_36,
           "rule": "length 36 joins Stage B iff its eligible train count is >= 80% of length 12's"}
    _write_json(_run_dir(run_id) / "precheck.json", out)
    print(json.dumps(out, indent=1))


# --- stage: train ---------------------------------------------------------------


def _read_selection(run, stage):
    return json.loads((run / "selection" / f"{stage}.json").read_text())


def _arms_for(run, stage):
    if stage == "A":
        return stage_a_arms(), ARM_SEEDS
    if stage == "B":
        winner = _read_selection(run, "A")["winner"]
        pre = json.loads((run / "precheck.json").read_text())
        return stage_b_arms(winner["loss"], pre["include_36"]), ARM_SEEDS
    if stage == "C":
        winner = _read_selection(run, "B")["winner"]
        return [dict(winner, name="C_pooled", pooled=True)], ARM_SEEDS
    if stage == "D":
        winner = _read_selection(run, "C")["winner"]
        return [winner], tuple(s for s in FINAL_SEEDS if s not in ARM_SEEDS)
    raise ValueError(stage)


def stage_train(run_id, stage):
    import torch

    from training.losses import make_criterion
    from training.model import TrendCNN
    from training.train import class_weights, train_model

    run = _run_dir(run_id)
    arms, seeds = _arms_for(run, stage)
    out = run / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    for a in arms:
        train_v, val_v = _train_views(a)
        for seed in seeds:
            ckpt = out / f"{a['name']}_seed{seed}.pt"
            if ckpt.exists():
                print(f"skip {ckpt.name}: exists")
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = TrendCNN(num_classes=5, channels=tuple(a["channels"]))
            criterion = make_criterion(a["loss"], class_weights(train_v, 5))
            model = train_model(train_v, val_v, 5, EPOCHS, torch.device("cpu"),
                                model=model, criterion=criterion)
            torch.save(model.state_dict(), ckpt)
            _write_json(out / f"{a['name']}_seed{seed}.json", {
                "arm": a, "seed": seed, "epochs": EPOCHS,
                "train_windows": len(train_v), "val_windows": len(val_v),
                "duration_s": round(time.time() - t0, 1)})
            print(f"{a['name']} seed {seed} done in {time.time() - t0:.0f}s")


# --- stage: select ---------------------------------------------------------------


def _val_logits(run, a, seed):
    import torch

    from conversion.common import load_checkpoint

    wds, owner, ngf = _data("weinstock", a["length"])
    idx = _eligible(wds["val"], owner, ngf, "val")
    windows = delta_windows(np.stack([wds["val"].windows[i] for i in idx]), True)
    model = load_checkpoint(run / "checkpoints" / f"{a['name']}_seed{seed}.pt", 5)
    with torch.no_grad():
        logits = model(torch.from_numpy(windows).float().unsqueeze(1)).numpy()
    y = np.array([wds["val"].labels[i] for i in idx])
    return logits, y


def _val_f1(run, a, seed):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1
    logits, y = _val_logits(run, a, seed)
    return macro_f1(confusion_counts(y, logits.argmax(axis=1)))


def stage_select(run_id, stage):
    from experiments.analyze_calibration_coverage import confusion_counts, macro_f1

    run = _run_dir(run_id)
    (run / "selection").mkdir(exist_ok=True)
    arms, _ = _arms_for(run, stage)
    if stage == "B":  # Stage A's winner competes as the reused baseline cell
        arms = arms + [_read_selection(run, "A")["winner"]]
    if stage == "C":
        arms = arms + [_read_selection(run, "B")["winner"]]

    if stage in ("A", "B", "C"):
        table, means = {}, {}
        for a in arms:
            per_seed = {str(s): _val_f1(run, a, s) for s in ARM_SEEDS}
            means[a["name"]] = float(np.mean(list(per_seed.values())))
            table[a["name"]] = {"per_seed": per_seed, "mean": means[a["name"]]}
        winner = pick_winner(arms, means)
        out = {"stage": stage, "table": table, "winner": winner}
    else:  # D: ensemble vs best single seed, all on val
        a = arms[0]
        per_seed, logit_stack, y = {}, [], None
        for s in FINAL_SEEDS:
            logits, y = _val_logits(run, a, s)
            logit_stack.append(logits)
            per_seed[str(s)] = macro_f1(confusion_counts(y, logits.argmax(axis=1)))
        best_seed = max(per_seed, key=per_seed.get)
        ens_preds = np.mean(logit_stack, axis=0).argmax(axis=1)
        ens_f1 = macro_f1(confusion_counts(y, ens_preds))
        carry = ens_f1 - per_seed[best_seed] >= ENSEMBLE_MARGIN
        out = {"stage": "D", "table": {"per_seed": per_seed},
               "ensemble_val_f1": ens_f1, "best_seed_val_f1": per_seed[best_seed],
               "best_seed": int(best_seed), "carry_ensemble": bool(carry),
               "winner": dict(a, ensemble=bool(carry))}
    _write_json(run / "selection" / f"{stage}.json", out)
    print(json.dumps({k: out[k] for k in out if k != "table"}, indent=1))
