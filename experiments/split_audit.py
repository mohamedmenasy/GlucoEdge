"""Split and calibration provenance audit for the weinstock trend windows.

Recomputes which GlucoBench split owns each window's rows, how much of the
5-minute grid is interpolated, and what the shipped INT8 calibration set
covered. It reads row identities and timestamps for every split, but glucose
values only from training and validation (calibration-pool) rows - never
labels, held-out glucose values or model predictions - so it can run before
the calibration-coverage study's locked evaluation.

Needs GlucoBench (cloned + unzipped) and ai-edge-litert (conversion
requirements). From the repo root:
    python -m experiments.split_audit
Writes paper/results/split_audit.json (aggregate counts only).
"""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
INT8_MODEL = GLUCOEDGE_ROOT / "android" / "app" / "src" / "main" / "assets" / "trend_int8.tflite"
OUT_PATH = GLUCOEDGE_ROOT / "paper" / "results" / "split_audit.json"
OWNERS = ("train", "val", "test", "held_out")
# Minute-rounded readings more than 1.5 sensor intervals apart have at least
# one missing reading between them.
MAX_ADJACENT_GAP = np.timedelta64(450, "s")


def row_owner(n_rows, train_idx, val_idx, test_idx, held_out_idx):
    """Split that scores each formatter row. GlucoBench prefixes val/test
    frames with context rows from the preceding split, so a row listed in
    several splits belongs to the earliest one."""
    owner = np.full(n_rows, "", dtype=object)
    for name, idx in (("held_out", held_out_idx), ("test", test_idx), ("val", val_idx), ("train", train_idx)):
        owner[np.asarray(idx, dtype=int)] = name
    return owner


def window_provenance(row_ids, owner):
    """Windows counted by the split that owns their target (last) row, plus
    training copies: windows made only of training rows, so the identical
    window and label also occur in the training set."""
    targets = Counter(owner[[rows[-1] for rows in row_ids]])
    return {
        "windows": len(row_ids),
        "target_owner": {name: targets.get(name, 0) for name in OWNERS},
        "training_copies": sum(bool((owner[rows] == "train").all()) for rows in row_ids),
    }


def observation_status(grid, readings):
    """Per grid row (by position): is it an original reading, and is it free
    of gap filling - an original reading, or between two consecutive readings
    of the same participant at most MAX_ADJACENT_GAP apart? Both frames hold
    `id` and minute-rounded `time`."""
    grid = grid.reset_index(drop=True)
    on_reading = np.zeros(len(grid), dtype=bool)
    no_gap_fill = np.zeros(len(grid), dtype=bool)
    reading_times = {pid: np.sort(g["time"].to_numpy()) for pid, g in readings.groupby("id")}
    for pid, g in grid.groupby("id"):
        times, t = reading_times[pid], g["time"].to_numpy()
        pos = np.searchsorted(times, t)  # first reading at or after t
        nxt = times[np.minimum(pos, len(times) - 1)]
        prv = times[np.maximum(pos - 1, 0)]
        hit = (pos < len(times)) & (nxt == t)
        bracketed = (pos > 0) & (pos < len(times)) & (nxt - prv <= MAX_ADJACENT_GAP)
        on_reading[g.index] = hit
        no_gap_fill[g.index] = hit | bracketed
    return on_reading, no_gap_fill


def main():
    import pandas as pd
    from ai_edge_litert.interpreter import Interpreter

    from conversion.convert import build_calibration_inputs
    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES
    from training.train import GLUCOBENCH_ROOT, load_formatter

    formatter = load_formatter("weinstock")
    data = formatter.data
    assert data.index.equals(pd.RangeIndex(len(data))), "row ids must be positions"
    owner = row_owner(len(data), formatter.train_idx, formatter.val_idx,
                      formatter.test_idx, formatter.test_idx_ood)
    frames = {"train": formatter.train_data, "val": formatter.val_data, "test": formatter.test_data}
    # The datasets compute labels internally; this audit never reads them.
    datasets = {name: GlucoseTrendDataset(frame, classes=FIVE_CLASSES) for name, frame in frames.items()}

    csv_path = GLUCOBENCH_ROOT / "raw_data" / "weinstock.csv"
    raw = pd.read_csv(csv_path, usecols=["id", "time"])
    raw["time"] = pd.to_datetime(raw["time"]).dt.round("1min")  # the formatter's own rounding
    encoder = formatter.encoders["id"]
    readings = raw[raw["id"].isin(encoder.classes_)].assign(
        id=lambda d: encoder.transform(d["id"]).astype(np.float32))
    on_reading, no_gap_fill = observation_status(data[["id", "time"]], readings)

    detail = Interpreter(model_path=str(INT8_MODEL)).get_input_details()[0]
    scale, zero_point = detail["quantization"]
    info, scale = np.iinfo(detail["dtype"]), np.float32(scale)
    int8_low, int8_high = float(scale * (info.min - zero_point)), float(scale * (info.max - zero_point))
    gl = data["gl"].to_numpy()

    windows = {}
    for name, ds in datasets.items():
        endpoints = [np.r_[rows[:ds.input_length], rows[-1]] for rows in ds.row_ids]
        windows[name] = {
            **window_provenance(ds.row_ids, owner),
            "participants": int(data.loc[[rows[-1] for rows in ds.row_ids], "id"].nunique()),
            "inputs_and_endpoint_on_readings": sum(bool(on_reading[e].all()) for e in endpoints),
            "inputs_and_endpoint_no_gap_fill": sum(bool(no_gap_fill[e].all()) for e in endpoints),
        }

    # Development data only: training rows and the validation calibration pool.
    above_int8 = {}
    for name in ("train", "val"):
        ds, frame_rows = datasets[name], frames[name].index
        above_int8[name] = {
            "readings": len(frame_rows),
            "readings_above": int((gl[frame_rows] > int8_high).sum()),
            "windows": len(ds),
            "windows_above": sum(bool((gl[rows[:ds.input_length]] > int8_high).any()) for rows in ds.row_ids),
        }

    val_ds = datasets["val"]
    cal_rows = val_ds.row_ids[:len(build_calibration_inputs(val_ds))]
    cal_inputs = np.unique(np.concatenate([rows[:val_ds.input_length] for rows in cal_rows]))
    cal_times = data.loc[cal_inputs, "time"]
    calibration = {
        "windows": len(cal_rows),
        "participants": int(data.loc[cal_inputs, "id"].nunique()),
        "segments": len(data.loc[cal_inputs, ["id", "id_segment"]].drop_duplicates()),
        "unique_input_readings": len(cal_inputs),
        "span_hours": float((cal_times.max() - cal_times.min()) / pd.Timedelta(hours=1)),
        "training_copies": window_provenance(cal_rows, owner)["training_copies"],
        # The only per-participant statistic published: the INT8 range already reveals it.
        "input_max_mgdl": float(gl[cal_inputs].max()),
    }

    report = {
        "generated_by": "python -m experiments.split_audit",
        "glucobench_commit": subprocess.run(["git", "-C", str(GLUCOBENCH_ROOT), "rev-parse", "HEAD"],
                                            capture_output=True, text=True, check=True).stdout.strip(),
        "weinstock_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "participants": {
            "raw": int(raw["id"].nunique()),
            "after_formatter": int(data["id"].nunique()),
            "held_out": int(data.loc[owner == "held_out", "id"].nunique()),
        },
        "grid_rows": {
            "total": len(data),
            "on_original_reading": int(on_reading.sum()),
            "no_gap_fill": int(no_gap_fill.sum()),
        },
        "windows": windows,
        "int8_input": {"scale": float(scale), "zero_point": int(zero_point),
                       "range_mgdl": [round(int8_low, 3), round(int8_high, 3)]},
        "development_inputs_above_int8_range": above_int8,
        "calibration": calibration,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
