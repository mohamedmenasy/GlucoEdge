"""Data-availability audit for the calibration-coverage study (plan Task 2).

Counts, per split, the windows that survive the provenance rules (scored
anchor + horizon rows, Section 2) and the no-gap-fill mask (Amendment 1),
with exclusion counts. Reads class labels for training and the calibration
pool only - the plan allows inspecting those before model fitting. For the
evaluation components (temporal test and held-out participants) it reports
eligibility counts only; their labels stay unread until Task 4's locked
evaluation.

Needs GlucoBench (cloned + unzipped). From the repo root:
    python -m experiments.data_availability
Writes paper/results/calibration-coverage/data_availability.json
(aggregate counts only).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from training.splits import (gap_free_window_indices, observation_status,
                             row_owner, scored_window_indices)

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = GLUCOEDGE_ROOT / "paper" / "results" / "calibration-coverage" / "data_availability.json"


def split_summary(ds, owner, split, no_gap_fill, classes=None):
    """Eligibility counts for one scored split; class counts only when
    `classes` is given (training and calibration pool)."""
    scored = scored_window_indices(ds, owner, split)
    gap_free = set(gap_free_window_indices(ds, no_gap_fill))
    eligible = [i for i in scored if i in gap_free]
    summary = {
        "windows_in_frame": len(ds),
        "scored": len(scored),
        "excluded_context_or_boundary_target": len(ds) - len(scored),
        "eligible_no_gap_fill": len(eligible),
        "excluded_gap_fill": len(scored) - len(eligible),
        "participants": len({ds.participants[i] for i in eligible}),
    }
    if classes is not None:
        counts = Counter(ds.labels[i] for i in eligible)
        summary["class_counts"] = {c: counts.get(k, 0) for k, c in enumerate(classes)}
    return summary


def load_weinstock_context():
    """Formatter plus the provenance arrays every study stage shares:
    (formatter, owner, on_reading, no_gap_fill)."""
    import pandas as pd

    from training.train import GLUCOBENCH_ROOT, load_formatter

    formatter = load_formatter("weinstock")
    data = formatter.data
    assert data.index.equals(pd.RangeIndex(len(data))), "row ids must be positions"
    owner = row_owner(len(data), formatter.train_idx, formatter.val_idx,
                      formatter.test_idx, formatter.test_idx_ood)

    csv_path = GLUCOBENCH_ROOT / "raw_data" / "weinstock.csv"
    raw = pd.read_csv(csv_path, usecols=["id", "time"])
    raw["time"] = pd.to_datetime(raw["time"]).dt.round("1min")  # the formatter's own rounding
    encoder = formatter.encoders["id"]
    readings = raw[raw["id"].isin(encoder.classes_)].assign(
        id=lambda d: encoder.transform(d["id"]).astype(np.float32))
    on_reading, no_gap_fill = observation_status(data[["id", "time"]], readings)
    return formatter, owner, on_reading, no_gap_fill


def main():
    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES

    formatter, owner, _, no_gap_fill = load_weinstock_context()

    train_ds = GlucoseTrendDataset(formatter.train_data, classes=FIVE_CLASSES)
    val_ds = GlucoseTrendDataset(formatter.val_data, classes=FIVE_CLASSES)
    test_ds = GlucoseTrendDataset(formatter.test_data, classes=FIVE_CLASSES)

    val_eligible = split_summary(val_ds, owner, "val", no_gap_fill, FIVE_CLASSES)
    report = {
        "generated_by": "python -m experiments.data_availability",
        "mask": "no-gap-fill, 7.5-minute threshold (Amendment 1)",
        "training": split_summary(train_ds, owner, "train", no_gap_fill, FIVE_CLASSES),
        "calibration_pool": val_eligible,
        # Evaluation components: counts only, labels unread (locked until Task 4).
        "evaluation_temporal": split_summary(test_ds, owner, "test", no_gap_fill),
        "evaluation_held_out": split_summary(test_ds, owner, "held_out", no_gap_fill),
        "calibration_budgets": {
            str(b): ("feasible" if val_eligible["eligible_no_gap_fill"] >= b else "infeasible")
            for b in (200, 1000)
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
