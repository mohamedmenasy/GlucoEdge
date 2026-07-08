"""Deterministic synthetic CGM-like trace for the Android app demo.

Entirely generated data - no GlucoBench rows. Piecewise linear glucose
dynamics with small seeded noise, shaped so every trend class occurs and
the values exceed the INT8 model's ~240 mg/dL input ceiling.
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# (duration_minutes, slope mg/dL per minute)
_SEGMENTS = [
    (60, 0.0),    # steady baseline
    (45, 2.6),    # fast meal rise            -> rising_fast
    (30, 1.2),    # slowing rise              -> rising
    (40, 0.0),    # high plateau (> 240)
    (45, -2.6),   # fast correction drop      -> falling_fast
    (40, -1.2),   # slowing drop              -> falling
    (80, 0.0),    # steady
    (50, 2.2),    # second rise
    (60, -0.4),   # gentle drift down         -> stable-ish
    (90, -1.4),   # long fall
    (180, 0.05),  # overnight flat
    (750, 0.0),   # pad to 24h (trimmed below)
]
_START_MGDL = 110.0
_CADENCE_MIN = 5
_POINTS = 288  # 24h


def generate_trace() -> list[tuple[datetime, float]]:
    rng = np.random.default_rng(0)
    t = datetime(2026, 1, 1, 0, 0)
    value = _START_MGDL
    out = [(t, round(value, 1))]
    for duration, slope in _SEGMENTS:
        for _ in range(duration // _CADENCE_MIN):
            if len(out) >= _POINTS:
                break
            value = float(np.clip(value + slope * _CADENCE_MIN + rng.normal(0.0, 0.8), 45.0, 390.0))
            t += timedelta(minutes=_CADENCE_MIN)
            out.append((t, round(value, 1)))
    assert len(out) == _POINTS, f"segment table yields {len(out)} points, expected {_POINTS}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="android/app/src/main/assets/synthetic_trace.csv"
    )
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "gl"])
        for t, v in generate_trace():
            writer.writerow([t.strftime("%Y-%m-%dT%H:%M"), f"{v:.1f}"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
