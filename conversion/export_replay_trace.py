"""Extract one contiguous CGM segment from the local GlucoBench clone into
a replay CSV for the Android app. The output is GITIGNORED - GlucoBench
data is never committed to this repository."""
import argparse
import csv
from pathlib import Path

from training.train import load_formatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weinstock")
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument(
        "--out", default="android/app/src/main/assets/real_trace.csv"
    )
    args = parser.parse_args()

    formatter = load_formatter(args.dataset)
    test = formatter.test_data
    # Longest single (id, id_segment) run with at least --min-points readings.
    groups = sorted(
        test.groupby(["id", "id_segment"]),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    key, seg = groups[0]
    if len(seg) < args.min_points:
        raise SystemExit(f"longest segment has only {len(seg)} points")
    seg = seg.sort_values("time")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "gl"])
        for _, row in seg.iterrows():
            writer.writerow(
                [row["time"].strftime("%Y-%m-%dT%H:%M"), f"{float(row['gl']):.1f}"]
            )
    print(f"wrote {path} ({len(seg)} readings from segment {key})")


if __name__ == "__main__":
    main()
