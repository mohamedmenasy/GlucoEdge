"""Export the Task 5 nine-artifact benchmark manifest for the iOS runner.

The workload is the SYNTHETIC trace only - no GlucoBench data may enter the
repository via this file. For each prespecified artifact (seed 0, draw 101,
plus both sequential budgets) it records the sha256, byte size, and the
expected outputs on every workload window, so the device runner can verify
hash + numerical parity before timing anything. Warm-up and timed index
ranges are frozen here, not chosen on the phone.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from conversion.synthetic_trace import generate_trace

ARTIFACT_DIR = Path("results/calibration-coverage/r1/artifacts")
OUT = Path("ios/BenchmarkArtifacts/benchmark_manifest.json")
WINDOW = 12
ARTIFACTS = ["seed0_float"] + [
    f"seed0_int8_{strat}_{budget}_{draw}"
    for strat in ("sequential", "uniform_random", "participant_balanced_random",
                  "participant_balanced_range_coverage")
    for budget in (200, 1000)
    for draw in (["d0"] if strat == "sequential" else ["d101"])
]


def _run_raw(interpreter, window):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    x = window.reshape(1, 1, WINDOW).astype(np.float32)
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        info = np.iinfo(np.int8)
        x = np.clip(np.round(x / scale) + zp, info.min, info.max).astype(np.int8)
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]


def main() -> None:
    from ai_edge_litert.interpreter import Interpreter

    values = np.array([v for _, v in generate_trace()], dtype=np.float32)
    windows = [values[i:i + WINDOW] for i in range(len(values) - WINDOW + 1)]
    assert len(windows) >= 200, "synthetic workload too short"

    artifacts = []
    for name in ARTIFACTS:
        path = ARTIFACT_DIR / f"{name}.tflite"
        interp = Interpreter(model_path=str(path))
        interp.allocate_tensors()
        out_detail = interp.get_output_details()[0]
        is_int8 = interp.get_input_details()[0]["dtype"] == np.int8
        outputs = [_run_raw(interp, w) for w in windows]
        entry = {
            "name": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "kind": "int8" if is_int8 else "float",
        }
        if is_int8:
            scale, zp = out_detail["quantization"]
            entry["output_quant"] = {"scale": float(scale), "zero_point": int(zp)}
            entry["raw_outputs"] = [[int(v) for v in o] for o in outputs]
        else:
            entry["logits"] = [[float(v) for v in o] for o in outputs]
        artifacts.append(entry)
        print(f"{name}: {entry['bytes']} bytes {entry['sha256'][:12]}")

    manifest = {
        "workload_source": "conversion/synthetic_trace.py (frozen synthetic trace)",
        "windows": [[round(float(x), 1) for x in w] for w in windows],
        # Frozen protocol: per round and artifact, 100 warm-up calls on
        # windows[0:100], then 200 timed calls on windows[77:277] in order.
        "warmup_indices": [0, 100],
        "timed_indices": [len(windows) - 200, len(windows)],
        "rounds": 30,
        "order_seed": 20261010,
        "artifacts": artifacts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(windows)} windows)")


if __name__ == "__main__":
    main()
