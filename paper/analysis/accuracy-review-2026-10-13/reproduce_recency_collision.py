"""Demonstrate a positional ambiguity in the shipped CNN using synthetic inputs.

Run from the repository root:
    venv/bin/python paper/analysis/accuracy-review-2026-10-13/reproduce_recency_collision.py

No training, real participant data, evaluation labels, or asset changes are used.
This is an architectural diagnostic, not evidence of predictive improvement.
GlucoEdge is not a medical device.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from conversion.common import load_checkpoint  # noqa: E402
from training.deltas import delta_windows  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_diagnostic(path: Path, windows: np.ndarray) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_spec = interpreter.get_input_details()[0]
    output_spec = interpreter.get_output_details()[0]
    outputs = []
    raw_outputs = []
    for window in windows:
        x = window.reshape(1, 1, -1).astype(np.float32)
        if np.issubdtype(input_spec["dtype"], np.integer):
            scale, zero = input_spec["quantization"]
            limits = np.iinfo(input_spec["dtype"])
            x = np.clip(np.round(x / scale) + zero, limits.min, limits.max).astype(input_spec["dtype"])
        interpreter.set_tensor(input_spec["index"], x)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_spec["index"])[0]
        y = raw.astype(np.float32)
        if np.issubdtype(output_spec["dtype"], np.integer):
            scale, zero = output_spec["quantization"]
            y = (y - zero) * scale
        raw_outputs.append(raw.tolist())
        outputs.append(y)
    outputs = np.stack(outputs)
    return {
        "relative_path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "outputs": outputs.tolist(),
        "raw_outputs": raw_outputs,
        "raw_outputs_exactly_equal": raw_outputs[0] == raw_outputs[1],
        "max_absolute_output_difference": float(np.max(np.abs(outputs[0] - outputs[1]))),
    }


def main() -> None:
    checkpoint = ROOT / "results/calibration-coverage/sweep-r2/checkpoints/M_M3_seed1.pt"
    protocol_path = ROOT / "paper/results/calibration-coverage/protocol_sweep_r3.json"
    protocol = json.loads(protocol_path.read_text())
    # Each row contains twelve readings, ending at minute zero. Only the
    # timing of a single 150 -> 160 change differs; the current value is 160.
    raw = np.array([[150] * 5 + [160] * 7, [150] * 7 + [160] * 5], dtype=np.float32)
    encoded = delta_windows(raw, anchor=True)
    model = load_checkpoint(checkpoint, 5).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(encoded).unsqueeze(1)).numpy()

    assets = ROOT / "android/app/src/main/assets"
    artifacts = {}
    for kind in ("float", "int8"):
        result = artifact_diagnostic(assets / f"trend_{kind}.tflite", encoded)
        expected = protocol["champion"][f"{kind}_sha256"]
        result["expected_sha256_from_r3_protocol"] = expected
        result["matches_frozen_champion_sha256"] = result["sha256"] == expected
        assert result["matches_frozen_champion_sha256"], "Bundled artifact differs from frozen r3 champion."
        source_name = ("winner_seed1_float.tflite" if kind == "float"
                       else "winner_seed1_int8_uniform_random_200_d101.tflite")
        source = ROOT / "results/calibration-coverage/sweep-r2/artifacts" / source_name
        result["matches_r2_seed1_artifact_sha256"] = result["sha256"] == sha256(source)
        assert result["matches_r2_seed1_artifact_sha256"], "Bundled artifact differs from r2 seed1."
        artifacts[kind] = result

    c1, c2, length, classes = 96, 144, 12, 5
    convolution_parameters = c1 * 1 * 3 + c1 + c2 * c1 * 3 + c2
    gap_parameters = convolution_parameters + c2 * classes + classes
    flatten_parameters = convolution_parameters + c2 * length * classes + classes
    actual_parameters = sum(p.numel() for p in model.parameters())
    assert gap_parameters == actual_parameters

    report = {
        "kind": "synthetic_architecture_diagnostic",
        "scope": "No training or real-data accuracy evaluation; not a medical device.",
        "checkpoint": {
            "relative_path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "provenance": "android/app/src/main/assets/MODELS.md identifies sweep-r2 M_M3, seed 1, channels (96,144).",
        },
        "source_sha256": {p: sha256(ROOT / p) for p in ("training/model.py", "training/deltas.py")},
        "readings_relative_minutes": list(range(-55, 1, 5)),
        "synthetic_raw_windows_mg_dl": raw.tolist(),
        "encoded_windows": encoded.tolist(),
        "pulse_delta_indices_zero_based": [4, 6],
        "rise_intervals_relative_minutes": [[-35, -30], [-25, -20]],
        "rise_first_observed_minutes_before_anchor": [30, 20],
        "pytorch": {
            "logits": logits.tolist(),
            "logits_exactly_equal": bool(np.array_equal(logits[0], logits[1])),
            "max_absolute_logit_difference": float(np.max(np.abs(logits[0] - logits[1]))),
        },
        "bundled_artifacts": artifacts,
        "architecture": {
            "current": "Two stride-one kernel-3 convolutions, ReLU, global average pool, five-logit linear head.",
            "local_receptive_field_steps": 5,
            "actual_parameter_count": actual_parameters,
            "proposed_flatten_head_parameter_count": flatten_parameters,
            "proposed_additional_parameters": flatten_parameters - actual_parameters,
            "parameter_increase_percent": (flatten_parameters / actual_parameters - 1) * 100,
            "parameter_formula": "conv = 96*1*3+96 + 144*96*3+144; GAP head = 144*5+5; flatten head = 144*12*5+5",
        },
        "interpretation": "The current CNN can assign identical outputs to different internal timing patterns. These pulses are sufficiently separated from boundaries and the appended anchor for pooling to erase their positions.",
        "limitations": [
            "This example does not establish how often the ambiguity matters on real data or whether the correct future label differs.",
            "A recency-preserving head is an untested improvement hypothesis; no accuracy gain is claimed.",
            "The CNN is not fully permutation invariant: local order, boundaries, and proximity to the appended anchor can affect outputs.",
            "This diagnostic concerns the shipped CNN; the larger receptive field of the r3 TCN changes its behavior.",
        ],
    }
    out = Path(__file__).with_name("recency_collision.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out.relative_to(ROOT)),
                      "pytorch_max_abs_difference": report["pytorch"]["max_absolute_logit_difference"],
                      "litert_max_abs_differences": {k: v["max_absolute_output_difference"] for k, v in artifacts.items()},
                      "current_parameters": actual_parameters,
                      "flatten_parameters": flatten_parameters}, indent=2))


if __name__ == "__main__":
    main()
