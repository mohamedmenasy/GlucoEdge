"""Benchmarks the converted LiteRT models against the original PyTorch
checkpoint: file size, inference latency, and accuracy (classification
report) on the same held-out test set for all three representations."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GLUCOEDGE_ROOT))

from conversion.common import load_checkpoint  # noqa: E402

WARMUP_RUNS = 20
TIMED_RUNS = 200


def file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def _quantize_input(window: np.ndarray, input_detail: dict) -> np.ndarray:
    x = window.reshape(1, 1, 12).astype(np.float32)
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return x
    scale, zero_point = input_detail["quantization"]
    return ((x / scale) + zero_point).astype(dtype)


def _dequantize_output(raw_output: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return raw_output
    scale, zero_point = output_detail["quantization"]
    return (raw_output.astype(np.float32) - zero_point) * scale


def benchmark_tflite_latency(tflite_path: Path, sample_window: np.ndarray) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x = _quantize_input(sample_window, input_details[0])

    for _ in range(WARMUP_RUNS):
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

    timings_ms = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": statistics.quantiles(timings_ms, n=100)[94],
    }


def predict_tflite(tflite_path: Path, windows: list) -> list:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    for window in windows:
        x = _quantize_input(window, input_details[0])
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])
        output = _dequantize_output(raw_output, output_details[0])
        preds.append(int(np.argmax(output[0])))
    return preds


def predict_pytorch(model, windows: list) -> list:
    preds = []
    with torch.no_grad():
        for window in windows:
            x = torch.from_numpy(window).float().reshape(1, 1, 12)
            out = model(x)
            preds.append(int(out.argmax(dim=1).item()))
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iglu", "weinstock"], required=True)
    parser.add_argument("--classes", choices=["5", "3"], default="5")
    args = parser.parse_args()

    from sklearn.metrics import classification_report

    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP
    from training.train import load_formatter

    results_dir = GLUCOEDGE_ROOT / "results"
    num_classes = 5 if args.classes == "5" else 3
    checkpoint_path = results_dir / f"{args.dataset}_{args.classes}class_model.pt"
    float_path = results_dir / "model_float.tflite"
    int8_path = results_dir / "model_int8.tflite"

    model = load_checkpoint(checkpoint_path, num_classes)

    classes, collapse_map = (FIVE_CLASSES, None) if args.classes == "5" else (THREE_CLASSES, THREE_CLASS_MAP)
    formatter = load_formatter(args.dataset)
    test_ds = GlucoseTrendDataset(formatter.test_data, classes=classes, collapse_map=collapse_map)
    test_windows = [test_ds.windows[i] for i in range(len(test_ds))]
    test_labels = list(test_ds.labels)
    print(f"test windows: {len(test_windows)}")

    pytorch_preds = predict_pytorch(model, test_windows)
    float_preds = predict_tflite(float_path, test_windows)
    int8_preds = predict_tflite(int8_path, test_windows)

    def report(preds):
        return classification_report(
            test_labels,
            preds,
            target_names=classes,
            labels=list(range(len(classes))),
            output_dict=True,
            zero_division=0,
        )

    output = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "size_kb": {
            "float_tflite": file_size_kb(float_path),
            "int8_tflite": file_size_kb(int8_path),
        },
        "latency_ms": {
            "float_tflite": benchmark_tflite_latency(float_path, test_windows[0]),
            "int8_tflite": benchmark_tflite_latency(int8_path, test_windows[0]),
        },
        "classification_report": {
            "pytorch": report(pytorch_preds),
            "float_tflite": report(float_preds),
            "int8_tflite": report(int8_preds),
        },
    }

    out_path = results_dir / "conversion_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
