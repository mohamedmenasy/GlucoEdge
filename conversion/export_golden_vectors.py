"""Export golden parity vectors for the Android instrumented test.

Windows come from the SYNTHETIC trace only - no GlucoBench data may enter
the repository via this file. Outputs are produced by the same corrected
quantize/dequantize path the benchmark uses, so the Android app is held to
exactly the numbers the conversion phase documented."""
import hashlib
import json
from pathlib import Path

import numpy as np

from conversion.benchmark import _dequantize_output, _quantize_input
from conversion.synthetic_trace import generate_trace

ASSETS = Path("android/app/src/main/assets")
OUT = Path("android/app/src/androidTest/assets/golden_vectors.json")
CLASS_NAMES = ["falling_fast", "falling", "stable", "rising", "rising_fast"]
WINDOW = 12
PER_CLASS_CAP = 4
MIN_TOTAL = 16


def _run(interpreter, window: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], _quantize_input(window, inp))
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])


def main() -> None:
    from ai_edge_litert.interpreter import Interpreter

    values = np.array([v for _, v in generate_trace()], dtype=np.float32)
    windows = [values[i : i + WINDOW] for i in range(len(values) - WINDOW + 1)]

    float_ip = Interpreter(model_path=str(ASSETS / "trend_float.tflite"))
    int8_ip = Interpreter(model_path=str(ASSETS / "trend_int8.tflite"))
    float_ip.allocate_tensors()
    int8_ip.allocate_tensors()
    int8_out_detail = int8_ip.get_output_details()[0]

    picked: dict[int, list[dict]] = {i: [] for i in range(5)}
    ceiling_window_included = False
    for w in windows:
        f_raw = _run(float_ip, w)[0]
        f_cls = int(np.argmax(f_raw))
        if len(picked[f_cls]) >= PER_CLASS_CAP and not (w.max() > 240 and not ceiling_window_included):
            continue
        q_raw = _run(int8_ip, w)[0]
        q_deq = _dequantize_output(np.array([q_raw]), int8_out_detail)[0]
        picked[f_cls].append(
            {
                "window": [round(float(x), 1) for x in w],
                "float_logits": [float(x) for x in f_raw],
                "float_class": f_cls,
                "int8_raw_output": [int(x) for x in q_raw],
                "int8_class": int(np.argmax(q_deq)),
            }
        )
        if w.max() > 240:
            ceiling_window_included = True

    vectors = [v for vs in picked.values() for v in vs]
    missing = [CLASS_NAMES[i] for i, vs in picked.items() if not vs]
    assert not missing, f"synthetic trace never float-predicts: {missing} - adjust _SEGMENTS"
    assert len(vectors) >= MIN_TOTAL, f"only {len(vectors)} vectors, need >= {MIN_TOTAL}"
    assert ceiling_window_included, "no window with values > 240 mg/dL made it in"

    sha = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ASSETS / "trend_float.tflite", ASSETS / "trend_int8.tflite"]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"class_names": CLASS_NAMES, "model_sha256": sha, "vectors": vectors},
            indent=1,
        )
    )
    print(f"wrote {OUT} ({len(vectors)} vectors, classes all covered)")


if __name__ == "__main__":
    main()
