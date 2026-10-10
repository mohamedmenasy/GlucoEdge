"""Converts a trained TrendCNN checkpoint to LiteRT: a float .tflite and a
static-INT8-quantized .tflite, calibrated on real held-out CGM windows."""
import argparse
import sys
from pathlib import Path

import torch

GLUCOEDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GLUCOEDGE_ROOT))

from conversion.common import load_checkpoint  # noqa: E402


def convert_float(model, sample_input: torch.Tensor, out_path: Path) -> None:
    import litert_torch

    edge_model = litert_torch.convert(model, (sample_input,))
    edge_model.export(str(out_path))


def convert_int8(model, sample_input, calibration_inputs, out_path: Path) -> None:
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    exported = torch.export.export(model, (sample_input,)).module()

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    prepared = prepare_pt2e(exported, quantizer)

    for calib_input in calibration_inputs:
        prepared(calib_input)

    quantized = convert_pt2e(prepared, fold_quantize=False)

    edge_model = litert_torch.convert(
        quantized, (sample_input,), quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    edge_model.export(str(out_path))


def build_calibration_inputs(val_ds, max_samples: int = 200) -> list:
    inputs = []
    for i in range(min(max_samples, len(val_ds))):
        x, _ = val_ds[i]
        inputs.append(x.unsqueeze(0))
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iglu", "weinstock"], required=True)
    parser.add_argument("--classes", choices=["5", "3"], default="5")
    # Explicit paths so calibration-coverage matrix runs cannot overwrite
    # one another (or the historical artifacts). Defaults preserve the
    # original CLI behavior exactly.
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--float-out", type=Path, default=None)
    parser.add_argument("--int8-out", type=Path, default=None)
    parser.add_argument("--calibration-selection", type=Path, default=None,
                        help="JSON from conversion.calibration.select; its window "
                             "indices address the validation dataset. Default: the "
                             "historical first-200 sequential windows.")
    args = parser.parse_args()

    import hashlib
    import json

    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES, THREE_CLASSES, THREE_CLASS_MAP
    from training.train import load_formatter

    results_dir = GLUCOEDGE_ROOT / "results"
    num_classes = 5 if args.classes == "5" else 3
    checkpoint_path = args.checkpoint or results_dir / f"{args.dataset}_{args.classes}class_model.pt"

    model = load_checkpoint(checkpoint_path, num_classes)
    sample_input = torch.zeros(1, 1, 12, dtype=torch.float32)

    float_path = args.float_out or results_dir / "model_float.tflite"
    float_path.parent.mkdir(parents=True, exist_ok=True)
    convert_float(model, sample_input, float_path)
    print(f"wrote {float_path}")

    classes, collapse_map = (FIVE_CLASSES, None) if args.classes == "5" else (THREE_CLASSES, THREE_CLASS_MAP)
    formatter = load_formatter(args.dataset)
    val_ds = GlucoseTrendDataset(formatter.val_data, classes=classes, collapse_map=collapse_map)
    selection = None
    if args.calibration_selection:
        selection = json.loads(args.calibration_selection.read_text())
        calibration_inputs = [val_ds[i][0].unsqueeze(0) for i in selection["indices"]]
    else:
        calibration_inputs = build_calibration_inputs(val_ds)
    print(f"calibration windows: {len(calibration_inputs)}")

    int8_path = args.int8_out or results_dir / "model_int8.tflite"
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    convert_int8(model, sample_input, calibration_inputs, int8_path)
    print(f"wrote {int8_path}")

    if selection is not None:
        meta = {
            "checkpoint": checkpoint_path.name,
            "selection": selection,
            "float_sha256": hashlib.sha256(float_path.read_bytes()).hexdigest(),
            "int8_sha256": hashlib.sha256(int8_path.read_bytes()).hexdigest(),
        }
        meta_path = int8_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
