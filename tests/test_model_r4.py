import pytest
import torch

from training.model import TrendCNN


@pytest.mark.parametrize("width", [1, 4])
def test_head_width_forward_shapes(width):
    m = TrendCNN(num_classes=width, channels=(96, 144))
    out = m(torch.zeros(2, 1, 12))
    assert out.shape == (2, width)


@pytest.mark.parametrize("width", [1, 4])
def test_litert_conversion_smoke_new_heads(tmp_path, width):
    pytest.importorskip("litert_torch")
    from conversion.convert import convert_float
    m = TrendCNN(num_classes=width, channels=(96, 144)).eval()
    out = tmp_path / f"w{width}.tflite"
    convert_float(m, torch.zeros(1, 1, 12), out)
    assert out.exists() and out.stat().st_size > 0
