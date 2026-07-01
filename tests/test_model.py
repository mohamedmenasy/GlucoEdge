import torch

from training.model import TrendCNN


def test_forward_pass_output_shape():
    model = TrendCNN(num_classes=5)
    x = torch.randn(4, 1, 12)
    out = model(x)
    assert out.shape == (4, 5)


def test_param_count_five_class_budget():
    model = TrendCNN(num_classes=5)
    count = sum(p.numel() for p in model.parameters())
    assert count == 2909


def test_param_count_three_class_budget():
    model = TrendCNN(num_classes=3)
    count = sum(p.numel() for p in model.parameters())
    assert count == 2835
