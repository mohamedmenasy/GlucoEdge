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


def test_wider_channels():
    model = TrendCNN(num_classes=5, channels=(48, 72))
    assert model.conv1.out_channels == 48 and model.conv2.out_channels == 72
    assert model(torch.zeros(2, 1, 24)).shape == (2, 5)  # length-agnostic


def test_load_checkpoint_infers_channels(tmp_path):
    from conversion.common import load_checkpoint
    for channels in ((24, 36), (96, 144)):
        src = TrendCNN(num_classes=5, channels=channels)
        p = tmp_path / f"{channels[0]}.pt"
        torch.save(src.state_dict(), p)
        loaded = load_checkpoint(p, 5)
        assert loaded.conv1.out_channels == channels[0]
        assert loaded.conv2.out_channels == channels[1]
