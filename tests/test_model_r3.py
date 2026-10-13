"""r3 model additions - no CGM data needed."""
import pytest
import torch

from training.model import TrendCNN, TrendTCN


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_trendcnn_in_channels_default_unchanged():
    m = TrendCNN(num_classes=5, channels=(96, 144))
    assert m(torch.zeros(2, 1, 12)).shape == (2, 5)


def test_trendcnn_three_channel_input():
    m = TrendCNN(num_classes=5, channels=(96, 144), in_channels=3)
    assert m(torch.zeros(2, 3, 12)).shape == (2, 5)


def test_tcn_shape_params_and_receptive_field():
    tcn = TrendTCN(num_classes=5)
    assert tcn(torch.zeros(2, 1, 12)).shape == (2, 5)
    cnn = TrendCNN(num_classes=5, channels=(96, 144))
    assert _n(tcn) <= 2 * _n(cnn)          # spec: within 2x of champion arch
    # receptive field 1 + 2*(1+2+4+8) = 31 > 12: a change at t=0 reaches the output
    a, b = torch.zeros(1, 1, 12), torch.zeros(1, 1, 12)
    b[0, 0, 0] = 5.0
    tcn.eval()
    assert not torch.allclose(tcn(a), tcn(b))


def test_tcn_three_channels():
    m = TrendTCN(num_classes=5, in_channels=3)
    assert m(torch.zeros(2, 3, 12)).shape == (2, 5)


def test_load_checkpoint_legacy_and_meta(tmp_path):
    from conversion.common import load_checkpoint
    legacy = TrendCNN(num_classes=5, channels=(96, 144))
    p1 = tmp_path / "legacy.pt"
    torch.save(legacy.state_dict(), p1)
    assert isinstance(load_checkpoint(p1, 5), TrendCNN)
    tcn = TrendTCN(num_classes=5, in_channels=3, channels=64)
    p2 = tmp_path / "meta.pt"
    torch.save({"state_dict": tcn.state_dict(), "arch": "tcn",
                "in_channels": 3, "channels": 64}, p2)
    loaded = load_checkpoint(p2, 5)
    assert isinstance(loaded, TrendTCN)
    assert loaded(torch.zeros(1, 3, 12)).shape == (1, 5)


def test_tcn_converts_to_litert(tmp_path):
    # CI's pytest job installs only requirements.txt; conversion pins are local
    pytest.importorskip("litert_torch")
    from conversion.convert import convert_float
    tcn = TrendTCN(num_classes=5)
    tcn.eval()
    out = tmp_path / "t.tflite"
    convert_float(tcn, torch.zeros(1, 1, 12), out)
    assert out.exists() and out.stat().st_size > 1000
