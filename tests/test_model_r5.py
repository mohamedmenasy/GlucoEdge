import numpy as np
import pytest
import torch

from training.model import TrendCNN

PARAM_CAP = 85_450  # 2x champion (spec section 3)


def _params(m):
    return sum(p.numel() for p in m.parameters())


def test_flat_head_shapes_and_params():
    gap = TrendCNN(num_classes=5, channels=(96, 144))
    flat = TrendCNN(num_classes=5, channels=(96, 144), head="flat")
    x = torch.zeros(2, 1, 12)
    assert gap(x).shape == (2, 5)
    assert flat(x).shape == (2, 5)
    assert _params(gap) == 42_725
    assert _params(flat) == 50_645
    assert _params(flat) <= PARAM_CAP


def test_flat_head_sees_position():
    # same values, different order -> different logits (GAP can't do this)
    torch.manual_seed(0)
    flat = TrendCNN(num_classes=5, channels=(96, 144), head="flat").eval()
    a = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    b = torch.flip(a, dims=[-1])
    with torch.no_grad():
        assert not torch.allclose(flat(a), flat(b))


def test_load_checkpoint_head_aware_and_backward_compat(tmp_path):
    from conversion.common import load_checkpoint
    flat = TrendCNN(num_classes=5, channels=(96, 144), head="flat")
    gap = TrendCNN(num_classes=5, channels=(96, 144))
    meta_flat = {"state_dict": flat.state_dict(), "arch": "cnn",
                 "in_channels": 1, "channels": [96, 144], "head": "flat"}
    meta_gap_no_head = {"state_dict": gap.state_dict(), "arch": "cnn",
                        "in_channels": 1, "channels": [96, 144]}
    pf, pg, praw = tmp_path / "f.pt", tmp_path / "g.pt", tmp_path / "raw.pt"
    torch.save(meta_flat, pf)
    torch.save(meta_gap_no_head, pg)
    torch.save(gap.state_dict(), praw)  # r2-era raw dict
    mf = load_checkpoint(pf, 5)
    assert mf.fc.in_features == 144 * 12
    mg = load_checkpoint(pg, 5)        # absent head -> gap
    assert mg.fc.in_features == 144
    mr = load_checkpoint(praw, 5)      # raw state_dict -> gap
    assert mr.fc.in_features == 144


def test_flat_head_converts_to_litert(tmp_path):
    pytest.importorskip("litert_torch")
    from conversion.convert import convert_float, convert_int8
    model = TrendCNN(num_classes=5, channels=(96, 144), head="flat").eval()
    sample = torch.zeros(1, 1, 12)
    fpath = tmp_path / "flat_float.tflite"
    convert_float(model, sample, fpath)
    assert fpath.exists() and fpath.stat().st_size > 0
    calib = [torch.randn(1, 1, 12) for _ in range(8)]
    ipath = tmp_path / "flat_int8.tflite"
    convert_int8(model, sample, calib, ipath)
    assert ipath.exists() and ipath.stat().st_size > 0


def test_unknown_head_raises():
    with pytest.raises(ValueError, match="unknown head"):
        TrendCNN(num_classes=5, head="flt")
