"""The Conv2d re-expression behind the full-integer reconversion must keep
TrendCNN's arithmetic exactly."""
import torch

from experiments.full_integer import TrendCNN2d
from training.model import TrendCNN


def test_conv2d_reexpression_matches_trendcnn_logits():
    torch.manual_seed(0)
    src = TrendCNN().eval()
    x = torch.randn(64, 1, 12) * 40 + 150
    with torch.no_grad():
        assert torch.allclose(TrendCNN2d(src).eval()(x), src(x), rtol=1e-5, atol=1e-4)
