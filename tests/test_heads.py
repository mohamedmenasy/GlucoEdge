import numpy as np
import torch

from training.heads import (HEADS, corn_loss, decode_outputs, head_outputs,
                            make_r4_criterion, sqrt_class_weights)
from training.labeling import FIVE_CLASSES


class _Tiny:
    labels = [0, 2, 2, 2, 4]  # imbalanced: stable 3x, fasts 1x each


def test_head_widths():
    assert [head_outputs(h) for h in HEADS] == [5, 5, 1, 4]


def test_sqrt_weights_are_sqrt_of_balanced():
    from training.train import class_weights
    w = class_weights(_Tiny(), 5)
    assert torch.allclose(sqrt_class_weights(_Tiny(), 5), w.sqrt())


def test_decode_ce_argmax():
    out = np.array([[0.1, 0.2, 3.0, 0.1, 0.1], [5.0, 0, 0, 0, 0]], dtype=np.float32)
    assert decode_outputs("ce", out).tolist() == [2, 0]


def test_decode_reg_thresholds_match_labeler():
    rates = np.array([[-2.0], [-1.0], [-0.5], [0.99], [1.0], [2.0]], dtype=np.float32)
    idx = decode_outputs("reg", rates)
    names = [FIVE_CLASSES[i] for i in idx]
    assert names == ["falling_fast", "falling", "stable", "stable", "rising", "rising_fast"]


def test_decode_corn_counts_positives_even_non_monotone():
    out = np.array([[-1.0, 2.0, -3.0, 4.0],    # 2 positives -> class 2
                    [1.0, 1.0, 1.0, 1.0],      # 4 -> class 4
                    [-1.0, -1.0, -1.0, -1.0]],  # 0 -> class 0
                   dtype=np.float32)
    assert decode_outputs("corn", out).tolist() == [2, 4, 0]


def test_corn_loss_hand_case():
    # One sample, y=2: outputs 0 and 1 are conditioned on (targets 1,1),
    # output 2 is conditioned on y>=2 (target 0), output 3 is NOT trained
    # (y >= 3 subset empty for this sample).
    logits = torch.zeros(1, 4)  # sigmoid = 0.5 everywhere -> BCE = ln 2 per term
    w = torch.ones(5)
    loss = corn_loss(logits, torch.tensor([2]), w)
    assert torch.isclose(loss, torch.tensor(float(np.log(2.0))), atol=1e-6)


def test_make_r4_criterion_reg_is_huber():
    crit = make_r4_criterion("reg", torch.ones(5))
    out = torch.tensor([[1.5], [0.0]])
    tgt = torch.tensor([1.5, 2.0])
    # |err| = 0 and 2.0 -> huber(delta=1): 0 and (2 - 0.5) = 1.5; mean = 0.75
    assert torch.isclose(crit(out, tgt), torch.tensor(0.75), atol=1e-6)
