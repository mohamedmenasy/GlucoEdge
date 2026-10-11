"""Hand-checked cases for the sweep's loss arms (spec Stage A)."""
import pytest
import torch
import torch.nn.functional as F

from training.losses import (EPSILON, GAMMA, LAMBDA, LOSSES, _smeared_targets,
                             make_criterion)

W = torch.tensor([1.0, 2.0, 0.5, 1.0, 1.0])
LOGITS = torch.tensor([[2.0, 0.1, -1.0, 0.0, 0.5],
                       [0.0, 3.0, 0.2, -0.5, 1.0]])
Y = torch.tensor([0, 2])


def test_ce_matches_torch_weighted_ce():
    loss = make_criterion("ce", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(
        F.cross_entropy(LOGITS, Y, weight=W).item())


def test_smeared_targets_rows_sum_to_one_boundary_gets_full_epsilon():
    q = _smeared_targets(5, EPSILON)
    assert torch.allclose(q.sum(dim=1), torch.ones(5))
    assert q[0, 1].item() == pytest.approx(EPSILON)      # boundary: one neighbor
    assert q[2, 1].item() == pytest.approx(EPSILON / 2)  # interior: split
    assert q[2, 3].item() == pytest.approx(EPSILON / 2)
    assert q[4, 3].item() == pytest.approx(EPSILON)


def test_ordinal_smooth_hand_computed():
    q = _smeared_targets(5, EPSILON)
    logp = F.log_softmax(LOGITS, dim=1)
    per = -(q[Y] * logp).sum(dim=1)
    w = W[Y]
    expected = (per * w).sum() / w.sum()
    loss = make_criterion("ordinal_smooth", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(expected.item())


def test_distance_ce_adds_expected_ordinal_distance():
    p = F.softmax(LOGITS, dim=1)
    d = (torch.arange(5.0)[None, :] - Y[:, None].float()).abs()
    expected = (F.cross_entropy(LOGITS, Y, weight=W)
                + LAMBDA * (p * d).sum(dim=1).mean())
    loss = make_criterion("distance_ce", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(expected.item())


def test_focal_hand_computed_and_vanishes_at_confident_correct():
    logp = F.log_softmax(LOGITS, dim=1).gather(1, Y[:, None]).squeeze(1)
    pt = logp.exp()
    w = W[Y]
    expected = (-w * (1 - pt) ** GAMMA * logp).sum() / w.sum()
    loss = make_criterion("focal", W)(LOGITS, Y)
    assert loss.item() == pytest.approx(expected.item())
    sure = torch.tensor([[50.0, 0.0, 0.0, 0.0, 0.0]])
    assert make_criterion("focal", W)(sure, torch.tensor([0])).item() == pytest.approx(0.0, abs=1e-6)


def test_unknown_loss_raises():
    with pytest.raises(ValueError):
        make_criterion("nope", W)
