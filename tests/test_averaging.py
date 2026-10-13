import pytest
import torch

from training.averaging import average_state_dicts


def _sd(v):
    return {"w": torch.full((2, 2), float(v)), "b": torch.tensor([float(v)])}


def test_average_is_mean():
    out = average_state_dicts([_sd(0), _sd(2), _sd(4)])
    assert torch.allclose(out["w"], torch.full((2, 2), 2.0))
    assert torch.allclose(out["b"], torch.tensor([2.0]))


def test_average_of_identical_is_identity():
    out = average_state_dicts([_sd(3), _sd(3)])
    assert torch.allclose(out["w"], _sd(3)["w"])


def test_mismatched_keys_refused():
    bad = {"w": torch.zeros(2, 2)}  # missing "b"
    with pytest.raises(ValueError):
        average_state_dicts([_sd(1), bad])
    with pytest.raises(ValueError):
        average_state_dicts([])
