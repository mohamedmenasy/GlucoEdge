import numpy as np
import torch

from training.augment import mixup_collate, noisy_delta_windows
from training.deltas import delta_windows


def test_noisy_deltas_sigma_zero_is_plain_encoding():
    levels = np.linspace(100, 155, 12, dtype=np.float32)[None]
    rng = np.random.default_rng(0)
    assert np.allclose(noisy_delta_windows(levels, 0.0, rng),
                       delta_windows(levels, True))


def test_noisy_deltas_deterministic_under_seed():
    levels = np.full((4, 12), 120.0, dtype=np.float32)
    a = noisy_delta_windows(levels, 2.0, np.random.default_rng(7))
    b = noisy_delta_windows(levels, 2.0, np.random.default_rng(7))
    c = noisy_delta_windows(levels, 2.0, np.random.default_rng(8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def _items(n):
    return [(torch.full((1, 12), float(i)), float(i)) for i in range(n)]


def test_mixup_convex_combination():
    fn = mixup_collate(alpha=0.4, seed=3)
    x, r = fn(_items(8))
    assert x.shape == (8, 1, 12) and r.shape == (8,)
    # every mixed rate lies between the min and max source rates
    assert float(r.min()) >= 0.0 and float(r.max()) <= 7.0
    # x and rate are mixed with the SAME lambda: x rows are constant-valued
    assert torch.allclose(x[:, 0, 0], r)


def test_mixup_batch_of_one_is_identity():
    fn = mixup_collate(alpha=0.4, seed=0)
    x, r = fn(_items(1))
    assert x.shape == (1, 1, 12) and torch.isclose(r[0], torch.tensor(0.0))
