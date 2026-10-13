"""r4 robustness-arm transforms (spec section 3, Stage R). Train-time only;
eval inputs are never touched."""
import numpy as np
import torch

from training.deltas import delta_windows


def noisy_delta_windows(levels, sigma, rng):
    """Delta-anchor encode levels + N(0, sigma). sigma in mg/dL; fresh draw
    per call (the caller re-encodes each epoch)."""
    levels = np.asarray(levels, dtype=np.float32)
    if sigma > 0:
        levels = levels + rng.normal(0.0, sigma, size=levels.shape).astype(np.float32)
    return delta_windows(levels, True).astype(np.float32)


def mixup_collate(alpha, seed):
    """collate_fn mixing (x, rate) batches: one Beta(alpha, alpha) lambda per
    batch, partner = seeded permutation. Linear in the regression target -
    only valid for the reg head (spec: R_mixup exists only if H_reg won)."""
    rng = np.random.default_rng(seed)

    def collate(items):
        x = torch.stack([it[0] for it in items])
        r = torch.tensor([it[1] for it in items], dtype=torch.float32)
        lam = float(rng.beta(alpha, alpha))
        perm = torch.from_numpy(rng.permutation(len(items)))
        return lam * x + (1 - lam) * x[perm], lam * r + (1 - lam) * r[perm]

    return collate
