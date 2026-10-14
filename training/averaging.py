"""Uniform state-dict averaging for Stage W (spec section 3): model soup
across seeds, SWA across epoch checkpoints. TrendCNN has no BatchNorm, so a
plain parameter mean is exact - no statistics recompute."""
import torch


def average_state_dicts(dicts):
    if not dicts:
        raise ValueError("average_state_dicts: empty input")
    keys = set(dicts[0])
    for d in dicts[1:]:
        if set(d) != keys or any(d[k].shape != dicts[0][k].shape for k in keys):
            raise ValueError("average_state_dicts: mismatched state dicts")
    return {k: torch.stack([d[k].float() for d in dicts]).mean(dim=0) for k in keys}
