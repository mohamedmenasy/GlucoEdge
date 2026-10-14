"""r4 objective heads (spec 2026-10-13 section 3, Stage H): output widths,
training criteria, and the decode back to the 5 trend classes. Decode rules
mirror training.labeling exactly; eval labels never change."""
import numpy as np
import torch
import torch.nn.functional as F

from training.labeling import FIVE_CLASSES, classify_rate

HEADS = ("ce", "wtemp", "reg", "corn")
_WIDTHS = {"ce": 5, "wtemp": 5, "reg": 1, "corn": 4}
HUBER_DELTA = 1.0


def head_outputs(head):
    return _WIDTHS[head]


def sqrt_class_weights(train_ds, num_classes):
    from training.train import class_weights
    return class_weights(train_ds, num_classes).sqrt()


def corn_loss(logits, targets, weights):
    """CORN conditional loss. Output k is trained only on samples with
    y >= k, target 1[y > k]; per-sample class weight w[y]; normalized by
    the summed weights of all trained (sample, output) terms."""
    total = logits.sum() * 0.0
    denom = logits.sum() * 0.0
    w = weights.to(logits.device)[targets]
    for k in range(logits.shape[1]):
        mask = targets >= k
        if not bool(mask.any()):
            continue
        t = (targets[mask] > k).float()
        per = F.binary_cross_entropy_with_logits(logits[mask, k], t, reduction="none")
        total = total + (per * w[mask]).sum()
        denom = denom + w[mask].sum()
    return total / denom


def make_r4_criterion(head, weights):
    weights = weights.float()
    if head in ("ce", "wtemp"):  # wtemp differs only in the weights passed in
        return torch.nn.CrossEntropyLoss(weight=weights)
    if head == "reg":
        huber = torch.nn.HuberLoss(delta=HUBER_DELTA)

        def reg(outputs, targets):
            return huber(outputs.squeeze(-1), targets.float())
        return reg
    if head == "corn":
        def corn(outputs, targets):
            return corn_loss(outputs, targets, weights)
        return corn
    raise ValueError(f"unknown head {head!r}")


def decode_outputs(head, outputs):
    """(B, width) raw model outputs -> (B,) FIVE_CLASSES indices."""
    out = np.asarray(outputs)
    if head in ("ce", "wtemp"):
        return out.argmax(axis=1)
    if head == "reg":
        idx = {c: i for i, c in enumerate(FIVE_CLASSES)}
        return np.array([idx[classify_rate(float(r))] for r in out[:, 0]], dtype=np.int64)
    if head == "corn":
        return (out > 0).sum(axis=1).astype(np.int64)
    raise ValueError(f"unknown head {head!r}")
