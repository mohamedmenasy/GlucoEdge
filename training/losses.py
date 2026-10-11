"""Loss arms for the accuracy sweep (spec 2026-10-11, Stage A). Every arm
keeps the plain 5-logit softmax head; `weights` is the balanced class-weight
vector from training.train.class_weights. Weighted arms normalize by the
batch's summed true-class weights, matching torch CrossEntropyLoss."""
import torch
import torch.nn.functional as F

LOSSES = ("ce", "ordinal_smooth", "distance_ce", "focal")
EPSILON = 0.15  # ordinal_smooth mass moved to ordinal neighbors
LAMBDA = 0.5    # distance_ce penalty weight
GAMMA = 2.0     # focal focusing exponent


def _smeared_targets(num_classes, epsilon):
    """Row y: 1-epsilon on y, epsilon split over its ordinal neighbors
    (a boundary class gives the whole epsilon to its single neighbor)."""
    q = torch.zeros(num_classes, num_classes)
    for y in range(num_classes):
        neighbors = [j for j in (y - 1, y + 1) if 0 <= j < num_classes]
        q[y, y] = 1.0 - epsilon
        for j in neighbors:
            q[y, j] = epsilon / len(neighbors)
    return q


def make_criterion(name, weights):
    weights = weights.float()
    if name == "ce":
        return torch.nn.CrossEntropyLoss(weight=weights)
    if name == "ordinal_smooth":
        q = _smeared_targets(len(weights), EPSILON)

        def ordinal_smooth(logits, targets):
            logp = F.log_softmax(logits, dim=1)
            per = -(q.to(logits.device)[targets] * logp).sum(dim=1)
            w = weights.to(logits.device)[targets]
            return (per * w).sum() / w.sum()
        return ordinal_smooth
    if name == "distance_ce":
        ce = torch.nn.CrossEntropyLoss(weight=weights)
        dist = torch.arange(len(weights), dtype=torch.float32)

        def distance_ce(logits, targets):
            p = F.softmax(logits, dim=1)
            d = (dist.to(logits.device)[None, :] - targets[:, None].float()).abs()
            return ce(logits, targets) + LAMBDA * (p * d).sum(dim=1).mean()
        return distance_ce
    if name == "focal":
        def focal(logits, targets):
            logp = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
            pt = logp.exp()
            w = weights.to(logits.device)[targets]
            return (-w * (1.0 - pt) ** GAMMA * logp).sum() / w.sum()
        return focal
    raise ValueError(f"unknown loss arm: {name}")
