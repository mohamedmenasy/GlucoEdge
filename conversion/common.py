"""Shared helpers for the conversion/ CLIs."""
from pathlib import Path

import torch

from training.model import TrendCNN, TrendTCN


def load_checkpoint(checkpoint_path: Path, num_classes: int):
    # our own files; meta dict is tensors + str/int, so the safe loader works
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        if obj["arch"] == "tcn":
            model = TrendTCN(num_classes=num_classes, channels=obj["channels"],
                             in_channels=obj["in_channels"])
        else:
            model = TrendCNN(num_classes=num_classes,
                             channels=tuple(obj["channels"]),
                             in_channels=obj["in_channels"])
        model.load_state_dict(obj["state_dict"])
    else:
        channels = (obj["conv1.weight"].shape[0], obj["conv2.weight"].shape[0])
        model = TrendCNN(num_classes=num_classes, channels=channels,
                         in_channels=obj["conv1.weight"].shape[1])
        model.load_state_dict(obj)
    model.eval()
    return model
