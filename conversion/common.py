"""Shared helpers for the conversion/ CLIs."""
from pathlib import Path

import torch

from training.model import TrendCNN


def load_checkpoint(checkpoint_path: Path, num_classes: int) -> TrendCNN:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    channels = (state["conv1.weight"].shape[0], state["conv2.weight"].shape[0])
    model = TrendCNN(num_classes=num_classes, channels=channels)
    model.load_state_dict(state)
    model.eval()
    return model
