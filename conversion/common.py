"""Shared helpers for the conversion/ CLIs."""
from pathlib import Path

import torch

from training.model import TrendCNN


def load_checkpoint(checkpoint_path: Path, num_classes: int) -> TrendCNN:
    model = TrendCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model
