import numpy as np
import torch

from training.train import class_weights, set_seed


class _FakeDataset:
    def __init__(self, labels):
        self.labels = labels


def test_class_weights_handles_missing_class_without_crashing():
    # Classes 0 and 2 present, class 1 (out of 3) has zero examples.
    labels = [0, 0, 0, 2, 2]
    weights = class_weights(_FakeDataset(labels), num_classes=3)
    assert weights.shape == (3,)
    assert weights[1].item() == 1.0  # neutral weight for the absent class
    assert torch.isfinite(weights).all()


def test_class_weights_matches_sklearn_when_all_classes_present():
    from sklearn.utils.class_weight import compute_class_weight

    labels = [0, 0, 1, 1, 1, 2]
    weights = class_weights(_FakeDataset(labels), num_classes=3)
    expected = compute_class_weight("balanced", classes=np.arange(3), y=np.array(labels))
    assert torch.allclose(weights, torch.tensor(expected, dtype=torch.float32))


def test_set_seed_makes_model_init_reproducible():
    from training.model import TrendCNN

    set_seed(42)
    model_a = TrendCNN(num_classes=5)

    set_seed(42)
    model_b = TrendCNN(num_classes=5)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)
