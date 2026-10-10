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


def test_build_eligible_dataset_filters_by_split_and_mask():
    import pandas as pd

    from training.dataset import GlucoseTrendDataset
    from training.labeling import FIVE_CLASSES
    from training.train import build_eligible_dataset

    rows = [{"id": "p1", "id_segment": 0,
             "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
             "gl": 100.0 + i} for i in range(10)]
    df = pd.DataFrame(rows)
    ds = GlucoseTrendDataset(df, classes=FIVE_CLASSES, input_length=3, horizon=2,
                             horizon_minutes=10.0)  # 6 windows, rows i..i+4

    owner = np.array(["train"] * 5 + ["val"] * 5, dtype=object)
    # Scored val windows: anchor (3rd input row) and both horizon rows
    # val-owned -> windows starting at rows 3, 4, 5 (indices 3, 4, 5).
    eligible = build_eligible_dataset(ds, owner, "val")
    assert len(eligible) == 3
    x, y = eligible[0]
    assert x.shape == (1, 3)
    assert y == ds.labels[3]

    # Masking row 6 as gap-filled drops the windows using it as an input
    # (starts 4 and 5); for the window at 3 it is only an intermediate
    # horizon step, so that window survives.
    no_gap_fill = np.ones(len(df), dtype=bool)
    no_gap_fill[6] = False
    eligible = build_eligible_dataset(ds, owner, "val", no_gap_fill=no_gap_fill)
    assert len(eligible) == 1
    x, y = eligible[0]
    assert y == ds.labels[3]


def test_class_weights_accepts_a_torch_subset():
    from torch.utils.data import Subset

    full = _FakeDataset([0, 0, 1, 1, 1, 2])
    sub = Subset(full, [0, 2, 3, 5])  # labels 0, 1, 1, 2
    weights = class_weights(sub, num_classes=3)
    from sklearn.utils.class_weight import compute_class_weight
    expected = compute_class_weight("balanced", classes=np.arange(3),
                                    y=np.array([0, 1, 1, 2]))
    assert torch.allclose(weights, torch.tensor(expected, dtype=torch.float32))


def test_set_seed_makes_model_init_reproducible():
    from training.model import TrendCNN

    set_seed(42)
    model_a = TrendCNN(num_classes=5)

    set_seed(42)
    model_b = TrendCNN(num_classes=5)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)
