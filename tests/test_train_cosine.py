import numpy as np
import torch

from training.train import train_model


class _Tiny:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(64, 1, 12)).astype(np.float32)
        self.y = rng.integers(0, 5, size=64)
        self.labels = self.y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, k):
        return torch.from_numpy(self.x[k]), int(self.y[k])


def _run(cosine):
    torch.manual_seed(0)
    np.random.seed(0)
    ds = _Tiny()
    model = train_model(ds, ds, 5, 3, torch.device("cpu"),
                        batch_size=16, cosine=cosine)
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_cosine_changes_training_and_default_is_off():
    base = _run(False)
    again = _run(False)
    cos = _run(cosine=True)
    assert torch.equal(base, again)          # determinism on cpu
    assert not torch.equal(base, cos)        # scheduler actually wired
