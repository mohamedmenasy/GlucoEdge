"""Pure mixture helpers for sweep r2: subject-level sampling and
fast-class oversampling. No I/O here."""
import numpy as np

TRAIN_COHORTS = ("CITY", "DLCP3", "RBG", "SENCE", "WISDM")  # observed Database names
HOLDOUT_COHORT = "PEDAP"  # reserved for the zero-shot secondary eval
CAP = 15_000_000
FAST_CLASSES = (0, 4)     # falling_fast, rising_fast in FIVE_CLASSES order


def sample_subjects(counts, budget, seed):
    """Draw whole subjects (shuffled by seed) until the window budget is
    met or exceeded. A subject's windows are never split."""
    rng = np.random.default_rng(seed)
    ids = sorted(counts)
    rng.shuffle(ids)
    out, total = [], 0
    for sid in ids:
        if total >= budget:
            break
        out.append(sid)
        total += counts[sid]
    return out, total


def oversample_extra_indices(labels, seed=0):
    """Indices to append so each fast class reaches the mean count of the
    other three classes. Empty when nothing to do."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    counts = {c: int((labels == c).sum()) for c in range(5)}
    target = int(np.mean([counts[c] for c in range(5) if c not in FAST_CLASSES]))
    extra = []
    for c in FAST_CLASSES:
        pool = np.flatnonzero(labels == c)
        if pool.size == 0 or counts[c] >= target:
            continue
        extra.append(rng.choice(pool, target - counts[c], replace=True))
    return np.concatenate(extra) if extra else np.array([], dtype=np.int64)
