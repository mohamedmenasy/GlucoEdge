"""Mixture sampler tests — pure, no data needed."""
import numpy as np

from training.mixtures import (CAP, FAST_CLASSES, HOLDOUT_COHORT,
                               TRAIN_COHORTS, oversample_extra_indices,
                               sample_subjects)


def test_pedap_is_not_a_training_cohort():
    assert HOLDOUT_COHORT == "PEDAP"
    assert "PEDAP" not in TRAIN_COHORTS
    assert CAP == 15_000_000


def test_sample_subjects_whole_subjects_until_budget():
    counts = {f"s{i}": 100 for i in range(10)}
    ids, total = sample_subjects(counts, 250, seed=0)
    assert total == 300            # 3 whole subjects; never a partial subject
    assert len(ids) == 3
    ids2, _ = sample_subjects(counts, 250, seed=0)
    assert ids == ids2             # seeded determinism
    ids3, _ = sample_subjects(counts, 250, seed=1)
    assert ids != ids3             # seed changes the draw (collision odds ~0)


def test_sample_subjects_budget_bigger_than_data_takes_all():
    counts = {"a": 5, "b": 7}
    ids, total = sample_subjects(counts, 10_000, seed=0)
    assert sorted(ids) == ["a", "b"] and total == 12


def test_oversample_reaches_mean_of_other_classes():
    labels = np.array([0] * 10 + [1] * 100 + [2] * 130 + [3] * 70 + [4] * 50)
    extra = oversample_extra_indices(labels, seed=0)
    target = 100  # mean of classes 1,2,3
    new_counts = {c: (labels[extra] == c).sum() + (labels == c).sum() for c in range(5)}
    assert new_counts[0] == target and new_counts[4] == target
    assert all(labels[i] in FAST_CLASSES for i in extra)


def test_oversample_noops_without_fast_windows_or_when_already_enough():
    assert oversample_extra_indices(np.array([1, 2, 3, 2, 1])).size == 0
    assert oversample_extra_indices(np.array([0] * 90 + [1, 2, 3] * 10)).size == 0
