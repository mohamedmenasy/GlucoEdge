"""Hand-checkable metrics for the round-three panel re-analyses."""
import numpy as np
import pytest

from experiments.analyze_calibration_coverage import confusion_counts
from experiments.panel_r3 import precisions, qwk


def test_qwk_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 400)
    p = np.clip(y + rng.integers(-2, 3, 400), 0, 4)
    assert qwk(confusion_counts(y, p)) == pytest.approx(
        cohen_kappa_score(y, p, weights="quadratic", labels=range(5)))


def test_precision_is_nan_for_a_class_never_predicted():
    cm = confusion_counts(np.array([0, 1, 2, 2]), np.array([0, 2, 2, 2]))
    pr = precisions(cm)
    assert pr[0] == 1.0 and pr[2] == pytest.approx(2 / 3)
    assert np.isnan(pr[1]) and np.isnan(pr[3])
