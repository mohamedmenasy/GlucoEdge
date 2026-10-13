import numpy as np

FIVE_CLASSES = ["falling_fast", "falling", "stable", "rising", "rising_fast"]

THREE_CLASSES = ["falling", "stable", "rising"]

THREE_CLASS_MAP = {
    "falling_fast": "falling",
    "falling": "falling",
    "stable": "stable",
    "rising": "rising",
    "rising_fast": "rising",
}


def classify_rate(rate: float) -> str:
    """Rate in mg/dL/min -> one of FIVE_CLASSES. Boundary rules are the
    clinical anchors: <= on the falling side, < on the rising side."""
    if rate <= -2:
        return "falling_fast"
    elif rate <= -1:
        return "falling"
    elif rate < 1:
        return "stable"
    elif rate < 2:
        return "rising"
    return "rising_fast"


def label_trend(last_value: float, future_value: float, horizon_minutes: float = 15.0) -> str:
    # last_value/future_value are mg/dL (GlucoBench leaves `gl` unscaled), so
    # rate is mg/dL per minute - the thresholds assume that unit.
    return classify_rate((future_value - last_value) / horizon_minutes)


def lsq_slope(points, step_minutes: float = 5.0):
    """(N, K) equally-spaced mg/dL readings -> (N,) least-squares slope in
    mg/dL/min. Closed form: sum((t-tbar)(g-gbar)) / sum((t-tbar)^2)."""
    g = np.asarray(points, dtype=np.float64)
    t = np.arange(g.shape[1], dtype=np.float64) * step_minutes
    tc = t - t.mean()
    return ((g - g.mean(axis=1, keepdims=True)) * tc).sum(axis=1) / (tc ** 2).sum()
