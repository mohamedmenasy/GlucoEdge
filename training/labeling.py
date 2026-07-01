FIVE_CLASSES = ["falling_fast", "falling", "stable", "rising", "rising_fast"]

THREE_CLASSES = ["falling", "stable", "rising"]

THREE_CLASS_MAP = {
    "falling_fast": "falling",
    "falling": "falling",
    "stable": "stable",
    "rising": "rising",
    "rising_fast": "rising",
}


def label_trend(last_value: float, future_value: float, horizon_minutes: float = 15.0) -> str:
    rate = (future_value - last_value) / horizon_minutes
    if rate <= -2:
        return "falling_fast"
    elif rate <= -1:
        return "falling"
    elif rate < 1:
        return "stable"
    elif rate < 2:
        return "rising"
    else:
        return "rising_fast"
