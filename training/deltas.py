"""Delta encoding of glucose windows (spec: D12). Pure float32; the same
transform must run before quantization on every platform that ships it."""
import numpy as np

ANCHOR_SCALE = 16.0


def delta_windows(windows, anchor, anchor_scale=ANCHOR_SCALE):
    """(N, L) absolute mg/dL -> (N, L-1) consecutive differences, or
    (N, L) with the last absolute value / anchor_scale appended when
    `anchor` is True. No clipping, float32 in and out."""
    w = np.asarray(windows, dtype=np.float32)
    d = np.diff(w, axis=1)
    if not anchor:
        return d
    return np.concatenate([d, (w[:, -1:] / np.float32(anchor_scale))], axis=1)


def tod_channels(anchor_hours, length=12, step_minutes=5.0):
    """sin/cos time-of-day per window step, derived backwards from the
    anchor at nominal spacing (sweep r3 spec section 4; device-local clock)."""
    anchor_hours = np.asarray(anchor_hours, dtype=np.float64)
    offsets = (np.arange(length) - (length - 1)) * step_minutes / 60.0
    h = (anchor_hours[:, None] + offsets[None, :]) % 24.0
    ang = 2.0 * np.pi * h / 24.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)
