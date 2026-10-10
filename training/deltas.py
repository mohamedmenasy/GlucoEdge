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
