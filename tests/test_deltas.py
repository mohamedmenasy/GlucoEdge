import numpy as np

from training.deltas import ANCHOR_SCALE, delta_windows


def test_delta_windows_match_hand_computation():
    w = np.array([[100., 103., 101., 110.]], dtype=np.float32)
    out = delta_windows(w, anchor=False)
    assert out.dtype == np.float32 and out.shape == (1, 3)
    assert out.tolist() == [[3.0, -2.0, 9.0]]
    # Ceiling-region values produce ordinary small deltas - nothing clips.
    cap = np.array([[395., 401., 399., 401.]], dtype=np.float32)
    assert delta_windows(cap, anchor=False).tolist() == [[6.0, -2.0, 2.0]]


def test_anchor_channel_is_scaled_into_delta_range():
    w = np.array([[40., 45., 50., 400.]], dtype=np.float32)
    out = delta_windows(w, anchor=True)
    assert out.shape == (1, 4)
    assert out[0, -1] == np.float32(400.0 / ANCHOR_SCALE)  # 25.0
    assert abs(out[0, -1]) <= 30.0  # inside the delta grid for 40-400 mg/dL


def test_variant_input_length():
    from experiments.delta_inputs import variant_length
    assert variant_length("plain") == 11
    assert variant_length("anchor") == 12
