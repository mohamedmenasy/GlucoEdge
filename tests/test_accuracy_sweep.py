"""Pure-logic tests for the accuracy sweep (no GlucoBench needed)."""
import numpy as np
import pytest

from experiments.accuracy_sweep import (ConcatView, arm, cost_key,
                                        paired_rows, param_count,
                                        pick_winner, stage_a_arms,
                                        stage_b_arms)


def test_param_count_matches_default_architecture():
    assert param_count((24, 36)) == 2909


def test_stage_a_arms_cover_all_losses_at_baseline():
    arms = stage_a_arms()
    assert [a["loss"] for a in arms] == ["ce", "ordinal_smooth", "distance_ce", "focal"]
    assert all(a["channels"] == [24, 36] and a["length"] == 12 for a in arms)


def test_stage_b_arms_exclude_reused_baseline_cell():
    arms = stage_b_arms("focal", include_36=False)
    names = {a["name"] for a in arms}
    assert "B_c24x36_L12" not in names
    assert len(arms) == 5  # 3 channel grids x 2 lengths - reused cell
    arms36 = stage_b_arms("focal", include_36=True)
    assert len(arms36) == 8


def test_pick_winner_tie_prefers_cheaper_arm():
    a = arm("A_ce", "ce", (24, 36), 12)
    b = arm("B_big", "ce", (96, 144), 24)
    # b higher but within the 0.1pp tie band -> cheaper a wins
    assert pick_winner([a, b], {"A_ce": 0.5000, "B_big": 0.5009}) == a
    # outside the band -> b wins
    assert pick_winner([a, b], {"A_ce": 0.5000, "B_big": 0.5011}) == b


def test_cost_key_orders_params_length_pooled_loss():
    cheap = arm("x", "ce", (24, 36), 12)
    pooled = arm("y", "ce", (24, 36), 12, pooled=True)
    fancy_loss = arm("z", "focal", (24, 36), 12)
    assert cost_key(cheap) < cost_key(pooled)
    assert cost_key(cheap) < cost_key(fancy_loss)


def test_concat_view_concatenates_labels_and_items():
    class V:
        def __init__(self, labels):
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, k):
            return ("item", self.labels[k])

    cv = ConcatView([V([0, 1]), V([2])])
    assert cv.labels == [0, 1, 2]
    assert len(cv) == 3
    assert cv[2] == ("item", 2)


def test_paired_rows_intersects_on_keys_in_a_order():
    ka = [("p1", "t1"), ("p1", "t2"), ("p2", "t1")]
    kb = [("p2", "t1"), ("p1", "t1")]
    ia, ib = paired_rows(ka, kb)
    assert ia == [0, 2]
    assert ib == [1, 0]
