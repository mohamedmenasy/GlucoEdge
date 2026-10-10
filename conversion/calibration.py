"""Deterministic calibration-window selection (calibration-coverage plan,
Section 3).

Index selection is separate from model conversion: this module needs numpy
only, never the LiteRT toolchain, so sampler contracts are testable without
it. Every sampler sees a label-free `pool` built from window inputs and
provenance metadata; future labels cannot influence selection by
construction.

All strategies produce one deterministic full-pool ordering per (strategy,
seed); a budget is its prefix, so the 200-window set is nested in the
1,000-window set by the plan's rule 5.
"""
import hashlib

import numpy as np

STRATEGIES = ("sequential", "uniform_random",
              "participant_balanced_random", "participant_balanced_range_coverage")


def build_pool(ds, indices):
    """Label-free pool arrays for the given window indices of a
    GlucoseTrendDataset: provenance metadata plus per-window input
    extremes (plan rule 1: no future endpoint, no class)."""
    windows = np.stack([ds.windows[i] for i in indices]) if indices else np.empty((0, 0))
    return {
        "participant": np.array([ds.participants[i] for i in indices]),
        "segment": np.array([ds.segments[i] for i in indices]),
        "anchor_time": np.array([ds.anchor_times[i] for i in indices]),
        "input_min": windows.min(axis=1) if len(indices) else np.array([]),
        "input_max": windows.max(axis=1) if len(indices) else np.array([]),
        "window_index": np.array(indices, dtype=int),
    }


def _array_bytes(a):
    a = np.asarray(a)
    if a.dtype.kind in "mM":
        return a.view("i8").tobytes()
    if a.dtype.kind in "fiub":
        return a.tobytes()
    return "\x1f".join(map(str, a.tolist())).encode()


def pool_hash(pool):
    h = hashlib.sha256()
    for key in sorted(pool):
        h.update(key.encode())
        h.update(_array_bytes(pool[key]))
    return h.hexdigest()


def _participants_sorted(pool):
    return np.unique(pool["participant"])


def _participant_order(pool, seed):
    """Shuffled participant order shared by both balanced arms (rule 3)."""
    participants = _participants_sorted(pool)
    perm = np.random.default_rng([seed, 0]).permutation(len(participants))
    return participants[perm]


def _round_robin(pool, seed, member_order_fn):
    """Cycle the shared participant order, skipping exhausted participants;
    `member_order_fn(k, members)` gives each participant's own window order
    (k = the participant's index in the sorted unique list)."""
    participants = _participants_sorted(pool)
    part_index = {p: k for k, p in enumerate(participants)}
    queues = {}
    for p in _participant_order(pool, seed):
        members = np.flatnonzero(pool["participant"] == p)
        queues[p] = list(member_order_fn(part_index[p], members))
    order, cursor = [], {p: 0 for p in queues}
    active = list(queues)
    while active:
        still = []
        for p in active:
            q, c = queues[p], cursor[p]
            if c < len(q):
                order.append(q[c])
                cursor[p] = c + 1
                if c + 1 < len(q):
                    still.append(p)
        active = still
    return order


def _range_cells(pool):
    """Low/middle/high cells from the pool's own 10th/90th percentiles of
    each feature (rule 2); coincident edges merge via deduplication."""
    edges = {}
    for feature in ("input_min", "input_max"):
        q10, q90 = np.percentile(pool[feature], [10, 90])
        edges[feature] = [float(e) for e in np.unique([q10, q90])]
    cell_of = np.stack([
        np.searchsorted(edges["input_min"], pool["input_min"], side="right"),
        np.searchsorted(edges["input_max"], pool["input_max"], side="right"),
    ], axis=1)
    return edges, cell_of


def _ordering(pool, strategy, seed):
    n = len(pool["window_index"])
    if strategy == "sequential":
        return list(np.lexsort((pool["anchor_time"], pool["segment"], pool["participant"]))), None
    if strategy == "uniform_random":
        return list(np.random.default_rng([seed, 4]).permutation(n)), None
    if strategy == "participant_balanced_random":
        fn = lambda k, members: members[np.random.default_rng([seed, 1, k]).permutation(len(members))]
        return _round_robin(pool, seed, fn), None
    if strategy == "participant_balanced_range_coverage":
        edges, cell_of = _range_cells(pool)

        def fn(k, members):
            cells = sorted({tuple(c) for c in cell_of[members]})
            cell_order = np.random.default_rng([seed, 2, k]).permutation(len(cells))
            per_cell = []
            for j, ci in enumerate(cell_order):
                cell_members = members[(cell_of[members] == np.array(cells[ci])).all(axis=1)]
                perm = np.random.default_rng([seed, 3, k, j]).permutation(len(cell_members))
                per_cell.append(list(cell_members[perm]))
            out, c = [], 0
            while any(per_cell):
                q = per_cell[c % len(per_cell)]
                if q:
                    out.append(q.pop(0))
                c += 1
            return out

        cells_record = {"input_min_edges": edges["input_min"],
                        "input_max_edges": edges["input_max"]}
        return _round_robin(pool, seed, fn), cells_record
    raise ValueError(f"unknown strategy {strategy!r}")


def select(pool, strategy, budget, seed):
    """First `budget` windows of the (strategy, seed) ordering, without
    replacement, as a recordable selection (rule 6)."""
    n = len(pool["window_index"])
    if budget > n:
        raise ValueError(f"infeasible budget {budget}: eligible pool has {n} windows")
    positions, cells = _ordering(pool, strategy, seed)
    chosen = positions[:budget]
    record = {
        "strategy": strategy,
        "seed": seed,
        "budget": budget,
        "actual_count": len(chosen),
        "indices": [int(pool["window_index"][i]) for i in chosen],
        "pool_sha256": pool_hash(pool),
    }
    if cells is not None:
        record["range_cells"] = cells
    return record
