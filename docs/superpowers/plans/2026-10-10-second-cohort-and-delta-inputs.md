# Second-Cohort Confirmation + Delta-Encoded Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm the sequential-calibration collapse on a second T1D cohort with a prespecified narrow-range control (dubosson + hall), then test whether delta-encoded inputs make INT8 insensitive to calibration choice — both into the paper before arXiv.

**Architecture:** Part A parameterizes the existing study machinery (`load_cohort_context`, stage `--dataset/--strategies/--budgets`) and runs a reduced locked-evaluation matrix per cohort. Part B adds a pure delta transform over the same eligible windows, a mirrored runner, and gated INT8/ship decisions.

**Tech Stack:** existing study venv (torch 2.12.1, litert-torch 0.9.1, ai-edge-litert); tectonic for the paper; Swift/Kotlin only in the conditional ship task.

**Spec:** `docs/superpowers/specs/2026-10-10-second-cohort-and-delta-inputs-design.md`

## Global Constraints

- Not a medical device: no code, comment, or output string implies clinical validity or treatment guidance.
- GlucoBench is read-only, never committed; no participant IDs, timestamps, or per-window predictions in committed files.
- Locked evaluation per cohort: the protocol JSON is committed BEFORE any scored label of that cohort's evaluation components is read; the availability audit reads counts only.
- Run everything with the repo venv (`venv/bin/python`); CI must never need GlucoBench (new unit tests use synthetic frames only).
- Bundled app models and goldens are untouched except by the conditional ship task (B5), and only if both gates pass.
- All randomness seeded: training seeds 0–4, draws 101–103, bootstrap seed 20261010.

## Review Focus

- A cohort whose grid interval is not 5 minutes: the no-gap-fill threshold must scale to 1.5× the modal interval, not stay at 450 s — pinned by `test_cohort_interval_scales_threshold` (Task 1).
- A cohort whose eligible calibration pool is smaller than the 200 budget: selection must fail loudly ("infeasible budget"), and the protocol records the reduced budget — exercised in Task 3 step 2 before freezing.
- Delta windows at INT8 ceiling inputs: a 401-cap window's deltas are small; the transform must not clip or wrap — pinned by `test_delta_windows_match_hand_computation` (Task 6).
- The anchor channel mixing scales: anchor/16 must stay within the delta grid for glucose 40–400 — pinned by `test_anchor_channel_is_scaled_into_delta_range` (Task 6).
- Evaluate-stage reuse with missing matrix cells (confirmation runs lack 1000-budget and balanced arms): the analyzer must emit the uniform-vs-sequential contrast and not crash on absent artifacts — exercised in Task 3 step 7 (its aggregate must contain `uniform200_vs_sequential200` with non-null delta).

---

### Task 1: Cohort-parameterized context with interval-scaled gap threshold

**Files:**
- Modify: `training/splits.py` (observation_status gains `max_gap=None`)
- Modify: `experiments/data_availability.py` (`load_cohort_context`, `cohort_interval`)
- Test: `tests/test_split_audit.py`, `tests/test_dataset.py`

**Interfaces:**
- Produces: `load_cohort_context(dataset: str = "weinstock") -> (formatter, owner, on_reading, no_gap_fill)`; `load_weinstock_context()` kept as `load_cohort_context()` wrapper so existing callers are untouched. `cohort_interval(times: np.ndarray) -> np.timedelta64` (modal positive diff). `observation_status(grid, readings, max_gap=None)` — `None` means the existing `MAX_ADJACENT_GAP`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_split_audit.py
def test_observation_status_accepts_a_custom_gap_threshold():
    t = lambda minutes: pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minutes)
    readings = pd.DataFrame({"id": ["a", "a"], "time": [t(0), t(10)]})
    grid = pd.DataFrame({"id": ["a"] * 3, "time": [t(0), t(5), t(10)]})
    # Default 7.5-min threshold: the 10-min bracket is a gap.
    _, strict = observation_status(grid, readings)
    assert strict.tolist() == [True, False, True]
    # A 15-minute threshold (1.5 x a 10-min cohort interval) accepts it.
    _, loose = observation_status(grid, readings,
                                  max_gap=np.timedelta64(900, "s"))
    assert loose.tolist() == [True, True, True]


def test_cohort_interval_scales_threshold():
    from experiments.data_availability import cohort_interval
    t = pd.Timestamp("2024-01-01")
    times = pd.Series([t + pd.Timedelta(minutes=m) for m in
                       (0, 15, 30, 45, 60, 61)]).to_numpy()
    assert cohort_interval(times) == np.timedelta64(15 * 60, "s")
```

- [ ] **Step 2: Run them, confirm both FAIL** (`venv/bin/python -m pytest tests/test_split_audit.py -q`) — Expected: TypeError on `max_gap` and ImportError on `cohort_interval`.

- [ ] **Step 3: Implement** — in `training/splits.py`, `observation_status(grid, readings, max_gap=None)` with `max_gap = MAX_ADJACENT_GAP if max_gap is None else max_gap` and the comparison using `max_gap`. In `experiments/data_availability.py`:

```python
def cohort_interval(times):
    """Modal positive spacing of a time array (np.timedelta64)."""
    diffs = np.diff(np.sort(times))
    diffs = diffs[diffs > np.timedelta64(0, "s")]
    vals, counts = np.unique(diffs, return_counts=True)
    return vals[np.argmax(counts)].astype("timedelta64[s]")
```

and rename `load_weinstock_context` to `load_cohort_context(dataset="weinstock")`: formatter/csv paths use `dataset`, `max_gap = 1.5 * cohort_interval(data["time"].to_numpy())` (as timedelta64[s]) passed to `observation_status`; add `def load_weinstock_context(): return load_cohort_context()`.

- [ ] **Step 4: Full suite passes** (`venv/bin/python -m pytest -q`) — Expected: all green (weinstock behavior unchanged: its modal interval is 5 min → 450 s, identical threshold).
- [ ] **Step 5: Runtime check on real data** — `venv/bin/python -m experiments.data_availability` regenerates `paper/results/calibration-coverage/data_availability.json` byte-identically (`git diff --stat` empty). Expected: no diff.
- [ ] **Step 6: Commit** `feat(study): cohort-parameterized context with interval-scaled gap threshold`

### Task 2: Stage CLI gains --dataset, --strategies, --budgets

**Files:**
- Modify: `experiments/calibration_coverage.py` (`_load_study_data(dataset)`, `stage_*` signatures, `main()`)
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `load_cohort_context(dataset)` from Task 1.
- Produces: `python -m experiments.calibration_coverage {train,select,convert,evaluate} --run-id R --dataset D [--strategies a,b] [--budgets 200]`. `_load_study_data(dataset="weinstock")`. `stage_select(run_id, dataset, strategies=None, budgets=None)` — `None` means all. The run's protocol/provenance stays per-run-id under `results/calibration-coverage/<run-id>/`.

- [ ] **Step 1: Write the failing test** (pure cell-filtering helper so the restriction is unit-testable without data):

```python
# tests/test_calibration.py
def test_selection_cells_filter_by_strategy_and_budget():
    from experiments.calibration_coverage import selection_cells
    cells = selection_cells(strategies=("sequential", "uniform_random"),
                            budgets=(200,))
    assert ("sequential", 200, 0) in cells
    assert ("uniform_random", 200, 101) in cells
    assert all(b == 200 for _, b, _ in cells)
    assert not any(s.startswith("participant") for s, _, _ in cells)
```

- [ ] **Step 2: Run it, confirm FAIL** — Expected: ImportError `selection_cells`.
- [ ] **Step 3: Implement** — extract the cell enumeration in `stage_select` into module-level `selection_cells(strategies=None, budgets=None)` returning `[(strategy, budget, draw)]` (sequential draw is 0); `stage_select` consumes it. Thread `dataset` through `_load_study_data(dataset)` → `load_cohort_context(dataset)` and through every stage; `main()` adds `--dataset` (default `weinstock`), `--strategies` (comma list, default all), `--budgets` (comma ints, default `200,1000`).
- [ ] **Step 4: Full suite passes.** Expected: green; existing weinstock run r1 untouched.
- [ ] **Step 5: Commit** `feat(study): dataset/strategy/budget-parameterized stages`

### Task 3: Dubosson confirmation run (audit → freeze → execute → aggregate)

**Files:**
- Create: `results/calibration-coverage/conf-dubosson/protocol.json` (local run dir, gitignored) and committed copy `paper/results/calibration-coverage/confirmation_dubosson.json` (protocol + final aggregate, merged at the end)
- Test: runtime stage outputs (no new unit tests)

**Interfaces:**
- Consumes: Tasks 1–2 CLIs.
- Produces: `paper/results/calibration-coverage/confirmation_dubosson.json` with keys `protocol`, `components`, `contrast_uniform200_vs_sequential200` — Task 5 reads these names.

- [ ] **Step 1: Labels-untouched availability audit** — run `venv/bin/python - <<'EOF'` script calling `load_cohort_context("dubosson")` + `split_summary` for train/val/test/held-out (counts only; classes only for train/calibration pool, mirroring `experiments/data_availability.py:main`). Print eligible counts, participant counts, pool size, modal interval. Expected: pool size known; note whether OOD (held_out) component is non-empty.
- [ ] **Step 2: Fix the budget** — if pool ≥ 200, budget = 200; else the largest multiple of 50 ≤ pool. Record the number and the pool size.
- [ ] **Step 3: Freeze the protocol** — write `results/calibration-coverage/conf-dubosson/protocol.json`:

```json
{"study": "second-cohort confirmation", "dataset": "dubosson",
 "prediction": "uniform_random minus sequential directional recall at the fixed budget is POSITIVE on this wide-range T1D cohort",
 "strategies": ["sequential", "uniform_random"], "budgets": [200],
 "draws": [101, 102, 103], "training_seeds": [0, 1, 2, 3, 4],
 "mask": "no-gap-fill at 1.5x modal grid interval",
 "components": "<held_out and/or temporal, per step 1 counts>",
 "bootstrap": {"n_boot": 2000, "seed": 20261010},
 "frozen_before_any_scored_label_read": true}
```

with the measured budget/components substituted, and commit it (plus a copy at `paper/results/calibration-coverage/confirmation_dubosson.json` holding only the protocol for now): `git commit -m "chore(study): freeze dubosson confirmation protocol"`. Expected: committed before any evaluate stage.
- [ ] **Step 4: Train** — `venv/bin/python -m experiments.calibration_coverage train --run-id conf-dubosson --dataset dubosson` (background; 5 seeds; small cohort so minutes). Expected: 5 checkpoints + metadata.
- [ ] **Step 5: Select + convert** — `select --run-id conf-dubosson --dataset dubosson --strategies sequential,uniform_random --budgets 200` then `convert ...` (same flags). Expected: 4 selections, 5 float + 20 INT8 artifacts with metadata.
- [ ] **Step 6: Evaluate (locked gate)** — `evaluate --run-id conf-dubosson --dataset dubosson`. Expected: components.npz + per-artifact caches; first label read of this cohort.
- [ ] **Step 7: Aggregate** — `venv/bin/python -m experiments.analyze_calibration_coverage --run-id conf-dubosson`; Expected: `analysis/aggregate.json` whose `contrasts.*.uniform200_vs_sequential200.delta` is non-null (missing matrix cells tolerated). Then merge into the committed JSON with a short script: `{"protocol": <frozen>, "components": {comp: {name: metrics for float seeds, sequential, uniform arms}}, "contrast_uniform200_vs_sequential200": <per component>}` — strip the 100-cell listing and anything empty.
- [ ] **Step 8: Commit** `results(study): dubosson confirmation aggregate` (committed JSON only; run dir stays local).

### Task 4: Hall confirmation run

**Files:** as Task 3 with `hall` / `conf-hall` / `confirmation_hall.json`.

**Interfaces:** Produces `paper/results/calibration-coverage/confirmation_hall.json`, same key names as Task 3.

- [ ] **Steps 1–8: identical to Task 3** with dataset `hall` and this frozen prediction line:

```json
"prediction": "uniform_random minus sequential directional recall at the fixed budget is NEAR ZERO on this narrow-range cohort (max glucose ~303, ~0% readings above 240): sequential selection cannot meaningfully under-cover the population range"
```

Expected: hall is 57 participants / 105k rows — training takes longer than dubosson; run stages in background and verify counts at each stage as in Task 3.

### Task 5: Paper — confirmation subsection and table

**Files:**
- Modify: `paper/main.tex` (new Results subsection after the sensitivity tiers; Limitations cohort bullet; conclusion sentence), `paper/README.md`
- Test: tectonic build + page render

- [ ] **Step 1: Write the subsection** — structure (values substituted from the two committed JSONs):

```latex
\subsection{Second-cohort confirmation}
Two prespecified predictions were frozen per cohort before any of its
evaluation labels were read: the uniform-minus-sequential directional-recall
difference at the fixed budget is positive on dubosson (type 1 diabetes,
nine participants, readings up to 400~mg/dL) and near zero on hall
(57 participants, maximum 303~mg/dL, almost no readings above 240) ...
\begin{table}[t] ... two cohorts x {float, seq, uniform, delta [CI]} ... \end{table}
```

reporting each cohort's component(s), budget actually used, and interval widths honestly (dubosson's will be wide). State plainly whether each prediction held.
- [ ] **Step 2: Update Limitations** ("conclusions are scoped to one cohort" bullet becomes the measured two-cohort statement) and the conclusion's future-work sentence (second-cohort confirmation no longer future work).
- [ ] **Step 3: Build + check** — `tectonic --keep-intermediates --keep-logs main.tex`; Expected: 0 overfull (wrap the new table in `\resizebox{\columnwidth}{!}{...}` if needed), render the new page with pdftoppm and look at it.
- [ ] **Step 4: paper/README.md revision note + page/table count update; commit** `docs(paper): second-cohort confirmation results`.

### Task 6: Delta transform (pure, TDD)

**Files:**
- Create: `training/deltas.py`
- Test: `tests/test_deltas.py`

**Interfaces:**
- Produces: `delta_windows(windows: np.ndarray, anchor: bool, anchor_scale: float = 16.0) -> np.ndarray` — input (N, 12) float32 absolute mg/dL; output (N, 11) of consecutive differences when `anchor=False`, else (N, 12) = 11 differences + `windows[:, -1] / anchor_scale` appended. Float32 in/out, no clipping. `ANCHOR_SCALE = 16.0` module constant. Task 7 and B5 both consume exactly this function.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deltas.py
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
```

- [ ] **Step 2: Run, confirm FAIL** — Expected: ModuleNotFoundError.
- [ ] **Step 3: Implement**

```python
"""Delta encoding of glucose windows (spec: D12). Pure float32; the same
transform must run before quantization on every platform that ships it."""
import numpy as np

ANCHOR_SCALE = 16.0


def delta_windows(windows, anchor, anchor_scale=ANCHOR_SCALE):
    w = np.asarray(windows, dtype=np.float32)
    d = np.diff(w, axis=1)
    if not anchor:
        return d
    return np.concatenate([d, (w[:, -1:] / np.float32(anchor_scale))], axis=1)
```

- [ ] **Step 4: Suite green; commit** `feat(deltas): pure delta-window transform with optional scaled anchor`

### Task 7: Delta-inputs runner and weinstock experiment

**Files:**
- Create: `experiments/delta_inputs.py`
- Create (frozen before evaluate): `results/calibration-coverage/delta-r1/protocol.json`; committed `paper/results/calibration-coverage/delta_inputs_r1.json`
- Test: `tests/test_deltas.py` (runner's pure helper), runtime stages

**Interfaces:**
- Consumes: `delta_windows` (Task 6); `_load_study_data`, `build_eligible_dataset`-style index logic, `convert_float/convert_int8`, `selection_cells`, analyzer metric functions — all existing.
- Produces: CLI `python -m experiments.delta_inputs {train,convert,evaluate,analyze} --run-id delta-r1 --variant {plain,anchor}`; committed `delta_inputs_r1.json` with `float_gate`, per-variant metrics, and `int8_robustness` keys that Task 8 reads.

- [ ] **Step 1: Failing test for the one pure helper** (variant input length):

```python
def test_variant_input_length():
    from experiments.delta_inputs import variant_length
    assert variant_length("plain") == 11
    assert variant_length("anchor") == 12
```

- [ ] **Step 2: FAIL, then implement the runner** — mirror `calibration_coverage.py` stages but: training/eval windows pass through `delta_windows(ws, anchor=(variant=="anchor"))`; conversion `sample = torch.zeros(1, 1, variant_length(variant))` and calibration inputs are delta-transformed the same way; checkpoints at `results/calibration-coverage/delta-r1/checkpoints/<variant>_seed{N}.pt`; artifacts `<variant>_seed{N}_float.tflite` and `<variant>_seed{N}_int8_{sequential_200_d0|uniform_random_200_d{draw}}.tflite` (reuse the EXISTING r1 selections — same val windows, so copy `results/calibration-coverage/r1/selections/{sequential_200_d0,uniform_random_200_d101..103}.json`); evaluate caches per component under `delta-r1/eval/`. TrendCNN itself is unchanged (AdaptiveAvgPool1d makes it length-agnostic).
- [ ] **Step 3: Freeze protocol** — commit `delta-r1/protocol.json` (variants, gates verbatim from the spec: float gate ≥ 0.482 five-seed held-out directional mean; INT8 prediction uniform−sequential ≈ 0; post-hoc-on-touched-data disclosure) BEFORE the first held-out metric is computed. `git commit -m "chore(study): freeze delta-inputs protocol"`.
- [ ] **Step 4: Train both variants** (background): `train --run-id delta-r1 --variant plain` and `--variant anchor` (5 seeds each, ~10 min/seed on weinstock).
- [ ] **Step 5: Float gate** — `evaluate` float-only paths; compute five-seed held-out directional means per variant. Expected: a decision — surviving variant per spec rule (≥ 0.482; ties → plain). If NEITHER passes, skip to Task 8 and write the negative result.
- [ ] **Step 6: INT8 arms for the survivor** — convert (sequential d0 + uniform 101–103, budget 200, 5 seeds = 20 INT8), evaluate, analyze: per-arm metrics + paired bootstrap for uniform−sequential and float−INT8(sequential), both components.
- [ ] **Step 7: Publish** `paper/results/calibration-coverage/delta_inputs_r1.json` (protocol + gate outcomes + metrics + contrasts); commit `results(study): delta-inputs experiment aggregate`.

### Task 8: Paper — delta-inputs subsection

**Files:**
- Modify: `paper/main.tex`, `paper/README.md`

- [ ] **Step 1: Write the subsection** — either outcome gets reported: the robustness table (variant × {float, INT8-sequential, INT8-uniform} directional recall with the two contrasts), or the negative result ("neither delta variant matched the absolute-input float baseline; delta encoding is not a free robustness fix at this model size"). Include the input-grid fact (measured INT8 input scale of the delta artifact vs 1.57 for absolute). Label the experiment post-hoc on previously touched held-out data, same framing as the sensitivity tiers.
- [ ] **Step 2: Build, render-check, update paper/README.md counts; commit** `docs(paper): delta-encoded inputs results`.

### Task 9 (CONDITIONAL — both ship-gate criteria met): Ship delta models to both apps

Run ONLY if Task 7 step 5 passed the float gate AND INT8-under-sequential is within 0.5 pp of the delta float (spec ship gate). Otherwise record `Ruling: ship gate failed — <numbers>` in the ledger and skip.

**Files:**
- Modify: `android/app/src/main/kotlin/com/glucoedge/app/inference/TrendClassifier.kt` (delta transform before quantization), `ios/GlucoEdge/Inference/TrendClassifier.swift` (same), `android/app/src/main/assets/{trend_float,trend_int8}.tflite` + `MODELS.md`, `conversion/export_golden_vectors.py` (delta-aware), regenerated `golden_vectors.json`, `README.md`
- Test: new Kotlin unit test + Swift test asserting the transform matches `delta_windows` on hand values; full parity suites

- [ ] **Step 1: RED** — Kotlin test `deltaTransformMatchesPython()` and Swift `deltaTransformMatchesPython()` with the Task 6 hand vectors (`[100,103,101,110] -> [3,-2,9]`, anchor `400/16 = 25`). Expected: fail (no transform).
- [ ] **Step 2: GREEN** — implement the identical float32 transform in both clients, applied before input quantization; same subtraction order (`w[i+1]-w[i]`).
- [ ] **Step 3: Swap assets** — copy the surviving variant's seed-0 float + INT8(uniform-200-d101) artifacts; regenerate goldens via the delta-aware exporter; sizes and sha256s into MODELS.md with provenance and gate numbers.
- [ ] **Step 4: Verify** — `pytest` green; `./android/gradlew -p android :app:assembleDebug :app:testDebugUnitTest`; iOS simulator suite; on-device parity if the iPhone is attached (ask the user), else mark pending in MODELS.md.
- [ ] **Step 5: Commit as its own PR** `feat(models): ship delta-encoded models` — separate from the research PR per spec.

### Task 10: Memory + wrap-up

- [ ] Update `~/.claude/.../memory/project_status.md` (phase outcomes, gate decisions, what shipped); final summary lists every Ruling and deferred minor; delete this plan's SDD workspace after the final review is clean.
