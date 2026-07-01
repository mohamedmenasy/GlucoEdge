# GlucoEdge: Training Pipeline Rebuild + Weinstock Scale-Up

Date: 2026-06-30
Status: Approved, pending implementation plan

## Context

GlucoEdge is a portfolio project demonstrating on-device ML deployment: a small
1D-CNN classifies a 5-class glucose trend arrow (`falling_fast` / `falling` /
`stable` / `rising` / `rising_fast`) 15 minutes ahead, using only public
GlucoBench CGM data. It is explicitly not a medical device, and must not
contain any proprietary code, data schemas, or algorithms from prior
FDA-regulated work.

The project brief described a working starting point,
`train_trend_classifier.py`, with a `DataFormatter` setup, a `label_trend()`
function, a `GlucoseTrendDataset`, and a `TrendCNN` model, verified end-to-end
on GlucoBench's small 4-patient iglu config. On starting this session, that
file — and every symbol it was said to define — was confirmed absent from the
entire machine (empty project directory, no filename or content match
anywhere, no prior session memory for this project). The user confirmed:
rebuild it fresh from the description rather than wait for a copy to be
recovered.

This spec covers only the rebuild + the immediate next step from the original
plan (swap to the 200-patient weinstock config, resolve the 5-class vs
3-class question). LiteRT conversion/quantization, the Android app, and the
README are separate follow-on sub-projects, each to get their own spec once
this one is implemented and verified.

## Grounding facts (from reading the actual GlucoBench source, not assumption)

GlucoBench was cloned to `GlucoBench/` inside the project to check these
before designing against them:

- `data_formatter.base.DataFormatter(config)` exposes `.train_data`,
  `.val_data`, `.test_data` as pandas DataFrames with columns `id`, `time`,
  `gl` (plus static covariates for weinstock, which we ignore).
- Both `iglu.yaml` and `weinstock.yaml` set `scaling_params.scaler: None` —
  `gl` stays in raw mg/dL. Thresholds can be defined directly in clinical
  units, no de-normalization needed.
- After interpolation, GlucoBench tags every contiguous gap-free run (at the
  5-minute observation interval) with an `id_segment` column. **Windows must
  never cross an `(id, id_segment)` boundary** — doing so would silently
  splice together time periods separated by a dropped gap. This constraint
  was not visible from the original description and only showed up from
  reading `data_formatter/utils.py::interpolate` and `::split`.
- `data_formatter` itself only imports numpy, pandas, torch, and
  scikit-learn — none of GlucoBench's heavier benchmark-only dependencies
  (darts, optuna, pytorch-lightning, xgboost, statsforecast, pmdarima). Our
  own `requirements.txt` needs those four plus PyYAML (to load the
  `config/*.yaml` files) and pytest (dev-only, for the tests below).
- GlucoBench has no LICENSE file; its README asks for citation on reuse.
  Treated as: clone it as an external dependency, never vendor its code or
  data into GlucoEdge's own git history, cite the paper in the final README.

## Decisions

### Code structure

A small modular package under `training/`, not a single script:

```
GlucoEdge/
├── GlucoBench/              # cloned upstream, gitignored, not vendored
├── training/
│   ├── __init__.py
│   ├── labeling.py          # label_trend()
│   ├── dataset.py           # GlucoseTrendDataset
│   ├── model.py             # TrendCNN
│   └── train.py             # CLI entry point (--dataset iglu|weinstock)
├── tests/
│   ├── test_labeling.py
│   ├── test_dataset.py
│   └── test_model.py
├── requirements.txt
└── docs/superpowers/specs/
```

Rejected alternatives: a single monolithic script (matches the original
description most literally, but phase 2 needs to re-import the same
`TrendCNN`/`label_trend` for eval parity, and phase 3 needs the labeling
logic isolated enough to port to Kotlin — a monolith fights both); a
notebook-first exploration (fine for eyeballing class balance, but leaves a
throwaway artifact and the brief already implies a script/CLI workflow).

`train.py` takes `--dataset {iglu,weinstock}` and resolves
`GlucoBench/config/{dataset}.yaml`, rather than a hardcoded `CONFIG_PATH`
constant that has to be hand-edited to switch datasets.

### Labeling (`label_trend`)

```
rate = (gl[t+3] - gl[t]) / 15          # mg/dL per minute
```

where `t` is the last point of the 1-hour (12-point, 5-min spacing) input
window and `t+3` is 3 steps / 15 minutes later — entirely outside the input
window, so this is a genuine forecast target, not a recomputation of
something the model can already see.

Thresholds, anchored to real CGM trend-arrow conventions and simplified from
7 buckets (most devices distinguish a "slowly" tier) down to the 5 this
project targets:

| Condition | Class |
|---|---|
| `rate ≤ -2` | `falling_fast` |
| `-2 < rate ≤ -1` | `falling` |
| `-1 < rate < 1` | `stable` |
| `1 ≤ rate < 2` | `rising` |
| `rate ≥ 2` | `rising_fast` |

These are a starting point, not fixed constants, but threshold nudging is
the lighter-touch remedy and is tried first, before the heavier remedy of
collapsing classes (below). Concretely: if a "fast" class looks rare, first
check whether nudging its boundary by a small amount (e.g. 2.0 → 1.5
mg/dL/min) meaningfully changes its example count — if the class is
genuinely sparse regardless of where the line is drawn, threshold tuning
would just be curve-fitting the definition to the data, so the response is
to collapse classes instead (see decision procedure below), not to keep
sliding the boundary until it looks better. Either path — a moved boundary
or a collapse — gets documented with the numbers that drove it.

### Windowing (`GlucoseTrendDataset`)

Group rows by `(id, id_segment)`, sort by time within each group, slide a
window of 12 input points + 3 horizon points with **stride 1** (a new sample
every 5 minutes). Only emit a window when the full 15-point span fits inside
one segment — never across a segment or patient boundary.

Stride 1 (full overlap) is a deliberate choice, not just a convenient
default: once deployed, the model will be queried on a new 1-hour window
every 5 minutes off a live/replayed stream. Training on every possible
5-minute-shifted window matches that inference-time distribution exactly,
rather than training on an artificially decorrelated subsample.

### Model (`TrendCNN`)

A small 1D-CNN, 2-3 conv layers plus a small FC head, input shape
`(batch, 1, 12)`, output 5 (or 3, see below) logits. Exact channel widths are
tuned in code to land near the ~2,850-param budget from the original brief —
this was chosen over a GRU/LSTM specifically because it converts to LiteRT
with a fixed input shape and no recurrent state to manage at inference.
Exact numbers get finalized during implementation (print `sum(p.numel() for p
in model.parameters())` and adjust channel widths), not decided in this spec.

### Class imbalance handling

Class-weighted cross-entropy loss, weights from inverse class frequency in
`train_data`. This is reported alongside — not instead of — per-class
recall / `classification_report`, since weighting changes what the loss
penalizes but cannot manufacture missing examples for a class that is
genuinely rare.

### 5-class vs 3-class decision procedure

Train 5-class on weinstock first. Collapse to 3-class
(`falling`/`stable`/`rising`, merging the "fast" tiers into their base
direction) **only if**, after training:

- either "fast" class has fewer than ~200 training-window examples, **or**
- its test-set recall stays within noise of the 5-class chance baseline
  (~20%)

If either trigger fires, retrain 3-class and report both results side by
side. `train.py` writes the raw counts and per-class recall numbers that
drove the call into a results artifact (see below) — the decision must be
traceable from data, not asserted.

### Output artifacts

`train.py` writes `results/{dataset}_{n_classes}class_report.json`
containing: the thresholds used, per-class support/precision/recall/f1 from
`classification_report`, and a one-line summary of whether the 3-class
collapse triggered and why. This is the concrete home for "document the
decision rather than hiding it" — not a console log that scrolls away.

### Testing

- `label_trend()`: pure function over plain values — unit tests for each
  threshold boundary (including the exact edges: `rate == -1`, `rate == 1`,
  `rate == ±2`).
- `GlucoseTrendDataset`: unit tests against a small synthetic DataFrame —
  verify it respects `(id, id_segment)` boundaries, correct horizon
  alignment, correct sample count for a known small input.
- `TrendCNN`: forward-pass output shape test, param-count budget test.
- Full pipeline: integration smoke test training a few epochs on iglu (fast,
  small) before committing to the full weinstock run (200 patients, slower).

### Repo hygiene

`GlucoEdge/` is now its own git repo (it had none). `GlucoBench/` (with its
own nested `.git`) is gitignored wholesale — treated as an external
dependency fetched by the setup instructions, never committed into
GlucoEdge's history. Raw CSVs, model checkpoints, and generated result files
are also gitignored as regenerable artifacts.

## Out of scope for this spec

- LiteRT conversion and INT8 quantization (original plan step 2)
- Android app scaffolding (original plan step 3)
- README / portfolio write-up (original plan step 4)
- The optional on-device LLM explanation stretch goal

Each gets brainstormed and specced separately once this pipeline is
implemented, verified on weinstock, and the class-count question is settled
with real numbers.
