# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GlucoEdge is a portfolio project: an on-device glucose trend classifier for
Android, built entirely on the public [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench)
CGM research benchmark. **Not a medical device** — no proprietary data,
schemas, or algorithms belong in this repo, and no code, comment, or output
string should imply clinical validity or treatment guidance.

## Commands

```bash
# One-time setup (GlucoBench is an external dependency, cloned as a
# sibling directory, never vendored - see .gitignore)
git clone https://github.com/IrinaStatsLab/GlucoBench.git
cd GlucoBench && unzip raw_data.zip && cd ..
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Tests (fast, no CGM data needed - GlucoBench doesn't need to be cloned
# for this; every module lazily imports it only where actually used)
pytest -v
pytest tests/test_labeling.py -v            # single file
pytest tests/test_labeling.py::test_five_classes_order -v  # single test

# Actual training runs (need GlucoBench cloned + unzipped per above)
python -m training.train --dataset iglu --classes 5 --epochs 2        # smoke test, seconds
python -m training.train --dataset weinstock --classes 5 --epochs 20  # full run, several minutes
```

CI (`.github/workflows/tests.yml`) runs only `pytest` on every PR/push to
`main` — it deliberately does not clone GlucoBench or run the training CLI,
since none of the unit tests need real CGM data and the real training runs
are too slow to gate every PR.

## Architecture

```
training/
├── labeling.py   # label_trend(): rate-of-change -> 1 of 5 trend classes
├── dataset.py    # GlucoseTrendDataset: segment-aware sliding windows
├── model.py      # TrendCNN: small 1D-CNN (~2,900 params)
└── train.py      # CLI: wires GlucoBench's DataFormatter into the above
```

**The one non-obvious correctness constraint in this codebase:** GlucoBench's
`DataFormatter` tags every contiguous gap-free run of CGM readings (after its
own interpolation step) with an `id_segment` column. `GlucoseTrendDataset`
groups by `(id, id_segment)`, not `id` alone — windows must never slide
across a segment boundary, since that would silently splice together time
periods separated by a dropped sensor gap. Any change to the windowing logic
must preserve this.

**Label thresholds are clinically-anchored, not curve-fit to the data.**
`label_trend()`'s 5 classes come from real CGM trend-arrow conventions
(rate in mg/dL/min: `stable` < 1, `falling`/`rising` 1-2, `falling_fast`/
`rising_fast` ≥ 2), checked against real class counts rather than tuned
until classes look balanced. See
`docs/superpowers/specs/2026-06-30-training-pipeline-rebuild-design.md` for
the full reasoning and `docs/superpowers/specs/2026-07-01-weinstock-class-count-decision.md`
for why the classifier stays 5-class rather than collapsing to 3 (both
"fast" classes cleared the collapse thresholds by a wide margin on the full
weinstock dataset — this is settled with real numbers, not assumed).

`GlucoBench/` is a clone of an external repo (own git history, own
license-free-but-cite-on-reuse terms) — read from, never modified, never
committed into this repo's history (`.gitignore`'d without a trailing slash,
since it's a symlink rather than a real directory in git worktrees).

**No model checkpoint or fixed random seed is persisted by `train.py`** — its
purpose so far has been settling the 5-vs-3-class question, not producing a
deployable artifact. The next phase (LiteRT conversion) needs to retrain
with a fixed seed first.

## Docs

- `docs/superpowers/specs/` — design specs and decision records, one file
  per dated decision
- `docs/superpowers/plans/` — implementation plans
- `results/*.json` — per-run classification reports (gitignored, regenerated
  by running `train.py`)
