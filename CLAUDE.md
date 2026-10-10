# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GlucoEdge is a portfolio project: an on-device glucose trend classifier for
Android and iOS, built entirely on the public [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench)
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

# Android app (android/)
cd android && ./gradlew :app:assembleDebug :app:testDebugUnitTest

# iOS app (ios/ — macOS, Xcode 26.6+, git-lfs installed). The LFS env
# override is required for package resolution: LiteRT keeps unused
# Android/Linux prebuilts in Git LFS that SwiftPM can't pull for a pinned
# commit.
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=lfs.fetchexclude GIT_CONFIG_VALUE_0='*' \
  xcodebuild test -project ios/GlucoEdge.xcodeproj -scheme GlucoEdge \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

CI (`.github/workflows/tests.yml`) runs three jobs on every PR/push to
`main`: `pytest`, the Android build + JVM unit tests + manifest/native-lib
guards, and the iOS simulator suite (golden parity included) + a
no-network source scan. It deliberately does not clone GlucoBench or run
the training CLI — no unit test needs real CGM data, and real training
runs are too slow to gate every PR.

## Architecture

```
training/
├── labeling.py   # label_trend(): rate-of-change -> 1 of 5 trend classes
├── dataset.py    # GlucoseTrendDataset: segment-aware sliding windows
├── model.py      # TrendCNN: small 1D-CNN (~2,900 params)
└── train.py      # CLI: wires GlucoBench's DataFormatter into the above
conversion/       # LiteRT float + INT8 export, benchmark, golden vectors
experiments/      # split/calibration provenance audit (paper)
android/          # Kotlin/Compose app; owns the shared .tflite assets and
                  #   golden_vectors.json under app/src/{main,androidTest}/assets
ios/              # SwiftUI app; bundles android's asset folders IN PLACE via
                  #   Xcode folder references (moving them breaks the iOS
                  #   build on purpose). LiteRT Swift CompiledModel, pinned to
                  #   commit 8f555ada (v2.3.0 release binaries; the v2.3.0 tag's
                  #   manifest is broken). Explain = Apple Foundation Models.
paper/            # manuscript (tectonic)
```

Both apps reuse training's window rule: a replay window resets when the gap
since the previous reading exceeds 5 minutes, mirroring `id_segment`.

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

**Reproducible checkpoints:** `train.py --seed 0 --save-checkpoint` writes
`results/weinstock_5class_model.pt` (gitignored); `conversion/` turns it
into the bundled `.tflite` files (deps pinned separately in
`conversion/requirements.txt`). The committed model assets, their sha256s
and the golden vectors must be regenerated together — both apps' parity
tests pin all three.

## Docs

- `docs/superpowers/specs/` — design specs and decision records, one file
  per dated decision
- `docs/superpowers/plans/` — implementation plans
- `results/*.json` — per-run classification reports (gitignored, regenerated
  by running `train.py`)
