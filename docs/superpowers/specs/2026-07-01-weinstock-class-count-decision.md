# Weinstock class-count decision

Date: 2026-07-01

Ran 5-class training on the full weinstock config (200 patients, `python -m
training.train --dataset weinstock --classes 5 --epochs 20`). Windowing
produced 424,462 train / 55,289 val / 127,165 test windows (all non-zero, no
data-pipeline problem). `val_loss` decreased from 1.2077 (epoch 1) to a
1.08-1.09 range by epoch 20, with normal epoch-to-epoch noise. Full output:
`results/weinstock_5class_report.json`.

Train window counts by class (`train_label_counts`):
- falling_fast: 13,260
- falling: 40,122
- stable: 314,257
- rising: 39,428
- rising_fast: 17,395

Test-set per-class recall (5-class model, from `classification_report`):
- falling_fast: 0.5333 (support 3,692)
- falling: 0.4978 (support 11,387)
- stable: 0.5735 (support 95,998)
- rising: 0.4196 (support 11,012)
- rising_fast: 0.5380 (support 5,076)

Overall test accuracy: 0.5508. Macro-avg recall: 0.5125.

Decision: **kept 5-class**, because neither `falling_fast` nor `rising_fast`
crosses either collapse trigger from the spec's decision procedure
(`train_label_counts[class] < 200` or test recall `< 0.30`):

- `falling_fast`: 13,260 training-window examples (>> the 200-example
  trigger) and test recall 0.5333 (>> the 0.30-recall trigger, and well
  above the ~20% random-guess baseline for 5 classes).
- `rising_fast`: 17,395 training-window examples (>> the 200-example
  trigger) and test recall 0.5380 (>> the 0.30-recall trigger).

Both "fast" classes clear both triggers by a wide margin (counts are
66-87x the 200-example floor; recalls are ~1.8x the 0.30 floor), so this is
not a borderline call. Per the decision procedure, neither condition fired
for either class, so the weinstock 3-class training (Step 4) was skipped —
`results/weinstock_3class_report.json` was not generated and is not
expected to exist.

GlucoEdge's classifier is **5-class** (`falling_fast` / `falling` /
`stable` / `rising` / `rising_fast`) going forward. This is the label set
that the next sub-project (LiteRT conversion) and the Android app must
match.

Note for that next sub-project: this run's purpose was to settle the class
count, not to produce a deployable model — `train.py` sets no random seed
and saves no checkpoint, so nothing from this run is reusable directly.
LiteRT conversion should start by retraining weinstock/5-class with a fixed
seed and persisting a `state_dict`, then convert that.
