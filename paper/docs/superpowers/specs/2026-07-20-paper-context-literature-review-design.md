# GlucoEdge Paper Context and Literature Review Design

**Date:** 2026-07-20
**Status:** Approved for implementation

## Goal

Strengthen the GlucoEdge manuscript as a standalone general preprint by adding
domain context, a synthesized literature review, and verified references while
preserving the reported experiments and the project's research-only,
non-clinical framing.

## Scope

The revision will expand the six-page manuscript to approximately eight pages.
It will add 12--18 primary scholarly sources, correct existing bibliographic
metadata, and add one compact comparison table. It will not add, rerun, or
reinterpret experiments; change reported values; claim clinical validity; or
present the prototype as a live CGM integration.

## Manuscript Changes

### Introduction

Add concise context on CGM sampling and short-horizon forecasting. Distinguish
among glucose-value regression, glycemic-event prediction, and trend
classification. Explicitly distinguish commercial arrows, which summarize
recent or current glucose movement using device-specific definitions, from
GlucoEdge's future-derived 15-minute rate class.

Replace unsupported general statements about server-side deployment with the
narrower observation that published evaluations largely emphasize offline
predictive performance. Position the contribution as a reproducible offline
replay path from public data through conversion and physical-device execution.

### Related Work

Reorganize the section into four connected themes:

1. CGM forecasting tasks, horizons, and input modalities.
2. Benchmarking, temporal and participant splits, and cross-dataset
   generalization.
3. Trend-arrow semantics and imbalanced trend classification.
4. Compact edge inference, post-training quantization, calibration data, and
   hardware-dependent measurement.

The prose will compare study designs rather than compare headline metrics that
were produced under incompatible datasets and protocols. A table of roughly
eight representative studies will summarize task, horizon, inputs/data,
evaluation scope, and whether physical-device execution was reported.

### Data and Methods

Add source-cohort context and explain why it limits population generalization.
Clarify that interpolation and stride-one windowing create dependent examples.
Describe the 60-minute history and 15-minute horizon as fixed engineering
choices. Explain why per-class recall and macro recall are more informative
than accuracy under the observed class imbalance.

### Experimental Protocol and Results

Disclose the aggregate test-frame composition and known context-prefix issue
before interpreting results. Preserve every numerical value. Describe INT8
saturation as a plausible contributor to degradation rather than a proven sole
cause, and keep latency conclusions limited to the observed device run.

### Discussion and Limitations

Connect the findings to work on calibration-set coverage, post-training
quantization, and model--runtime--hardware benchmarking. Replace unsupported
generalizations with case-study language. Expand limitations to cover CGM-only
inputs, cohort representativeness, dependent windows, interpolated samples,
single-seed uncertainty, absent simple predictive baselines, absent probability
calibration, and the lack of external or prospective validation.

## Reference Plan

Add verified primary sources spanning:

- recurrent and convolutional-recurrent CGM forecasting;
- cross-dataset generalization;
- time-series evaluation, leakage, and benchmark variance;
- mobile inference and real-device benchmarking; and
- post-training quantization, rounding, clipping, and small calibration sets.

Correct the missing author in the existing Mirshekarian entry, rename the
misleading Liu systematic-review citation key, and update official LiteRT
documentation metadata. Implementation-specific GlucoBench and LiteRT claims
will use authoritative versioned sources where a stable citable record is
available.

## Quality Checks

The completed revision must:

- compile with `tectonic --keep-intermediates --keep-logs main.tex`;
- contain no missing or unused citation keys and no unresolved references;
- retain all reported measurements exactly unless correcting an independently
  verified transcription error;
- use only verified bibliographic metadata;
- preserve the explicit research-only and non-treatment statements; and
- render without clipped text, broken tables, overlapping elements, or
  unreadable references.

## Deliberate Exclusions

No new model training, corrected split evaluation, additional device benchmark,
clinical interpretation, systematic-review protocol, or meta-analysis is part
of this revision. Those require new evidence rather than stronger prose.
