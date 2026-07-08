# README quantization tradeoff table + finalized disclaimer

Date: 2026-07-07
Status: approved design

## Purpose

Final numbered step (4) of the original project brief: expand the README
with the measured size/latency/accuracy tradeoff table and finalize the
disclaimer for the shipped app. Docs-only — no code, no new measurements.
Every number comes from two committed sources:

- `docs/superpowers/specs/2026-07-01-litert-conversion-results.md`
  (sizes, accuracy, macro recall, per-class recalls, calibration coverage)
- `docs/superpowers/specs/2026-07-05-emulator-compiledmodel-sigill.md`
  (2026-07-07 device addendum: Galaxy S22 Ultra latency via CompiledModel)

## Decisions (settled with the user)

1. **Depth: summary table + per-class recall table.** The per-class view
   is the project's signature "accuracy misleads" argument and must be
   on-page, not just linked.
2. **Placement: Approach A** — a dedicated section immediately after the
   "Android app" section, since the table draws on both the conversion
   phase and the app's device measurements.

## Changes to README.md

### 1. New section `## The quantization tradeoff, measured`

Placed directly after the "Android app" section, before "Roadmap".

**Headline table.** Rows: PyTorch checkpoint, float `.tflite`, INT8
`.tflite`. Columns: size, on-device latency mean/p95, accuracy, macro-avg
recall. Values (verbatim from the committed sources):

| | Size | Latency mean / p95 (device) | Accuracy | Macro-avg recall |
|---|---|---|---|---|
| PyTorch checkpoint | 14.4 KB | — (not deployable) | 0.5184 | 0.5060 |
| Float `.tflite` (shipped default) | 16.95 KB | 0.334 ms / 0.540 ms | 0.5184 | 0.5060 |
| INT8 `.tflite` | 11.70 KB | 0.484 ms / 0.665 ms | 0.5866 | 0.4135 |

Latency caption: measured in-app on a Samsung Galaxy S22 Ultra
(Android 16) via the CompiledModel path — the full Kotlin classify path
(buffer writes + quantization + softmax), not just kernel invoke; the
float figure is a full 100-inference rolling window, INT8 a 21-inference
window. The conversion phase's dev-machine CPU proxy agrees with the
qualitative conclusion: INT8 is not faster at this model size.

**Prose paragraph** (one, short): float conversion is exactly lossless
(classification report byte-identical to PyTorch); INT8's *higher*
accuracy is majority-class bias, not improvement; INT8's only genuine win
is the ~31% smaller file; that is why the app ships float as the default.

**Per-class recall table** (from the results doc, values verbatim):

| Class | Support (% of test) | Float recall | INT8 recall | Change |
|---|---|---|---|---|
| falling_fast | 3,692 (2.9%) | 0.5119 | 0.1907 | −0.3212 (−62.7% relative) |
| falling | 11,387 (9.0%) | 0.5484 | 0.4306 | −0.1178 |
| stable | 95,998 (75.5%) | 0.5225 | 0.6529 | +0.1304 |
| rising | 11,012 (8.7%) | 0.4761 | 0.3845 | −0.0916 |
| rising_fast | 5,076 (4.0%) | 0.4712 | 0.4090 | −0.0622 |

Punchline sentence: `stable` (75.5% of the test set) is the only class
INT8 helps; `falling_fast` — the rarest class, and half the reason a
trend classifier exists — loses 62.7% of its recall. A degenerate
always-`stable` model scores ~75.5% accuracy with 0.20 macro recall,
which is why accuracy alone is never reported here.

**Closing sentences:** one on calibration coverage (200-window static
calibration gives an input ceiling of ~240 mg/dL vs 401 max in test data,
so ~28% of test windows saturate at the INT8 input — a real cost of
narrow static calibration), plus the link to the full results doc.

### 2. New section `## Disclaimer`

Placed directly before "Data attribution". Consolidates, in one place:

- GlucoEdge is a portfolio demonstration of on-device ML engineering.
- It is **not a medical device**: it does not diagnose, monitor, or
  advise on diabetes management, and no output may inform a treatment
  decision.
- No reported number has been clinically validated; accuracy/recall
  figures describe performance on a public research benchmark only.
- The app displays this footer verbatim: "Portfolio demo on public
  research data — not a medical device, not treatment guidance."
- All data is the public GlucoBench research benchmark; no proprietary
  data, schemas, or algorithms are used.

The existing top-of-README disclaimer paragraph stays unchanged (the
hook); this section is the formal version the brief's "finalize the
disclaimer" asks for.

### 3. Bookkeeping edits

- Strike through Roadmap item 4 with a one-line "done" pointer to the new
  section.
- Remove "this README's own quantization tradeoff table" from the
  "Not yet built" list in "What this is, right now" (leaving only the
  on-device explanation stretch goal).

## Verification

- Every number in both tables cross-checked against the two committed
  source docs (no rounding drift, no transposition).
- README read top-to-bottom once for internal consistency (no section may
  contradict another — the failure mode a prior review caught).
- No clinical-validity language introduced anywhere.
- Docs-only: CI unaffected; no test changes.

## Out of scope

The paper branch, the LiteRT-LM stretch goal, any new measurements, any
restructuring of README sections other than the three changes above.
