# README Quantization Tradeoff Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the measured quantization tradeoff section and a consolidated Disclaimer section to README.md, completing the final numbered step of the project brief.

**Architecture:** Docs-only edit to one file. Every number is copied verbatim from two committed sources: `docs/superpowers/specs/2026-07-01-litert-conversion-results.md` and the 2026-07-07 device addendum in `docs/superpowers/specs/2026-07-05-emulator-compiledmodel-sigill.md`. No new measurements, no code.

**Tech Stack:** Markdown.

## Global Constraints

- No string may imply clinical validity or treatment guidance; the app footer is quoted verbatim: `Portfolio demo on public research data — not a medical device, not treatment guidance.`
- Every number must match its committed source exactly (no re-rounding, no transposition).
- README sections must not contradict each other (a prior review caught exactly this failure mode).
- Only README.md changes.

---

### Task 1: Add tradeoff section, Disclaimer section, and bookkeeping edits

**Files:**
- Modify: `README.md` (three edits: new section after the Android app section ~line 119; new Disclaimer section before "Data attribution" ~line 150; strike Roadmap item 4 ~line 136; trim the "Not yet built" list ~line 26)

**Interfaces:**
- Consumes: numbers from the two committed source docs named above.
- Produces: nothing downstream — terminal docs task.

- [ ] **Step 1: Insert the tradeoff section**

Directly after the line `emulator are labeled as such and are not device measurements.` and before `## Roadmap`, insert:

```markdown
## The quantization tradeoff, measured

The headline comparison across the three model artifacts, on the full
127,165-window weinstock test set (accuracy/recall) and a real phone
(latency):

| | Size | Latency mean / p95 (device) | Accuracy | Macro-avg recall |
|---|---|---|---|---|
| PyTorch checkpoint | 14.4 KB | — (not deployable) | 0.5184 | 0.5060 |
| Float `.tflite` (shipped default) | 16.95 KB | 0.334 ms / 0.540 ms | 0.5184 | 0.5060 |
| INT8 `.tflite` | 11.70 KB | 0.484 ms / 0.665 ms | 0.5866 | 0.4135 |

Latency is measured in-app on a Samsung Galaxy S22 Ultra (Android 16) via
the CompiledModel path — the full Kotlin classify path (buffer writes,
quantization, softmax), not just kernel invoke; float over a full
100-inference rolling window, INT8 over a 21-inference window. The
conversion phase's dev-machine CPU proxy reached the same qualitative
conclusion: INT8 is not faster at this model size — per-call overhead
dominates a ~2,900-parameter network.

Float conversion is exactly lossless: its classification report is
byte-identical to the PyTorch checkpoint's. INT8's *higher* accuracy is
not an improvement — it's majority-class bias, visible immediately in
per-class recall:

| Class | Support (% of test) | Float recall | INT8 recall | Change |
|---|---|---|---|---|
| `falling_fast` | 3,692 (2.9%) | 0.5119 | 0.1907 | −0.3212 (−62.7% relative) |
| `falling` | 11,387 (9.0%) | 0.5484 | 0.4306 | −0.1178 |
| `stable` | 95,998 (75.5%) | 0.5225 | 0.6529 | +0.1304 |
| `rising` | 11,012 (8.7%) | 0.4761 | 0.3845 | −0.0916 |
| `rising_fast` | 5,076 (4.0%) | 0.4712 | 0.4090 | −0.0622 |

`stable` — 75.5% of the test set — is the only class INT8 helps;
`falling_fast`, the rarest class and half the reason a trend classifier
exists, loses 62.7% of its recall. (A degenerate model that always
predicts `stable` scores ~75.5% accuracy with a macro-avg recall of
exactly 0.20 — which is why this project never reports accuracy alone.)
Part of the INT8 degradation is calibration coverage: static quantization
calibrated on 200 validation windows learned an input ceiling of
~240 mg/dL while the test data reaches 401 mg/dL, so ~28% of test windows
saturate at the INT8 input boundary even with correct clipping.

So the measured tradeoff is: INT8 buys a ~31% smaller file (5.25 KB) and
costs 9.25 macro-recall points, with no latency benefit — which is why
the app ships float as its default and offers INT8 as a toggle. Full
analysis:
[`docs/superpowers/specs/2026-07-01-litert-conversion-results.md`](docs/superpowers/specs/2026-07-01-litert-conversion-results.md).
```

- [ ] **Step 2: Insert the Disclaimer section**

Directly before `## Data attribution`, insert:

```markdown
## Disclaimer

GlucoEdge is a portfolio demonstration of on-device ML engineering. It is
**not a medical device**: it does not diagnose, monitor, or advise on
diabetes management, and no output may inform a treatment decision. No
reported number has been clinically validated — accuracy and recall
figures describe performance on a public research benchmark, nothing
more. The app states this on-screen, verbatim: *"Portfolio demo on public
research data — not a medical device, not treatment guidance."* All data
comes from the public GlucoBench research benchmark; no proprietary data,
schemas, or algorithms are used anywhere in the project.
```

- [ ] **Step 3: Bookkeeping edits**

Replace Roadmap item 4:
```markdown
4. ~~Expand this README with the quantization tradeoff table and finalize
   the disclaimer for the shipped app~~ — done, see
   [The quantization tradeoff, measured](#the-quantization-tradeoff-measured)
   and [Disclaimer](#disclaimer) above.
```

In "What this is, right now", replace
```markdown
below, with the
`CompiledModel` path verified on a physical device). Not yet built:
this README's own quantization tradeoff table and the on-device
explanation stretch goal — see [Roadmap](#roadmap).
```
with
```markdown
below, with the
`CompiledModel` path verified on a physical device). Not yet built: the
on-device explanation stretch goal — see [Roadmap](#roadmap).
```

- [ ] **Step 4: Verify every number against its committed source**

Run and compare by eye against the two tables just written:
```bash
grep -n "0.5184\|0.5866\|0.5060\|0.4135\|16.95\|11.70\|0.1907\|0.6529\|0.5119\|0.5484\|0.5225\|0.4761\|0.4712\|0.4306\|0.3845\|0.4090" docs/superpowers/specs/2026-07-01-litert-conversion-results.md
grep -n "0.334\|0.540\|0.484\|0.665" docs/superpowers/specs/2026-07-05-emulator-compiledmodel-sigill.md
```
Expected: every value in the README's two tables appears in its source doc. Also confirm supports (3,692 / 11,387 / 95,998 / 11,012 / 5,076), percentages, and change column against the results doc's per-class table, and 14.4 KB against `results/weinstock_5class_model.pt`'s documented size (14,793 bytes ≈ 14.4 KB, as stated in the paper and MODELS.md context).

- [ ] **Step 5: Full read-through for self-consistency**

Read README.md top to bottom once. Checklist: no section claims something another contradicts; the anchor links `#the-quantization-tradeoff-measured` and `#disclaimer` match the actual GFM slugs of the new headings; no clinical-validity language anywhere; the top-of-README disclaimer paragraph is unchanged.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Add measured quantization tradeoff section and consolidated disclaimer to README"
```

---

## Self-Review Notes

Spec coverage: headline table (Step 1), per-class table + punchline + calibration sentence + link (Step 1), Disclaimer section (Step 2), Roadmap strike + Not-yet-built trim (Step 3), number verification + read-through (Steps 4-5). No placeholders; single-file task; no type surface.
