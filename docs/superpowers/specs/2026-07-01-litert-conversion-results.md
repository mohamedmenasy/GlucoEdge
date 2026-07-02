# LiteRT conversion and INT8 quantization results

Date: 2026-07-01

Retrained weinstock/5-class with a fixed seed (`--seed 0`) and converted
the checkpoint to LiteRT, then post-training quantized to static INT8
(PT2E, calibrated on 200 real windows from the validation split). Full
numbers: `results/conversion_report.json`.

## Size

- Float `.tflite`: 16.95 KB
- INT8 `.tflite`: 11.70 KB

INT8 is 5.25 KB smaller than float — a ~31.0% size reduction. This is the
one clearly genuine, unambiguous win quantization delivers here.

## Latency (this development machine's CPU, not an Android device)

|  | mean (µs) | median (µs) | p95 (µs) |
|---|---|---|---|
| Float `.tflite` | 2.4527 | 2.3340 | 2.4979 |
| INT8 `.tflite` | 2.7953 | 2.7500 | 2.8339 |

(Raw measurements are in `results/conversion_report.json`'s `latency_ms`
field, e.g. float mean = 0.0024527 ms; shown here in microseconds since
every value is sub-millisecond.)

There is essentially no latency difference between float and INT8 — both
run in the 2.3-2.8 microsecond range per inference, and INT8 is if
anything marginally *slower* (by roughly 0.3-0.4 µs across mean/median/
p95) rather than faster. This model is tiny (~2,900 params), so fixed
per-call interpreter overhead dominates; there isn't enough compute in a
single inference for INT8 arithmetic to show a measurable saving. Real
on-device Android latency will be measured once the app exists (next
sub-project) — these are a CPU proxy, not a phone measurement, and given
how small the differences are relative to likely measurement noise, no
latency conclusion should be drawn beyond "the two are indistinguishable
on this proxy."

## Accuracy (same weinstock test set as the original class-count decision)

|  | accuracy | macro-avg recall |
|---|---|---|
| Original PyTorch checkpoint | 0.5184 | 0.5060 |
| Float `.tflite` | 0.5184 | 0.5060 |
| INT8 `.tflite` | 0.5552 | 0.3983 |

Float conversion is lossless: its accuracy and macro-avg recall match the
PyTorch checkpoint exactly, class-by-class (format conversion only, no
precision loss).

INT8 quantization's accuracy is *higher* than the float/PyTorch baseline
(0.5184 -> 0.5552, +3.68 points) but its macro-avg recall is *lower*
(0.5060 -> 0.3983, -10.77 points). Reporting only the accuracy number here
would be actively misleading, and would repeat the exact mistake this
project has already flagged and avoided once before (the original iglu
smoke test's whole reason for existing: always predicting the majority
class beats a real model on raw accuracy while being useless). `stable` is
95,998 of 127,165 test windows (75.5% of the test set) — heavily
imbalanced. Per-class recall shows the mechanism directly: INT8 only
improves recall on the majority class; every other class gets worse.

| class | support (% of test set) | PyTorch/float recall | INT8 recall | change |
|---|---|---|---|---|
| falling_fast | 3,692 (2.9%) | 0.5119 | 0.1324 | -0.3795 (-74.1% relative) |
| falling | 11,387 (9.0%) | 0.5484 | 0.3919 | -0.1565 (-28.5% relative) |
| stable | 95,998 (75.5%) | 0.5225 | 0.6133 | +0.0909 (+17.4% relative) |
| rising | 11,012 (8.7%) | 0.4761 | 0.4108 | -0.0653 (-13.7% relative) |
| rising_fast | 5,076 (4.0%) | 0.4712 | 0.4431 | -0.0282 (-6.0% relative) |

INT8 quantization biased predictions toward the majority `stable` class:
`stable`'s recall rose while all four minority-class recalls fell,
`falling_fast` (the smallest class, 2.9% of the test set) hit hardest with
a 74.1% relative drop. The accuracy improvement is an artifact of that
majority-class bias, not a genuine gain — on an imbalanced test set like
this one, predicting the majority class more often mechanically raises
accuracy while making the classifier worse at catching the minority trend
classes (`falling_fast` / `rising_fast`) it exists to catch.

## Net assessment

INT8 quantization's real, measured effect on this model is: a genuine
~31% smaller file (5.25 KB saved), no measurable latency benefit on this
CPU proxy (the model is too small for compute savings to show up over
fixed interpreter-call overhead), and a real macro-avg-recall cost
(-10.77 points, driven by majority-class bias) that the top-line accuracy
number alone hides rather than reveals. Whether that size-for-recall
tradeoff is worth taking is a decision for the next sub-project (the
Android app) to make explicitly against real constraints, rather than
defaulting to `model_int8.tflite` on the strength of its accuracy number.
