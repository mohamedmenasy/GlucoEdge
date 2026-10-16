# Model card: GlucoEdge bundled trend classifier

Structured after Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019).
GlucoEdge is a research prototype. **This model is not a medical device and must
not inform any diagnosis, monitoring, treatment, or dosing decision.**

## Model details

- **Files** (shared by the Android and iOS apps):
  - `android/app/src/main/assets/trend_float.tflite`: SHA-256
    `bfb37468c5896be9f8c30f515c7499586a71da1c0ca0f77319080f8fcaa76153`, 208,560 bytes.
  - `android/app/src/main/assets/trend_int8.tflite`: SHA-256
    `eacc0a741403b901330f0dfcc529195779b29f7bb0ed2de9ae90da8912dcf554`, 61,936 bytes.
- **Architecture:** one-dimensional convolutional neural network (CNN) with
  (96, 144) channels and a flatten head over all 12 positions; 50,645 parameters.
- **Training recipe:** class-weighted cross-entropy, cosine learning-rate decay over
  20 epochs, kept epoch 11 (chosen by validation directional recall under a
  stable-recall floor), seed 4, batch size 512.
- **Quantization:** static post-training 8-bit integer (INT8) quantization with
  LiteRT Torch (PyTorch 2 Export), calibrated on
  200 windows drawn uniformly at random (`uniform_random_200_d101`) from the
  27,109 eligible Weinstock validation windows. Both apps default to the float model.
- **Provenance:** seed-4 artifact of accuracy-sweep round five, bundled on
  2026-10-15. Protocol `paper/results/calibration-coverage/protocol_sweep_r5.json`
  (frozen at commit `f9010b6`); record `accuracy_sweep_r5.json`; history of earlier
  bundled pairs in `android/app/src/main/assets/MODELS.md`; method in `paper/main.tex`.

## Intended use

Offline research replay of synthetic or historical continuous glucose monitoring
(CGM) traces in the GlucoEdge
apps, to study on-device conversion, quantization, and class-wise evaluation.
Intended users are machine-learning and mobile-systems researchers.

## Out-of-scope uses

- Any clinical use: live decision support, alerts or alarms, insulin or
  carbohydrate dosing, diagnosis, or monitoring.
- Connection to a live sensor stream, or display to people with diabetes as a
  trend arrow. The class names resemble CGM trend-arrow categories, but the model
  reproduces no manufacturer's algorithm and has no clinical validation.
- Commercial use of any kind (see Licence).

## Inputs and output

- **Input:** 12 consecutive CGM readings in mg/dL at a 5-minute cadence, most recent
  last. Both apps restart the window when more than 5 minutes separate two readings,
  so no window spans a sensor gap. mmol/L inputs, other cadences, and gap-filled
  values are not supported.
- **Encoding:** the clients pass 11 consecutive differences plus the last reading
  divided by 16 (`DeltaEncoding` in both apps); the INT8 model quantizes them with
  the scale and zero point stored in the file.
- **Output:** five logits for `falling_fast`, `falling`, `stable`, `rising`,
  `rising_fast`. The classes are bands of the mean rate of change over the next
  15 minutes, (g[t+15] - g[t]) / 15, cut at -2, -1, +1 and +2 mg/dL/min.

## Training data

- Five DiaData v3 cohorts (raw 5-minute archive), 898 subjects with type 1 diabetes:
  CITY (153), DCLP3 (DiaData label DLCP3; 169), REPLACE-BG (226), SENCE (144),
  WISDM (206) (`paper/results/diadata_audit_r1.json`). 15,135,091 training windows
  after whole-subject sampling (`accuracy_sweep_r3.json`, `per_arm_train_windows`).
- No Weinstock or other GlucoBench window was used for training. Checkpoint and seed
  selection used eligible Weinstock validation windows, which also supplied the
  INT8 calibration set.

## Evaluation

Metrics: directional recall, the unweighted mean recall (sensitivity) of the four
non-stable classes; stable recall; macro-F1. Intervals are paired
participant-cluster bootstrap 95% intervals (2,000 draws), in recall points.
"Previous" is the previously bundled INT8 model on the same windows.

| Data | Role | Windows | Directional, INT8 (previous) | Stable (previous) | Macro-F1 (previous) |
|---|---|---|---|---|---|
| RT-CGM (DiaData; JDRF CGM randomized trial) | Pre-registered zero-shot primary | 12,674,509 | 0.555 (0.553); +0.20 [+0.11, +0.29] | 0.605 (0.607) | 0.410 (0.427) |
| Weinstock held-out (20 participants) | Descriptive; labels read in earlier rounds | 46,376 | 0.520 (0.523); -0.3 [-0.8, +0.1] | 0.436 (0.459) | 0.342 (0.364) |
| Weinstock temporal | Descriptive; labels read in earlier rounds | 37,361 | 0.505 (0.502); +0.3 [-0.6, +1.2] | 0.499 (0.520) | 0.335 (0.356) |
| PEDAP (never trained on) | Descriptive zero-shot probe | 7,003,788 | 0.539 (0.541) | 0.391 (0.394) | n/a |

Sources: `paper/results/calibration-coverage/accuracy_sweep_r5.json` (components,
`pedap_zero_shot`). PEDAP per-class recall, INT8 (previous): `falling_fast` 0.705
(0.638), `falling` 0.396 (0.460), `rising` 0.423 (0.453), `rising_fast` 0.633
(0.611).

The RT-CGM gain compares one seed-4 artifact with one champion artifact. It is
smaller than the 1.1-point five-seed held-out spread seen in round three
(`accuracy_sweep_r3.json`), and the five-seed RT-CGM comparison of the two recipes
was not run. On both Weinstock components the model would fail the
beat-the-champion and stable-floor gates. Golden vectors pin float logits to within
1e-5 of Python and INT8 outputs bit-exactly in the Android unit tests, the iOS
simulator, and on an iPhone 18 Pro Max (`MODELS.md`).

## Known failure modes

From the confusion matrices in `panel_revision_r1.json` (`shipped_model`):

- **Movement labels during stable glucose.** Stable recall is 0.436 on held-out
  windows (0.605 on RT-CGM), and about 63% of the movement labels the model emits on
  either set fall in windows whose next-15-minute rate stays within ±1 mg/dL/min.
  Positive predictive value of the four movement classes is 0.14-0.25 on held-out
  windows and 0.21-0.27 on RT-CGM.
- **Over-calling the fast classes.** 36% of truly `falling` held-out windows are
  labelled `falling_fast` (27% on RT-CGM), and 28% of truly `rising` windows are
  labelled `rising_fast` (19%).
- **Direction reversals.** 13.9% of truly falling held-out windows (either falling
  class) are labelled `rising` or `rising_fast`, and 15.1% of truly rising windows
  are labelled `falling` or `falling_fast` (8.6% and 8.8% on RT-CGM).
- **Low glucose.** When the last reading is below 70 mg/dL, `falling` recall is
  0.27 on held-out windows and 0.45 on RT-CGM, against 0.45 and 0.56 at
  70-180 mg/dL, and the model recognizes 2 of the 96 RT-CGM `falling_fast` windows
  in that band (held-out has one).
- **Sensor limit values.** Readings of 39 and 401 mg/dL, consistent with LOW and HIGH
  codes, are treated as glucose values in training, calibration, and evaluation; a
  rate computed from them is only a bound.
- **Labels follow the sensor.** Labels are computed on the CGM trace, which lags
  blood glucose; label robustness was checked only against independent noise of
  1-2 mg/dL.

## Populations and conditions not represented

- No type 2 diabetes, pregnancy, inpatient, or critical-care data; every training and
  evaluation cohort of this model has type 1 diabetes.
- Training sensors are Dexcom G4 (505 software), G5, and G6 only
  (`diadata_audit_r1.json`); other manufacturers appear only in the RT-CGM
  evaluation, from first-generation devices.
- Training cohorts span children aged 2 to under 8 (SENCE) to adults aged 60 and
  over (WISDM); preschool children on automated insulin delivery (PEDAP) were held
  out of training.
- Wear modes are mixed: masked run-in and real-time wear, which DiaData does not
  separate, plus closed-loop arms (DCLP3). How wearers would react to a displayed
  trend was never studied.
- Sex, race and ethnicity, socioeconomic status, and country were not analysed.

## Licence and contact

- The weights derive from DiaData v3 (doi:10.5281/zenodo.17285631), released under
  CC BY-NC 4.0, and are released for **non-commercial research use only**. Some
  DiaData cohorts originate from the Barbara Davis Center, Jaeb Center for Health
  Research, Joslin Diabetes Center, T1D Exchange, University of Colorado, and
  University of Virginia; this model and its evaluation have not been reviewed or
  approved by these institutions.
- Contact: Mohamed Menasy (mohamedmenasy@gmail.com).
