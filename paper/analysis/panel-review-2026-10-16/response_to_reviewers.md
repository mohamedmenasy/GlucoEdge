# Response to Reviewers

- **Reviewed manuscript:** "GlucoEdge: Calibration Coverage and Quantization Trade-offs in On-Device Glucose Trend Classification" (`paper/main.tex` at commit `2439f39`).
- **Revised manuscript:** "GlucoEdge: Quantization-Calibration Coverage and Staged Model Selection in On-Device Glucose Trend Classification" (`paper/main.tex`, branch `docs/paper-panel-revision`; checked against the final text, commit `8136d13` plus the last verification edits). Change markup: `git diff 2439f39 -- paper/main.tex`.
- **Review:** simulated panel `glucoedge-main-tex-round-1`, round 1; decision Major Revision, 2026-10-16. Roadmap and decision letter: `synthesis.md` in this directory.
- **Conventions.** Reviewer points keep the reviewed version's section numbers. "Change and location" uses the revised manuscript's section numbers and `\label` names. Outcomes: *addressed* means the requested change was made; *addressed differently* means the revision responds but departs from the requested remedy, because the record contradicts its premise or a stated constraint applies (evidence in Part 6); *not addressed* means no change. Numbers quoted from the manuscript were checked against the revised source; numbers taken from records name the record.

**Location key (revised manuscript).** §1 Introduction; §2.2 Benchmarks; §2.3 Trend representations; §2.4 Edge inference and quantization (`sec:ptqrelated`); §3 Problem Formulation; §4.1 Dataset; §4.2 (`sec:eligible`); §4.3 Model and training; §5.1 Conversion and quantizer; §5.2 (`sec:design`); §5.3 Client implementations; §6.1 Calibration study protocol (`sec:calibprotocol`); §6.2 Accuracy sweep protocol (`sec:sweepprotocol`); §7.1 (`sec:provenance`); §7.2 Coverage manipulation; §7.3 (`sec:maineffects`); §7.4 (`sec:mechanism`); §7.5 Baselines; §7.6 (`sec:sensitivity`); §7.7 Two further cohorts; §7.8 (`sec:deltainputs`); §7.9 (`sec:deploycost`); §8 Staged accuracy sweeps (`sec:sweepresults`), §8.1–§8.5 rounds one to five (`sec:sweep1`, `sec:sweep5`); §9.1 Input-range coverage, not sample count (`sec:calibevidence`); §9.2 Scope of the accuracy illusion; §9.3 What the sweeps show (`sec:sweepdiscussion`); §10 Limitations, items 1–10 (`sec:limitations`); §11.1–§11.2 (`sec:repro`); §12 Conclusion; App. A Provenance and revision history (`app:history`); App. B Cohorts, model variants, read ledger (`app:cohorts`); App. C Calibration-study details, C.1–C.7 (`app:calibdetails`); App. D Sweep details, D.1–D.4 (`app:sweepdetails`).

**Reviewed to revised sections.** §6 → §6.1 (new §6.2); §7.1–§7.8 keep their numbers (§7.7 retitled "Two further cohorts"); §7.9–§7.13 → §8.1–§8.5; §7.14 → App. A.2; §7.15 → §7.9, with device observations in App. A.3 and the benchmark in App. C.6; §8.1–§8.2 → §9.1–§9.2 (new §9.3); §9 → §10; §10 → §11; §11 → §12.

## Part 1. Summary of the decision and the revision

The panel's decision was Major Revision, from repairable blocks on four mandatory dimensions (methodology, domain accuracy, argumentative coherence, contribution). No seat disputed the calibration core: the 13.4-point contrast, the clip-only mechanism and the free deployment. The revision keeps that core and rescopes the claims built on it:
- two research questions, with a new title, a sweep protocol (§6.2) and a sweep discussion (§9.3);
- novelty stated as a measurement of a known lever, positioned against the PTQ calibration-data literature and the converter guidance quoted verbatim; "toolchain-default" removed;
- null contrasts scoped to the saturated input-range regime, and the take-home stated as input-range coverage;
- second-cohort predictions reported as not confirmed, with exact sign-flip tests, class support, undefined-draw fractions and clipped-window shares;
- round five restated as an artifact-level ship decision, with its costs and round three's seed spread beside the +0.20-point gain;
- round four renamed a label-definition stability check, and the label-noise conclusion limited to the noise model tested;
- new cohort, model-variant, read-ledger, protocol-to-report and error-profile tables, a model card, re-analyses from cached predictions (`panel_revision_r1.json`) and hashes of 190 artifacts.

Outcomes: required items, 20 addressed and 1 addressed differently (R3, in part); suggested items, 34 addressed and 5 addressed differently (S3, S10, S17, S21, S34); none unaddressed.

## Part 2. Author decisions

These decisions bind several items below.

1. **Two research questions, sweeps in the body (R1).** The paper asks how quantization-calibration coverage affects directional recall and deployment cost, and which model and data changes improve directional recall when each selection round may spend at most one pre-registered read. The title, abstract, contribution list, protocol (§6.1, §6.2), discussion (§9.1–§9.3) and conclusion follow that structure.
2. **No new inference.** Re-analyses use only cached predictions, saved artifacts and committed records; no model was trained, converted or run. R4 therefore takes the artifact-level route (the five-seed RT-CGM comparison was not run) and R20 the scoping route (no range-aware selection test on Dubosson or Hall). The optional colored-noise rerun (R19) and fixed-input-range arm (S25) were not run either.
3. **Weights for non-commercial research use (S11).** The bundled weights were trained on DiaData cohorts released under CC BY-NC 4.0 and are released for non-commercial research use only (§1, §11.2, `MODEL_CARD.md`).
4. **Page budget.** The body (Introduction through Conclusion) was kept to 18 pages by moving detail tables to Appendices C and D. The abstract stays under arXiv's 1,920-character limit at 1,915 characters.

## Part 3. Required revisions (R1–R21)

### R1. Two studies, no single takeaway (EIC-W1, major; DA-O1). *Addressed.*
**Point.** The calibration study and five accuracy sweeps share one manuscript; the sweeps' methods sit only in Results, and no single takeaway emerges.

**Response.** Agreed. Of the two routes the EIC offered, the revision takes the second: two research questions stated up front, with matching structure (author decision 1). The first route, a calibration-only paper with the sweeps in an appendix, was not taken.

**Change and location.** New title. The abstract and §1 state both questions. The sweeps get their own protocol (§6.2, `sec:sweepprotocol`), results (§8) and discussion (§9.3, `sec:sweepdiscussion`), and §12 answers each question in its own paragraph. Client-engineering detail moved to App. C.7 (emulator workaround) and App. A.3 (earlier rounding rule).

### R2. Novelty against the PTQ calibration-data literature (EIC-W2, R2-W2, major; DA counter-argument). *Addressed.*
**Point.** §2.4 and §8.1 did not engage work in which calibration-data choice is itself the studied lever, so the main observation read as new.

**Response.** Agreed. The revision adopts the framing implied by the DA's counter-argument: the lever is known, and the contribution is its measurement in this setting. That measurement covers how contiguous first-N selection of overlapping, time-ordered windows departs from the guidance, and the class-level consequence (`falling_fast` recall collapses while accuracy rises). It uses participant-clustered intervals and a clip-only control.

**Change and location.** §2.4 (`sec:ptqrelated`) has a new paragraph opening "Which samples calibrate the quantizer is itself a studied lever". It cites Krishnamoorthi (2018), Nagel et al. (2021), Wu et al. (2020), Williams and Aletras (2024), Yu et al. (2021) and Cai et al. (2020), quotes both converter guides, and ends "established guidance, not a finding of this paper". The novelty statement is restated in §1 ("Representative calibration data is established guidance…") and in §9.1 (`sec:calibevidence`: "What this study adds is a measurement of that departure where it is easy to make"). The nulls are worded as R20 requires and the cohort evidence as R6 requires.

### R3. Closest on-device and health time-series work (EIC-W2 major / R2-W10 minor; SPLIT D1). *Addressed differently (in part).*
**Point.** No quantized on-device health or wearable classifier is cited; Table 1 lacks Zhu et al. and cites no source for OhioT1DM.

**Response.** Partly agreed. Zhu et al. and the OhioT1DM citation are added, and §2.4 now discusses glucose predictors run on device hardware. No other quantized health or wearable time-series classifier was verified for this revision, so none is cited. The Zhu row follows its cited record (Part 6.1).

**Change and location.** `tab:relatedwork` (§2) has a Zhu et al. (2023) row: evaluation scope "Three clinical datasets; 47 T1D participants (+ in silico trial)", device evidence "Wearable system-on-chip", horizon and inputs "---" (caption: "not stated in the cited work's abstract"). OhioT1DM is now cited (Marling and Bunescu 2020) in the Mirshekarian and Martinsson rows. §2.4 adds: "Glucose predictors have been run on such hardware: Li et al. … and Zhu et al. … Both forecast values, not trend classes."

### R4. Round five framed as a design result (EIC-W3, R1-W3, DA-M3, major). *Addressed (route b).*
**Point.** One seed-4 artifact compared with one champion artifact (+0.20 points, a fifth of the paper's 1.1-point seed spread) was presented as confirming the flatten head.

**Response.** Agreed. The revision takes route (b), the artifact level (author decision 2); the five-seed RT-CGM comparison of the two recipes (route a) was not run. Each citation of the +0.20-point gain now gives the spread and says what it is: the float, Weinstock held-out directional-recall span of round three's five seeds. Round five's own seed spread on RT-CGM was not measured. R1-W3's count of round-two seeds is restated to match the record (Part 6.2).

**Change and location.** Abstract; §1 contribution list; §8.5 (`sec:sweep5`: "a choice between two artifacts, not evidence for a design"); §9.3 ("It did not show that a flatten head, a cosine schedule or validation-chosen epochs improve this task"); §12; `tab:sweep` caption; Limitation 5; `MODEL_CARD.md` (Evaluation).

### R5. "Toolchain-default" label (EIC-W4, R2-W1, major; DA-m11). *Addressed.*
**Point.** Six passages called contiguous, dataset-order first-N selection the toolchain default, but neither toolchain chooses calibration data, and the cited guidance asks for random, unordered samples.

**Response.** Agreed. The selection came from this project's own conversion script, not from a default. The arm is renamed, the guidance is quoted verbatim, and no prevalence claim is made.

**Change and location.** The arm is named "contiguous first-N (dataset-order) selection" at first use (§1; §5.2, `sec:design`) and "contiguous first-N selection" after that. §5.1 says the script took its 200 windows "in dataset order, with no randomization or range balancing". §2.4 quotes the RepresentativeDataset text ("a small subset of a few hundred samples randomly chosen, in no particular order, from the training or evaluation dataset") and the LiteRT guide (around 100–500 samples). §9.1 adds: "the sequential arm contravenes it by taking contiguous windows in stored order". The word "toolchain" no longer occurs.

### R6. Second-cohort "confirmation" (EIC-W5 minor / R1-W1, DA-M1 major; SPLIT D2). *Addressed.*
**Point.** "Reproduces" and "Both predictions held" overstate the result. The frozen predictions set no decision rule or margin, both designated intervals include zero, and one premise failed.

**Response.** Agreed, following R1's remedy as arbitrated under D2. Each prediction's outcome is reported separately, and the interpretation is labelled exploratory. The EIC's added sentence on Hall's temporal component is kept. The EIC's proposed wording, that the predictions "held on their designated components", is not used, because the frozen records state no decision rule (Dubosson) and no margin (Hall).

**Change and location.** §7.7 says each protocol froze "one prediction, without a decision rule or margin", then reports:
- Dubosson: +1.98 points, exact sign-flip p = 0.094 ("the data do not separate it from zero");
- Hall held-out: +1.31 points, p = 0.25, "inconclusive";
- the Hall premise "failed": the sequential-200 set reached only 165.0 mg/dL;
- Hall temporal: +4.31 [+1.14, +7.34];
- "The following reading is exploratory."

The abstract ("Two small cohorts did not confirm the frozen predictions"), §1, Limitation 3 and §12 now agree. "Reproduces", "predictions held" and "as predicted" no longer describe the cohorts.

### R7. Cohorts behind the bundled model (EIC-W6, R2-W8 minor / R3-W2 major; SPLIT D3). *Addressed.*
**Point.** The five DiaData training cohorts, and the RT-CGM cohort that certified the bundled model, are not characterized.

**Response.** Agreed. A cohort table now covers every cohort used. Each cell comes from a cited primary report or a committed audit; unverified cells read "---". RT-CGM is identified as DiaData's copy of the JDRF CGM randomized trial. Its trial facts are those the cited primary report supports, and its sensors come from DiaData's audit. Its wear entry reads "wear not stated in the cited report", and the paper says that per-subject device and wear mode cannot be recovered (Part 6.2).

**Change and location.** `tab:cohorts` (App. B, `app:cohorts`) gives role, source, n, age, diabetes and therapy, CGM and wear, years and range for Weinstock, Dubosson, Hall, CITY, DCLP3, REPLACE-BG, SENCE, WISDM, PEDAP and RT-CGM. RT-CGM is also described in §2.2, §6.2, §8.3 ("the cohort name denotes this trial, not current real-time CGM"), the App. B paragraph (sensors, cadence re-audit, subjects without windows) and §11.2.

### R8. "Gained only on the majority class" (EIC-W7, R1-W9, DA-m5). *Addressed.*
**Point.** The abstract's "only" is contradicted by the per-class table (rising recall 0.365 against 0.334).

**Response.** Agreed. The rejection now rests on the directional-recall drop alone.

**Change and location.** Abstract ("rejected post hoc as directional recall fell (0.517 to 0.438)"); §1 contribution list; §8.1 (`sec:sweep1`: "That drop is the basis of the rejection"); per-class recall, precision and F1 in `tab:sweeprecall` (App. D.2).

### R9. One read per round (EIC-W8, R1-W6, DA-m1). *Addressed.*
**Point.** The abstract and conclusion say every round spent one confirmatory read; round four spent none.

**Response.** Agreed.

**Change and location.** Abstract ("Four of five sweep rounds spent one pre-registered read (round four none)"); §1; §6.2; §12 ("Four rounds spent one pre-registered confirmatory read each; round four … spent none"). All four now match §8.4 and the `tab:sweepfour` caption ("No read was spent").

### R10. "At most eight hundredths of a point" (EIC-W9, R1-W8, DA-m2). *Addressed.*
**Point.** The min–max observer moved recall by 0.12 points, more than the stated bound.

**Response.** Agreed.

**Change and location.** The abstract, §1, §9.1 and §12 now say "at most about a tenth of a point". §7.3 gives the components: budget at most +0.08 points, observer at most 0.12 points. `tab:protocolmap` (App. C.3) lists the observer arm as "At most 0.12 from the histogram observer".

### R11. Historical ceiling in §7.7 (EIC-W10, R1-W8, DA-M1). *Addressed.*
**Point.** §7.7 paired the historical artifact's 239.9 mg/dL ceiling with the corrected-protocol 13.4-point effect.

**Response.** Agreed.

**Change and location.** §7.7 now reads: "Weinstock's sequential-200 ceiling of 250.9 mg/dL clips 27.8% of held-out windows against none under uniform calibration, and costs 13.4 points." 239.9 mg/dL appears only for the historical artifact (§7.1, `fig:range`, `tab:provenance`); §7.7 mentions the historical 240 mg/dL only to explain Hall's frozen premise.

### R12. Clipped-window shares (EIC-W10, minor; DA-M1, major; DA-A5). *Addressed.*
**Point.** The mg/dL gaps do not order the cohort effects. The reviewers asked for each component's clipped-window share, with any dose-response statement based on those shares.

**Response.** Done. The shares separate Weinstock from the small cohorts but do not order the small-cohort components, so no finer dose-response is claimed (Part 6.2). The table also shows that uniform calibration itself under-covers both small cohorts.

**Change and location.** `tab:confirmation` adds "Clipped, all" and "Clipped, ff" columns (sequential-200 / uniform-200), e.g. Weinstock held-out 27.8 / 0.0% and Dubosson temporal 15.2 / 10.9%. The exploratory paragraph in §7.7 reports that excess clipping in the small cohorts is at most 4.3 percentage points, against effects of 0.3 to 4.3 points, and that "the shares do not order the effects". The mg/dL-gap argument is removed.

### R13. Neutral rate-band wording (EIC-W15, R2-W9, R3-W5, DA-m12; CONSENSUS-3). *Addressed.*
**Point.** "Clinical thresholds" and "clinical-shape classifier" contradict the research-only framing.

**Response.** Agreed.

**Change and location.** §3 and §8.4 use "trend-arrow rate bands (±1 and ±2 mg/dL/min)", and §8.4 decodes the regression head "at the same rate bands". §9.2 reads "a worse directional classifier with better accuracy". Neither flagged phrase remains; the other uses of "clinical" are scope disclaimers or describe cited work.

### R14. Small-cohort fragility in Table 11 (R1-W1, major; DA-m6). *Addressed.*
**Point.** Undefined bootstrap draws (1.2% for Dubosson) and small directional support were not disclosed.

**Response.** Agreed.

**Change and location.** `tab:confirmation` adds a "Support" column (`falling_fast`/`falling`/`rising`/`rising_fast` windows, e.g. 28/58/68/57 for Dubosson temporal and 41/261/303/50 for Hall held-out). Its caption gives the undefined-draw fraction of the superseded Dubosson temporal interval (1.2%) and states that no other component had undefined draws. §6.1 (`sec:calibprotocol`) states the rule: undefined draws are dropped before the percentile step, and the interval is omitted if more than 5% of draws are undefined.

### R15. Inference at six clusters (R1-W2, major). *Addressed.*
**Point.** A percentile pairs-cluster bootstrap at six clusters gives intervals that are too narrow.

**Response.** Agreed. The six-participant designated components now get an exact sign-flip test over participants instead of an interval, and per-participant contributions are reported. The frozen percentile intervals stay in the caption, marked superseded, so the pre-registered result remains visible. The Cameron, Gelbach and Miller citation is used only for what it shows, that cluster-based inference tends to over-reject with few clusters; it does not evaluate percentile intervals directly (citation log, entry 19).

**Change and location.** §7.7 reports each participant's additive contribution to the pooled difference, sign-flipped in all 64 patterns (smallest attainable two-sided p = 0.031).
- Dubosson: one participant contributes +1.62 of the +1.98 points, the other five −0.04 to +0.16; p = 0.094.
- Hall held-out: one participant contributes +1.08 of the +1.31 points, the other five −0.09 to +0.21; p = 0.25.

See also the `tab:confirmation` "Inference" column and Limitation 4 ("six-participant components get an exact sign-flip test instead of a valid interval"). Every contribution is listed under `confirmation` in `panel_revision_r1.json`.

### R16. Attribute round five to the bundle (R1-W4, minor; DA-M3, major). *Addressed.*
**Point.** The abstract credited the flatten head alone, although head, schedule, kept epoch and seed changed together.

**Response.** Agreed.

**Change and location.** Abstract ("a seed-4 flatten-head artifact (cosine schedule, validation-chosen epoch)"); §1 ("Its four changed factors are not separately identified"); §8.5 (the candidate "differs from the champion in head, learning-rate schedule, kept epoch and seed at once"); §12 ("Its head, schedule, kept epoch, and seed changed together and are not separately identified").

### R17. Take-home as input-range coverage (R1-W4, minor; DA-M2, major). *Addressed.*
**Point.** "Coverage means covering participants" conflicts with the full recovery of the 6-participant sequential-1,000 set and with the clip-only control.

**Response.** Agreed.

**Change and location.** §12: "What mattered was covering the deployment input range, which any randomized selection did in this pool". §9.1 (`sec:calibevidence`, retitled "Input-range coverage, not sample count") states the rule. It notes that the 1,000-window sequential set recovers fully with 6 participants because it reaches the full range.

### R18. Rename the round-four quantity (R1-W10, R2-W3, DA-m10). *Addressed.*
**Point.** The 0.954 agreement compares two labels computed from the same future readings, so the "ceiling" and "headroom" bound nothing a past-only model can reach.

**Response.** Agreed.

**Change and location.** §8.4 is retitled "Round four: a label-definition stability check and an identity outcome". It defines both labels and states three points: both are functions of the same future readings, their agreement "bounds nothing that a model seeing only the 12 past readings can reach", and the recall attainable from CGM-only inputs remains unknown. It also says the abort rule "could hardly fire". §1 and §12 use "label-definition stability check". Neither "label-noise ceiling" nor "headroom" occurs.

### R19. Limit the label-noise conclusion (R1-W10, R2-W3 minor / R3-W3 major; SPLIT D4). *Addressed.*
**Point.** "Places the remaining error in the model and data, not the labels" goes beyond a test with independent noise of σ = 1–2 mg/dL.

**Response.** Agreed, as arbitrated under D4. The claim is removed, and every remaining label-robustness sentence names the noise model and cites a published sensor-error model. The optional colored-noise rerun was not run (Part 8).

**Change and location.** §8.4 now names the noise ("independent Gaussian noise of σ = 1 or 2 mg/dL"), calls it "gentler than published CGM error models, which add blood-to-interstitial lag, sensor-calibration drift and autoregressive noise" (Facchinetti et al. 2014), and says the labels are "defined on the sensor trace, which itself lags blood glucose". The same scope appears in Limitation 7, in §12 ("the recall attainable from CGM history is unknown") and in `MODEL_CARD.md` ("Labels follow the sensor").

### R20. Scope the null contrasts (DA-M2, major; D7). *Addressed (scoping route).*
**Point.** Every randomized 200-window selection received the same 0–400.8 mg/dL input range, so the null primary and budget contrasts cannot support a design rule.

**Response.** Agreed. The revision takes the scoping route (author decision 2). Range-aware selection was not tested on Dubosson or Hall, where uniform draws under-cover the range.

**Change and location.**
- §7.3: the primary "could not test H1 … as posed", because both arms "receive the same 0–400.8 mg/dL input range"; the budget and interaction results are stated "within this shared-range regime".
- §9.1: the nulls show that "once the range is covered" the other choices moved recall by at most about a tenth of a point, "not that range-aware selection is useless". That selection was not tested where uniform-200 sets still clip 10.9% (Dubosson temporal) and 0.5–1.0% (Hall) of windows.
- §12: "These nulls, the null primary contrast among them, are scoped to that saturated regime". The abstract, §1 and Limitation 2 carry the same scope.
- `tab:protocolmap` marks H1, H2 and the primary "Untestable as posed".

### R21. Round-five costs wherever the result is summarized (DA-M3, major). *Addressed.*
**Point.** The abstract reported only the gain. On the record, the candidate fails gates 1 and 2 on both Weinstock components, lowers macro-F1 on the primary cohort and ties on PEDAP.

**Response.** Agreed. §1, §8.5, §9.3, §12 and `tab:sweep` report the Weinstock gate outcomes, the RT-CGM macro-F1 change and PEDAP directional recall (0.539 against 0.541) beside the +0.20-point gain. The abstract carries the gates and macro-F1 but not PEDAP, because it is at 1,915 of arXiv's 1,920 characters.

**Change and location.**
- Abstract: "lowered macro-F1 there (0.410 vs. 0.427), failed two gates on both Weinstock components".
- §1 contribution list and §9.3: failing two gates on both Weinstock components, lowering RT-CGM macro-F1 (0.410 against 0.427), and "PEDAP is a near-tie (0.539 against 0.541)" (§9.3: "with PEDAP a near-tie").
- §8.5: held-out −0.32 points with stable recall 0.436 against 0.459; temporal +0.33 with 0.499 against 0.520; macro-F1 lower on all three paired splits; PEDAP 0.539 against 0.541.
- §12 (including the PEDAP near-tie); `tab:sweep` rows and caption.

## Part 4. Suggested revisions (S1–S39)

| Item | Reviewer point | Outcome and response | Location |
|---|---|---|---|
| S1 | EIC-W6: Methods, Figure 1 and Table 1 describe a model no longer bundled; add a model-variants table. | Addressed. Figure 1 and §4.3 give the bundled models' range (2,909–50,645 parameters; trained on Weinstock or DiaData). A variants table lists architecture, parameters, input encoding, loss, training data and recipe, INT8 calibration pool and hashes. Table 1's row lists both training sources and all evaluation cohorts. The range covers bundled variants only (Part 6.2). | §4.3; `fig:pipeline`; `tab:variants` (App. B); `tab:relatedwork` |
| S2 | EIC-W6, R1-W11: the bundled model's INT8 calibration pool is not stated. | Addressed. Every sweep candidate is calibrated with uniform-200 (draw 101), drawn from the study's pool of 27,109 eligible Weinstock validation windows; from round two on, this calibration is cross-cohort. The frozen round-five protocol file misdescribes the pool; the file stays unchanged, and App. D.1 records the discrepancy (Part 6.2). | §6.2; §8.5; `tab:variants`; App. D.1; `MODEL_CARD.md` |
| S3 | EIC-W7.1: shorten the abstract to about 200 words. | Addressed differently. The abstract keeps 278 words and 1,915 characters, under arXiv's 1,920 (Part 6.1). | Abstract |
| S4 | EIC-W7.2, R1-W5: label the 13.4-point headline a prespecified secondary; state the null primary. | Addressed. | Abstract ("In a prespecified secondary contrast…"; "The primary contrast (range stratification) was null"); §1; §7.3; §12 |
| S5 | EIC-W7.3, R1-W6, DA-m9: delta encoding is post hoc and its variant held-out-selected; "nearly calibration-proof" overreaches. | Addressed. The experiment is labelled post hoc. The anchor variant "was chosen on held-out data (0.5088 against 0.4969)", and its margin is "selection-affected". The encoding makes INT8 "far less sensitive to calibration selection, not insensitive", and the frozen ship criterion passed at 0.4955 points against a 0.5-point limit. | Abstract; §7.8 (`sec:deltainputs`); `tab:readledger` |
| S6 | EIC-W9: "the same 200 windows" is wrong for randomized sets. | Addressed: "Any randomized selection of 200 windows". | §12 |
| S7 | EIC-W11, R1-W7: stale device-timing sentence and artifact count. | Addressed, with the count the synthesis resolved on the record. One float and 8 of the 110 study INT8 artifacts were timed. The other 102, and the 60 second-cohort and delta-encoded INT8 artifacts, were not. All 170 share one size and operator inventory. | §6.1; §7.9; Limitation 9; `tab:devicebench` caption |
| S8 | EIC-W12, W13, W14, W16: revision history in the text; small type; Figure 3 legend over the bars; shifting reference conventions; "Macro" column label. | Addressed. Revision history and the historical comparison moved to App. A; §1 keeps one sentence. The four sweep confirmatory tables are merged into one. No table uses `\tiny`, `\scriptsize` or `\resizebox`, and the bibliography is back to `\small` from 7.5 pt. The `fig:recall` legend sits above the axes. Each champion is named once, and every quoted level names its artifact and split. Five-class columns read "Macro recall". | App. A; `tab:sweep`; `fig:recall`; §6.2; §8 (opening); `tab:calibmain`; `tab:baselines` |
| S9 | EIC-W17, R1-W5, DA-m7: the frozen budget-by-strategy interaction is not reported. | Addressed for both components. `protocol.json` names the interaction without a formula, so its definition (the difference between two strategies' 1,000-vs-200 effects) is disclosed as fixed after the primary analysis. Randomized interactions run from −0.05 to +0.06 points (temporal up to +0.07), every interval including zero; uniform against sequential is −13.33. | §6.1; §7.3; `tab:contrasts` |
| S10 | EIC-W18: `recency_collision.json` and round four's frozen design are not versioned. | Addressed differently. The diagnostic is versioned in `paper/analysis/accuracy-review-2026-10-13/`, and round four's design is in its specification at commit `c8f3de6`; the paper now gives both (Part 6.1). | §8.5; §11.1; App. D.1 |
| S11 | EIC-W19: licence of the distributed weights; conventional ethics statement. | Addressed. The weights, trained on DiaData cohorts under CC BY-NC 4.0, "are released for non-commercial research use only". The ethics statement says the analysis used public, de-identified data under the providers' terms, recruited no participants, and sought no ethics review. The repository has no LICENSE file (Part 8). | §1 (end); §11.2; `MODEL_CARD.md` (Licence) |
| S12 | R1-W5, DA-m7, DA-m8: protocol-to-report table; seed and draw variation; interval for sequential 1,000 vs 200; state H1–H4. | Addressed. `tab:protocolmap` maps H1–H4 (abbreviated from the study plan), the primary and every secondary to outcome and status. Across training seeds the 13.4-point effect has SD 0.47 points (range 13.05 to 14.14); float directional recall has SD 1.34 points; within-seed draw SD is 0.02 to 0.21. Sequential 1,000 vs 200 is +13.35 [+10.35, +16.77]. DA-m8's premise is corrected (Part 6.2). | §6.1; §7.3; `tab:contrasts`; `tab:protocolmap` (App. C.3) |
| S13 | R1-W6, DA-m1: add a read ledger; disclose split reuse and the post-hoc round-one decision in the abstract. | Addressed. The ledger lists every analysis that read held-out, temporal or RT-CGM labels, with date, commit and decision; it includes the delta-variant choice and this revision's re-analyses. The abstract says three reads were "on reused Weinstock test components" and that the round-one winner was "rejected post hoc". | Abstract; `tab:readledger` (App. B); Limitation 5 |
| S14 | R1-W7: give the operator inventory and size for the second-cohort and delta artifacts. | Addressed. All 40 second-cohort and 20 delta-encoded INT8 artifacts are 11,976 bytes and share the study's single operator inventory. | §7.9 (`sec:deploycost`) |
| S15 | R1-W8: a spliced sensitivity-tier interval. | Addressed. The text now quotes the widest single-tier intervals: held-out gap-filled inputs +11.44 [+6.56, +15.90]; temporal gap-filled endpoint +10.53 [+6.48, +14.56]. | §7.6 (`sec:sensitivity`) |
| S16 | R1-W9: per-class F1 and precision for round one. | Addressed. F1 rose in every class; `stable` contributed +14.1 points, half of the summed per-class gain. Precision rose in all four directional classes while recall fell in three. | §8.1; `tab:sweeprecall` (App. D.2) |
| S17 | R1-W11: per-artifact hashes; per-participant confusion counts; sweep batch size ("512 vs the stated 64"). | Addressed differently. Hashes of all 190 artifacts are published. The batch size is stated per run: 64 for the study and round one, 512 for rounds two to five. Confusion counts stay local, and the paper states the limitation (Part 6.1). | §4.3; §6.2; §11.1; App. D.1; `artifact_hashes_r1.json` |
| S18 | R2-W4, R3-W9: forecasting evaluation is described as RMSE-only. | Addressed. §2.3 describes CG-EGA's rate-error grid and its use on predictors, the J index's time gain, and coherence penalties. It also says what per-class recall does not capture: adjacent versus reversed errors, level and timing. | §2.3 |
| S19 | R2-W5: Kistkins et al. are misquoted. | Addressed: "best models reach recalls of 0.85–0.98, with euglycemia recall of 0.59–0.70 at 1 h, under within-patient splits". The study had 11 participants and 15-minute data, so its 15-minute horizon is one step ahead. | §2.3 |
| S20 | R2-W6, R3-W7, R3-Q2: 39 and 401 mg/dL are treated as measurements. | Addressed. Among distinct readings in eligible held-out windows, 922 equal 39 and 1,253 equal 401, against 33 at 40 and 22 at 400. The paper calls the values "consistent with stored LOW and HIGH codes" and a coded window's rate a bound. Held-out, 6.6% of windows have a coded input and 4.2% a coded endpoint; inputs range from 3.3% to 9.8% by class, and endpoints concentrate in `stable` (5.4% against 0.7–1.7%). A post-hoc check without coded windows keeps the contrast at +13.27 [+10.25, +16.75]. The data dictionary was not checked, and training still uses the codes as values (Limitation 6). | §4.1; §7.6; `tab:sensitivity` (App. C.4); Limitation 6; `MODEL_CARD.md` |
| S21 | R2-W7: what calibration changes inside the INT8 artifact; per-op dtype map; CPU backend and threads. | Addressed differently. The proposed wording ("changes only quantization parameters") is not adopted, because recalibration also changes the stored int32 bias of the final layer (Part 6.1). The paper states the 12-of-41-tensor change and that only the final fully connected layer runs in int8. It names the backend (LiteRT's CPU accelerator through the XNNPACK delegate) and states that no thread count is set. The per-operator type map is published. | §5.1; §5.3; §7.9; `artifacts` in `panel_revision_r1.json` |
| S22 | R2-W8, R3-N1: the RT-CGM cadence audit cannot fail; 96 subjects without windows; collinear fraction. | Addressed. The paper reports that the rule "could not fail", the alternative the reviewer allowed to replacing it. The 96 subjects without a scored window "contain no run of 15 consecutive five-minute readings". The collinear fraction is defined, and its excess (0.305 against at most 0.229) is attributed mainly to runs of identical readings, which the statistic cannot tell from interpolation. | §8.3; App. B |
| S23 | R2-W11, R3-W8: sweep methods are not cited. | Addressed. The paper cites focal loss, TCN, cosine schedule, CORN, model soups, linear mode connectivity and imbalanced regression. The distance penalty and ordinal label smoothing are marked as introduced here. | §6.2; §8.1; §8.3; §8.4 |
| S24 | R2-W11: soup collapse and regression toward the mean are known results. | Addressed. Both are framed as expected outcomes, with citations. | §8.4 |
| S25 | R2-W12, DA-A1: a fixed input range from the sensor's reportable range is not considered. | Addressed as discussion; the optional arm was not run (author decision 2). Fixing the input range at 39–401 mg/dL would remove the dependence on calibration data. PT2E offers fixed quantization parameters, but LiteRT Torch applies them only to sigmoid and softmax outputs, so the input would need a custom annotation, which was not tested. | §9.1 |
| S26 | R2-W13: 8-bit error analysis predicts the absent resolution cost. | Addressed. The step is 1.572 mg/dL, the maximum rounding error 0.79 mg/dL and the RMS error about 0.45 mg/dL, against the 15 mg/dL change that separates classes and the up to 150 mg/dL that clipping removes. | §9.1; §7.4 |
| S27 | R2-W14: "6.4× finer" omits the level channel. | Addressed. The differences get 0.247 mg/dL per code (6.4 times finer than 1.572) and the level channel about 3.95 mg/dL (2.5 times coarser). | §7.8 |
| S28 | R3-W1, R3-N1, DA stakeholders: explain the bundled model's error profile (minimum: a paragraph plus a confusion matrix). | Addressed. §9.3 says what a displayed class means and how often the indicated rate follows. 63% of displayed movement classes are followed by a mean rate within ±1 mg/dL/min. A displayed `stable` is right in 92% (held-out) and 94% (RT-CGM) of windows. 9–15% of truly falling or rising windows are shown in the opposite direction. App. D.4 gives recall (sensitivity), precision (PPV), adjacent-error share, reversals, recall by anchor band and row-normalized confusion matrices. | §9.3; `tab:shippederrors`, `tab:shippedconfusion` (App. D.4) |
| S29 | R3-Q1, Q4, Q5, Q6. | Answered below. Q2 is answered under S20 and Q3 under S28. | This response |
| S30 | R3-W2.3–6: wear and therapy regimes, feedback loop, PEDAP population, unrepresented groups; expand RT-CGM. | Addressed. A new Limitations item covers blinded, mixed and closed-loop wear, the feedback a deployed display would cause, PEDAP's preschool population, and the absence of type-2, pregnancy and inpatient data. The abstract no longer leaves "RT-CGM" bare ("a never-evaluated 2008 trial cohort"), and later uses name the trial. | Limitation 8; Abstract; §1; §2.2; §8.3 |
| S31 | R3-W4, W5, W8: "calibration" vs sensor calibration; target wording; acronyms; recall vs sensitivity. | Addressed. The title says "Quantization-Calibration". §1 separates it from CGM sensor calibration and from probability calibration. The target is "the mean rate of change over the next 15 minutes". Acronyms are expanded at first use, including macro-F1 in the abstract and TCN in the §8.3 heading. Recall is mapped to sensitivity and precision to PPV. | Title; Abstract; §1; §8.3; `tab:sweeprecall`; `tab:shippederrors` |
| S32 | R3-W6: Weinstock's case-control enrichment and blinded wear. | Addressed. The cohort has 101 cases with a severe hypoglycemic event and 100 controls, so it is enriched for hypoglycemia and variability. CGM was blinded, so trajectories lack the corrections a real-time display would prompt. | §4.1; Limitations 2 and 8 |
| S33 | R3-W10, DA stakeholders: intended-use statement, model card, "bundled" wording. | Addressed. A new `MODEL_CARD.md`, structured after Mitchell et al., gives intended use, out-of-scope uses, inputs, evaluation, failure modes and unrepresented populations; §11.2 summarizes it. "Bundled in the research applications" replaces "ships". | §11.2; `MODEL_CARD.md`; Abstract; §1; §12 |
| S34 | DA-A2, DA-UP: state that no INT8 latency or size benefit was measured; scope "trade-offs". | Addressed differently. INT8 shows no mean-latency advantage in any measurement, but its size saving was measured: 31% for the study model, 70% for the bundled one. "Trade-offs" is gone from the title. Limitation 9 states the trade-off as an accuracy risk against a size saving, with no measured speed benefit (Part 6.1). | Title; §1; §7.9; Limitation 9; §12 |
| S35 | DA-A3: report PEDAP per-class changes; consider an operating-point shift. | Addressed. Recall changes: `falling_fast` +6.7 points, `rising_fast` +2.3, `falling` −6.5, `rising` −3.1, `stable` −0.3. `falling_fast` is predicted in 16.0% of windows against 12.3%, with precision falling from 0.284 to 0.242. The paper reads this as an operating-point shift. DA-A3's −6.4 for `falling` came from rounded values; the record gives −6.45. | §8.5 |
| S36 | DA-A4, DA-m14 (D5): reconcile the §8.1 mechanism statements. | Addressed. Both accounts are presented as fitting the mechanism controls: fast falls start high, and clipping erases the slope of a high fall. The paper says the study "does not separate them". There is no new test, so the D5 dissent stays unresolved. | §9.1 |
| S37 | DA-m3 (D6): "add nothing measurable" conflicts with a budget interval that excludes zero. | Addressed. The phrase is replaced by "at most about a tenth of a point", and §7.3 notes that the balanced-random budget interval [+0.02, +0.15] excludes zero. | §1; §7.3 |
| S38 | DA-m4, DA-m13: randomized participant range; "never-read". | Addressed. The range is now 90–127 participants. "Never-read" becomes "never-evaluated", and the rejected round-three training arm is disclosed. | §7.2; Abstract; §1; §8.5; §12; Limitation 5 |
| S39 | DA-O2: no gate-override rule. | Addressed. §6.2 says no protocol stated one, and gives the rule both decisions fit: override only for a directional-recall loss on the primary component. | §6.2 |

### Answers to Reviewer 3's questions (S29)

**Q1. Which blinded sensor did the Weinstock study use, did it need fingerstick calibration, and can calibration-related jumps be identified?** GlucoBench lists the sensor as a Dexcom G4. The device could not be confirmed in the source study's own publication, whose full text could not be retrieved, so whether that protocol required fingerstick calibrations is unknown. The GlucoBench Weinstock file holds a participant id, a timestamp, one glucose column and static participant covariates, but no calibration-event field. Calibration-related jumps therefore cannot be located against calibration times, and no attempt was made to detect them from the trace alone. *Change:* §4.1 now names the sensor as GlucoBench lists it and states what blinded wear implies; Limitation 7 notes the sensor-calibration drift in the trace the labels follow.

**Q4. Should the recent-slope baseline be presented as the persistence assumption clinicians make when they read the current arrow as a forecast?** In substance it is that assumption. The baseline maps the past 15-minute rate, (g_t − g_{t−15})/15, through the same five bands, so it assumes the latest rate continues (§4.3). On held-out participants the float CNN reaches 0.487 directional recall against the baseline's 0.278, at lower accuracy (0.507 against 0.662; §7.5, `tab:baselines`). *Change:* the value now reads 0.278, matching the record (0.278492); the reviewed text printed 0.279. The clinician framing was not added, because the sources on how people use trend arrows were not verified for this revision (citation log, unverified list).

**Q5. Does the sequential ceiling make the INT8 model blind across the consensus band above 250 mg/dL?** With the arm renamed (R5): under contiguous first-N selection the INT8 input range is 0–250.9 mg/dL, and every reading above 250.9 mg/dL maps to one input code (§9.1). Within the band above 250 mg/dL the INT8 input therefore carries almost no level information, since the model sees such readings as about 250.9 mg/dL. 27.8% of held-out windows have at least one clipped input slot, as do 60.9% of `falling_fast` windows (§7.4). The historical artifact's 0–239.9 mg/dL range clipped lower still, below that band's 250 mg/dL edge. *Change:* none; the manuscript uses the consensus band edges only in the bundled model's error profile (`tab:shippederrors`).

**Q6. Does the application show the predicted class next to the current glucose value?** Yes. Both applications show the current reading in mg/dL as a headline directly above the predicted class. The class appears as an arrow glyph with its name and the caption "predicted trend, next 15 min", with a line chart of recent readings and one probability bar per class (`android/app/src/main/kotlin/com/glucoedge/app/ui/MainScreen.kt`; `ios/GlucoEdge/UI/MainView.swift`). *Change:* none; the manuscript does not describe the screen layout, and §9.3 states what a displayed class means.

## Part 5. Arbitrated disagreements (D1–D7)

| | Disagreement | How the revision resolved it |
|---|---|---|
| D1 | Severity of the missing on-device health and glucose work: EIC major, R2 minor. | R3. R2's concrete items are added (the Zhu et al. row, the OhioT1DM citation), and §2.4 discusses on-device glucose prediction. The EIC's wider request, for quantized health or wearable time-series classifiers in general, is met only by these works, since no other source was verified (Part 6.1, Part 8). |
| D2 | Second-cohort confirmation: EIC minor, keep "held"; R1 and DA major, report each outcome. | R6, following R1. Outcomes are stated separately, the interpretation is labelled exploratory, and "reproduces" and "held" are gone. The EIC's Hall-temporal sentence is kept (+4.31 [+1.14, +7.34]). The EIC's "held on their designated components" is not used, because the frozen records state no decision rule or margin. |
| D3 | Severity of the undescribed cohorts: R3 major; R2 and EIC minor. | R7. The cohort table and RT-CGM descriptions answer R3's concern that the abstract's cohort could not be interpreted. R2's point that the paired contrast stays internally valid still holds, since both models are scored on the same windows (§6.2). |
| D4 | Label-noise conclusion: R1 and R2 minor (definitional); R3 major (noise model). | R18 and R19. The definitional reading is adopted ("label-definition stability check"), and R3's qualifier applies to every remaining label-robustness sentence. The colored-noise rerun was not run (Part 8). |
| D5 | Mechanism narrative: R3-S4's start-high account against the DA's slope erasure. | S36. The dissent stays unresolved: §9.1 presents both accounts as consistent with the mechanism controls and states that the study does not separate them. |
| D6 | "Add nothing measurable": the DA notes that the balanced budget interval excludes zero; R1 judges the effect practically negligible. | S37. Both points stand in the text: the phrase is replaced by "at most about a tenth of a point", and §7.3 names the one budget interval that excludes zero. |
| D7 | What the null contrasts show: the EIC counts the nulls as new; DA-M2 says there was no input-range difference to detect. | R20 and R2. DA-M2 is upheld. The nulls are scoped to the saturated regime wherever they are stated, the §1 contribution list presents them as conditional on full range coverage, and `tab:protocolmap` marks H1, H2 and the primary "Untestable as posed". |

## Part 6. Where the revision departs from a reviewer request

### 6.1 Remedy not carried out as asked

- **R3 (EIC-W2; R2-W10).**
  - The new §2.4 text covers on-device glucose predictors only. The citation log verified 30 works for this revision; apart from Zhu et al., none is an on-device health or wearable time-series model, and unverified works are not cited.
  - The Zhu row follows its cited record. The issue year is 2023 (vol. 10, no. 5); 2022 is the early-access date (citation log, entry 7). Horizon and inputs are printed "---" because the cited abstract states neither.
- **S3 (EIC-W7.1).**
  - A plain-text count of the revised abstract gives 278 words and 1,915 characters, against arXiv's 1,920-character limit.
  - A 200-word abstract would have to drop disclosures the roadmap itself requires: the secondary-contrast label and null primary (S4), the saturated-regime scope (R20), the post-hoc delta result (S5), the unconfirmed cohort predictions (R6), the read count and split reuse (R9, S13), round five's attribution, seed-spread context and costs (R4, R16, R21), and the "bundled" wording (S33). These come on top of the two research questions (R1).
  - The EIC's remedy also assumed the calibration-only route, which was not taken.
- **S10 (EIC-W18).**
  - `recency_collision.json` is versioned with its reproduction script at `paper/analysis/accuracy-review-2026-10-13/` (commit `3755d55`, 2026-10-14); the reviewer checked `paper/results/` only.
  - Round four, which spent no read, has no protocol file. Its design and pre-registered abort rule are in `docs/superpowers/specs/2026-10-13-accuracy-sweep-r4-design.md` (commit `c8f3de6`, 2026-10-13), committed before the round's record (`cdd26ea`, 2026-10-14).
  - The paper now cites the path and the commit (§11.1; App. D.1).
- **S17 (R1-W11).**
  - The roadmap's "512 vs the stated 64" is right for rounds two to five and wrong for round one. `sweep_settings` in `panel_revision_r1.json` records batch size 64 on the CPU for round one (the study's setting; `protocol.json` lists `batch_size` 64) and 512 on MPS for rounds two to five (their smoke records). The paper states both (§4.3; §6.2; App. D.1; `tab:variants`).
  - Per-participant confusion counts are not published. Like the predictions and window keys, they derive from participant-level data distributed under the providers' terms, so they stay local, and §11.1 states the limitation. The reviewer's fix allows this as the alternative.
- **S21 (R2-W7, proposed wording).**
  - The proposed sentence ("changes only quantization parameters (input, intermediate and output scales and zero points)") is incomplete. Between the sequential-200 and uniform-200 artifacts of one checkpoint, 12 of 41 tensors differ. One of them is the int32 bias constant of the final layer, whose stored values change because its scale follows its input's (`calibration_changes_inside_int8` in `panel_revision_r1.json`). The int8 weights are identical.
  - The per-operator type map (`study_int8_dtype_map`) shows the depthwise and standard convolutions, ADD, SUM and MUL running in float32 on dequantized tensors, and only FULLY_CONNECTED on int8 inputs.
  - No thread count is given because neither client sets one and runtime threading was not instrumented.
- **S34 (DA-A2, DA-UP).**
  - No mean-latency benefit was measured. INT8's mean latency is higher than float's in every benchmark arm (+0.64 to +0.70 μs; `tab:devicebench`), in the historical observations (App. A.3) and in the bundled pair's live session (0.126 against 0.124 ms), where the INT8 p95 was marginally lower (0.179 against 0.182 ms; App. C.6).
  - A size benefit was measured. The study model falls from 17,352 to 11,976 bytes (31.0%), and the bundled pair from 208,560 to 61,936 bytes (70.3%, per `MODELS.md`; §7.9, Limitation 9).
  - The DA's "no benefit side" therefore holds for mean latency but not for size.

### 6.2 Reviewer premise corrected within an addressed item

- **S2; R1-W11 (round-five protocol wording).** `protocol_sweep_r5.json` says the INT8 calibration selections come "from weinstock train only". Every round's selections carry the main-study pool hash, which identifies the 27,109 eligible Weinstock validation windows (`flags` and `sweep_settings` in `panel_revision_r1.json`), so every round calibrated on that validation pool. The frozen file is left unchanged. The paper states the pool (§6.2; `tab:variants`) and records the discrepancy (App. D.1).
- **S1 (parameter range).** "2,909 to 50,645 parameters" in §4.3 and Figure 1 describes the four bundled variants (`tab:variants`), not every model trained. The round-three TCN (63,005 parameters) and the other arms that were never bundled are reported in their rounds (§8).
- **R12; EIC-W10 (dose-response).** The EIC expected the clipped-window shares to order the cohorts.
  - The shares do separate Weinstock (27.8 percentage points of excess clipping, 13.4 points) from the small cohorts (at most 4.3 percentage points; 0.3 to 4.3 points).
  - They do not order the small-cohort components among themselves. Dubosson temporal has the largest excess (4.3 percentage points) but a smaller effect (+1.98) than Hall temporal (excess 3.1; +4.31).
  - The paper says so and claims no finer dose-response (§7.7).
- **S12; DA-m8 (hypotheses on record).** `protocol.json` has no hypothesis field, but it names the study plan (`docs/superpowers/plans/2026-09-19-calibration-coverage-paper-enhancement.md`) and records the freeze commit `d63cc8f`. That commit (2026-10-10 00:49 −0500) adds the plan, which states H1–H4, before the primary-results commit `9dd88bb` at 02:16 the same day. The paper names the plan and commit (§6.1) and abbreviates H1–H4 in `tab:protocolmap`.
- **R4; R1-W3 (round-two seeds).** R1-W3 counts 5 of 5 round-two seeds beating the champion. Seed 4's held-out INT8 directional recall is 0.517440 against the champion's 0.517431 (`accuracy_sweep_r2.json`), a lead of 0.001 points. The paper therefore says the recipe "matched or beat" the champion "at every seed", with "none below it" (§8.2, §9.3). Round one's 5 of 5 stands (§8.1).
- **R7; R3-W2 (JDRF trial details).** For RT-CGM the paper prints the trial facts its cited primary report supports: 322 randomized, intensively treated children and adults in three age groups (8–14, 15–24 and ≥25 years). It takes the sensors from DiaData's audit. In `tab:cohorts` the enrolment years read "---" (not verified from the cited source) and the wear entry reads "wear not stated in the cited report". The paper also states that DiaData records neither device nor wear mode per subject.

## Part 7. New analyses and their records

All re-analyses use cached per-window predictions, saved artifacts and committed records; no model was trained, converted or run (author decision 2). The read ledger logs them as a descriptive read (`tab:readledger`, 2026-10-16, `c739c76`).

**`paper/results/calibration-coverage/panel_revision_r1.json`**, written by `experiments/panel_revision.py` (commit `c739c76`):

| Block | Contents | Items | Used in |
|---|---|---|---|
| `weinstock.seed_draw_variation` | per-seed float and per-draw INT8 directional recall | S12 | §7.3 |
| `weinstock.interaction`, `weinstock.sequential_1000_vs_200` | interaction contrasts, with the post-hoc definition recorded; sequential budget contrast; same bootstrap as the published contrasts | S9, S12 | `tab:contrasts` |
| `weinstock.censoring` | evidence for the 39/401 codes; coded-input and coded-endpoint shares by class; contrast without coded windows | S20 | §4.1; `tab:sensitivity` |
| `weinstock.facts` | observer differences; randomized-200 participant range; INT8 input ceiling by cell; widest single-tier intervals | R10, R20, S15, S38 | §7.2; §7.3; §7.6 |
| `confirmation` | per-participant contributions; exact sign-flip tests; clipped-window shares, including the Weinstock comparison | R12, R15 | §7.7; `tab:confirmation` |
| `shipped_model` | bundled INT8 model on Weinstock held-out and RT-CGM: confusion matrices, recall by anchor band, RT-CGM anchor range | S28 | §8.5; §9.3; App. D.4 |
| `round_one_f1` | round one per-class recall, precision and F1 | S16 | §8.1; `tab:sweeprecall` |
| `pedap_per_class` | PEDAP per-class recall, precision and predicted shares, round-five candidate against the champion | S35 | §8.5 |
| `artifacts` | timed and untimed counts; sizes and operator inventories of the 60 follow-up INT8 artifacts; per-operator type map; tensor comparison of two calibrations | S7, S14, S21 | §6.1; §7.9; Limitation 9 |
| `sweep_settings` | batch size and device per round; calibration pool | S2, S17 | §6.2; §11.1 |
| `flags` | four contradictions between records and earlier text: the round-five pool wording; the batch size; what calibration changes inside INT8; the randomized participant range (96–127 printed, 90–127 observed) | S2, S17, S21, S38 | Part 6 |

**Other records.**
- `paper/results/calibration-coverage/artifact_hashes_r1.json`: SHA-256 hashes of 190 artifacts (115 study, 25 Dubosson, 25 Hall, 25 delta-encoded); S17, §11.1.
- `paper/analysis/panel-review-2026-10-16/citation_verification.md`: 30 works verified against primary records and added to `paper/references.bib`, with reviewer-supplied details corrected (e.g. the Zhu et al. year and the Cameron et al. scope). Used for R2, R3, R7, R15, R19, S18, S23, S24, S28 and S33.
- `MODEL_CARD.md` (new): intended use, out-of-scope uses, inputs, evaluation, failure modes, unrepresented populations and licence; S11, S33, §11.2.

## Part 8. Remaining limitations

The revision does not resolve the following.

1. **Five-seed RT-CGM comparison not run.** Round five's seed variability on RT-CGM is unmeasured, so the +0.20-point result supports an artifact choice only, and no design factor (head, schedule, epoch rule) is credited.
2. **Range-aware selection untested where it could matter.** In this pool every randomized selection reached the full input range, so H1 was untestable as posed. Dubosson and Hall, where uniform draws under-cover the range, were not used for that test.
3. **Small-cohort evidence is exploratory.** The designated components have six participants, and their frozen predictions set no decision rule or margin (sign-flip p = 0.094 and 0.25). Hall's temporal component (+4.31) runs against its frozen near-zero prediction, whose premise failed.
4. **Label stability against realistic sensor error is untested.** Round four perturbed labels only with independent noise of σ = 1–2 mg/dL; lag, drift and autocorrelated error were not simulated. The recall attainable from CGM history alone is unknown.
5. **Device timing on one phone.** Nine study artifacts were timed on one iPhone 18 Pro Max, and the 60 follow-up INT8 artifacts were not timed. Energy and memory were not measured, and the Android phone of the historical observations is no longer available.
6. **No LICENSE file.** The repository has no licence file. The non-commercial research terms for the weights appear only in the paper (§1, §11.2) and `MODEL_CARD.md`, and no licence is stated for the code.
7. **Related-work gap.** Apart from on-device glucose predictors, no quantized health or wearable time-series classifier is cited, because none was verified.
8. **Coded readings.** Readings of 39 and 401 mg/dL are still treated as glucose values in training, calibration and evaluation. Their encoding was not confirmed in the source data dictionary, and the only check is an exclusion analysis.
9. **Recomputation.** Recomputing any interval requires regenerating predictions and confusion counts that stay local, and retraining is not bitwise reproducible across platforms.
10. **RT-CGM provenance.** RT-CGM was admitted by a cadence rule that could not fail, and its first-generation sensors cannot be assigned to subjects.

The author thanks the panel for its reviews.
