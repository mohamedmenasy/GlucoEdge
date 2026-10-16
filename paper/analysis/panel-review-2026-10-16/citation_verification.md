# Citation verification log (Task 2, panel revision)

Date: 2026-10-16. Scope: the 30 works listed in Task 2 of
`docs/superpowers/plans/2026-10-16-paper-panel-revision.md`. Reviewer-supplied
details (domain, perspective, methodology and EIC cards; synthesis R2, R3, R15,
R16, S23, S24, S33) were treated as untrusted and re-derived from primary
records.

**Method.** Bibliographic fields come from the Crossref API record of each DOI,
cross-checked against the publisher or index page named in the "Verified via"
column: PubMed E-utilities, PMC full text, the arXiv API, PMLR, CEUR-WS, ACL
Anthology, CVF Open Access and OpenAlex. DBLP, OpenReview and IEEE Xplore block
automated access, so they were not used, and nothing rests on them alone.
Searches used only titles, authors and topics. The content lines rest on each
work's abstract, plus full text where noted.

**Result.** 30 of 30 works were verified and appended to `paper/references.bib`.
The 350 existing lines are byte-identical. 0 works were already present, and
0 are UNVERIFIED. As a render check, a scratch `tectonic` build citing all 30
keys with `unsrtnat` gave 0 BibTeX warnings and 30 `\bibitem`s. The file now
has 66 entries, no duplicate keys or DOIs, and balanced braces.

Existing keys reused: none. Two existing keys look similar but are different
works: `nagel2020adaround` (AdaRound) is not `nagel2021whitepaper`, and
`facchinetti2011index` (J index) is not `facchinetti2014sensor`.

## Per-work record

| # | Key | DOI / URL | Verified via | Correction to reviewer-supplied details | What it shows (cite it for) |
|---|-----|-----------|--------------|------------------------------------------|-----------------------------|
| 1 | `williams2024calibration` | 10.18653/v1/2024.acl-long.544 | Crossref; ACL Anthology | None (ACL 2024 Long, pp. 10100–10118, DOI match) | First broad empirical study of calibration data in LLM post-training quantization and pruning; finds "substantial variations in downstream task performance", contrary to assumed robustness. Cite for: calibration-data choice is a known PTQ lever. |
| 2 | `yu2021crossdomain` | 10.1109/CVPRW53098.2021.00339 | Crossref; CVF Open Access (ECV workshop); arXiv 2105.07331 | Pages: reviewer's 3043–3052 is the CVF open-access pagination. The DOI (IEEE) record has 3037–3046, which the entry uses. The reviewer gave no DOI. | Out-of-domain calibration sets give stable PTQ accuracy (10 tasks, 13 calibration sets); accuracy correlates with Gram-matrix similarity between source and calibration domains. Cite for: calibration statistics matter more than domain identity. |
| 3 | `cai2020zeroq` | 10.1109/CVPR42600.2020.01318 | Crossref; CVF Open Access; arXiv 2001.00281 | Pages: reviewer's 13169–13178 is CVF pagination. The DOI record has 13166–13175 (used). | Zero-shot (data-free) quantization: optimizes a "Distilled Dataset" matching the network's batch-norm statistics, then sets ranges and mixed precision with no training or validation data. Cite for: range setting needs the right statistics, not real samples. |
| 4 | `krishnamoorthi2018quantizing` | arXiv:1806.08342 | arXiv API; full text | None (arXiv-only) | INT8 PTQ (per-channel weights, per-layer activations) within 2% of float across CNNs; QAT closes to 1%. The text notes activations need "calibration data" for their dynamic ranges. Cite for: standard INT8 PTQ practice. |
| 5 | `nagel2021whitepaper` | arXiv:2106.08295 | arXiv API; full text | None (arXiv-only) | PTQ and QAT pipelines; PTQ usually suffices for 8-bit near-float accuracy. §3.1 "Quantization range setting" covers clipping thresholds and the small calibration set. Cite for: range setting from calibration data. |
| 6 | `wu2020integer` | arXiv:2004.09602 | arXiv API (no journal-ref or DOI on record); full text | Reviewer named only "Wu et al. 2020 empirical evaluation of integer quantization". Resolved to this arXiv-only paper. | Scale/zero-point math plus evaluation across vision, speech and language. Compares three activation calibration methods (max, entropy/KL, percentile). An 8-bit workflow stays within 1% of float on all networks studied. Cite for: calibration-method choice. |
| 7 | `zhu2023iomt` | 10.1109/JIOT.2022.3143375 | Crossref; OpenAlex abstract | Year: reviewer's "2022" is the early-access date (online 2022-01-14). The issue is vol. 10, no. 5, pp. 3706–3719, **2023**, so the key is `zhu2023iomt` and the paper should say "Zhu et al. (2023)". | Attention-based evidential RNN embedded in a wearable IoMT device whose low-power SoC does Bluetooth and edge computing for real-time glucose prediction and predictive hypoglycemia detection; evaluated on 3 clinical datasets, 47 T1D subjects. Cite for: prior on-device glucose prediction with device evidence (Table 1 row). |
| 8 | `marling2020ohiot1dm` | https://ceur-ws.org/Vol-2675/paper11.pdf | CEUR-WS Vol-2675 index and PDF text | None (CEUR issues no DOI, so URL used) | Documents OhioT1DM: 8 weeks of CGM, insulin, physiological-sensor and self-reported data for each of 12 people with T1D (6 released 2018, 6 more 2020). Cite for: the OhioT1DM dataset. |
| 9 | `kovatchev2004cgega` | 10.2337/diacare.27.8.1922 | Crossref; PubMed 15277418 (full title) | None (Crossref omits the subtitle; full title taken from PubMed) | Introduces continuous glucose-error grid analysis (CG-EGA), which scores CGM accuracy on point values and on the direction and rate of glucose change. Cite for: rate/direction accuracy is an established CGM evaluation target. |
| 10 | `zanderigo2007prediction` | 10.1177/193229680700100508 | Crossref title search; PubMed | Reviewer gave no DOI. Note that the adjacent DOI …100507 is a different paper. | Applies CG-EGA to glucose *predictions* (polynomial and AR models, 30/45-min horizons, 28 T1D, GlucoDay sensor) to grade point and dynamic accuracy. Cite for: rate-aware evaluation of glucose predictors. |
| 11 | `facchinetti2014sensor` | 10.1109/TBME.2013.2284023 | Crossref; PubMed | Reviewer gave no pages; added 620–629 | CGM sensor-error model: blood-to-interstitial diffusion, linear time-varying calibration/drift error and **autoregressive** additive noise, identified on 36 datasets with 4 simultaneous Dexcom SEVEN Plus sensors. Cite for: a published sensor-error model. Its noise is colored, not the independent σ = 1/2 mg/dL noise this paper used. |
| 12 | `lin2017focal` | 10.1109/ICCV.2017.324 | Crossref; CVF Open Access; arXiv 1708.02002 | Pages: reviewer's 2980–2988 is CVF pagination. The DOI record has 2999–3007 (used). | Focal loss reshapes cross-entropy to down-weight well-classified examples under extreme class imbalance. Cite for: the focal-loss sweep variant. |
| 13 | `bai2018tcn` | arXiv:1803.01271 | arXiv API | None (arXiv-only) | A generic temporal convolutional network outperforms canonical recurrent networks such as LSTMs across sequence-modeling benchmarks. Cite for: the TCN architecture. |
| 14 | `loshchilov2017sgdr` | arXiv:1608.03983 (ICLR 2017) | arXiv API (comment "ICLR 2017 conference paper"); ICLR proceedings copy header via web search | None. The URL is arXiv because the OpenReview link could not be fetched; this follows the file's `kingma2015adam` ICLR precedent. | Warm-restart SGD with cosine annealing of the learning rate. Cite for: the cosine learning-rate schedule. |
| 15 | `shi2023corn` | 10.1007/s10044-023-01181-9 | Crossref; arXiv 2111.08851 record (journal-ref and DOI point to this version) | The domain card gave arXiv:2111.08851 (2021). The peer-reviewed version is used: *Pattern Analysis and Applications* 26(3):941–955, 2023. | CORN: rank-consistent ordinal regression via conditional training subsets and the chain rule, without CORAL's weight-sharing constraint. Cite for: the CORN ordinal head. |
| 16 | `wortsman2022soups` | https://proceedings.mlr.press/v162/wortsman22a.html | PMLR page (meta and abstract); arXiv 2203.05482 for Gadre's full name | None (PMLR 162:23965–23998). The entry lists 11 authors as 3 + "others", matching the file's style. | Averaging the weights of models fine-tuned from one pre-trained model, which "lie in a single low error basin", improves accuracy at no inference cost. Cite for: soups assume a shared pre-trained starting point, so independently seeded runs fall outside that regime. |
| 17 | `frankle2020linear` | https://proceedings.mlr.press/v119/frankle20a.html | PMLR page | Reviewer's "D. M. Roy": PMLR lists "Daniel Roy" (used) | Even runs from one initialization end in linearly connected minima only once training is "stable to SGD noise", which happens early in training for large models and at initialization only in small settings such as MNIST. Cite for: why naively averaging separately trained runs can collapse. |
| 18 | `yang2021imbalanced` | https://proceedings.mlr.press/v139/yang21m.html | PMLR page | Reviewer's "Y.-C. Chen": PMLR lists "Yingcong Chen" (used) | Defines deep imbalanced regression (continuous targets with rare ranges) and proposes label/feature distribution smoothing. Cite for: regression on imbalanced targets underperforms in rare ranges and is pulled toward dense ones. |
| 19 | `cameron2008bootstrap` | 10.1162/rest.90.3.414 | Crossref; NBER t0344 abstract | No bibliographic correction. Scope: the paper shows that standard cluster-robust tests over-reject with few (5–30) clusters and that cluster bootstrap-t refinements restore near-nominal size. It does not directly evaluate percentile-bootstrap intervals. | Cite for: few-cluster inference is anti-conservative, which supports withholding or replacing percentile CIs at 6 clusters (R15), not a measured coverage figure. |
| 20 | `mitchell2019modelcards` | 10.1145/3287560.3287596 | Crossref; arXiv 1810.03993 | None | Model cards: short documents accompanying trained models that state intended use, report benchmarked evaluation across conditions and groups, and discourage unsuitable uses. Cite for: `MODEL_CARD.md` (S33). |
| 21 | `battelino2019timeinrange` | 10.2337/dci19-0028 | Crossref; PMC6973648 full text (range definitions) | None. Note that the reviewer's 4-band scheme (<70, 70–180, 181–250, >250) merges the consensus's two below-range levels. | International consensus on CGM metrics and targets. Ranges: <54 (TBR level 2), 54–69 (TBR level 1), 70–180 (TIR), 181–250 (TAR level 1), >250 mg/dL (TAR level 2). Cite for: the band edges in `level_stratified_recall`. |
| 22 | `jdrf2008cgm` | 10.1056/NEJMoa0805017 | Crossref; PubMed 18779236 | Group author per PubMed/NEJM byline (Crossref lists none) | RCT of 322 adults and children on intensive therapy (HbA1c 7.0–10.0%), CGM vs meter, 26 weeks; HbA1c benefit only at ≥25 years. Source trial of DiaData's RT-CGM cohort: mixed DexCom SEVEN, MiniMed Paradigm and FreeStyle Navigator sensors per `diadata_audit_r1.json`. |
| 23 | `wadwa2023pedap` | 10.1056/NEJMoa2210834 | Crossref; PubMed 36920756 | None | PEDAP: 13-week multicenter RCT, children aged 2 to under 6, 2:1 hybrid closed-loop vs standard care (pump or injections plus CGM); primary outcome time in range 70–180 mg/dL. Source of the PEDAP cohort. |
| 24 | `brown2019closedloop` | 10.1056/NEJMoa1907863 | Crossref; PubMed 31618560 | None | iDCL Protocol 3 (DCLP3): 6-month RCT, 168 patients aged 14–71, 2:1 closed-loop vs sensor-augmented pump; primary outcome time in range. Source of DiaData's "DLCP3" cohort (Dexcom G6 per audit). |
| 25 | `aleppo2017replacebg` | 10.2337/dc16-2482 | Crossref (abstract) | None | REPLACE-BG: noninferiority RCT at 14 T1D Exchange sites; adults ≥18 on pumps with HbA1c ≤9.0%; CGM without vs with confirmatory meter checks. Source of the REPLACE-BG cohort (Dexcom G4 with 505 software per audit). |
| 26 | `laffel2020city` | 10.1001/jama.2020.6940 | Crossref; PubMed 32543683 | Crossref gives only the first page; end page 2396 and the subtitle come from PubMed | CITY: RCT of 153 participants aged 14–24 (HbA1c 7.5–10.9%) at 14 US practices, CGM vs meter 1:1; primary outcome HbA1c change. Source of the CITY cohort (Dexcom G5 per audit). |
| 27 | `pratley2020wisdm` | 10.1001/jama.2020.6928 | Crossref; PubMed 32543682 | End page 2406 from PubMed (Crossref gives only the first page) | WISDM: RCT of 203 adults aged ≥60 at 22 US practices, CGM vs meter 1:1; primary outcome % time <70 mg/dL over 6 months. Source of the WISDM cohort (Dexcom G5 per audit). |
| 28 | `sence2021cgm` | 10.2337/dc20-1060 | Crossref; PubMed 33334807 abstract | Primary paper as requested: the 6-month RCT whose primary analysis is time in range. PubMed lists only the group author. Issue year 2021 (online 2020-12-17). | SENCE: RCT of 143 children aged 2 to under 8 at 14 centers comparing CGM plus family behavioral intervention, CGM alone, and meter. Time in range did not improve; time <70 mg/dL fell with CGM. Source of the SENCE cohort (Dexcom G5 per audit). |
| 29 | `dubosson2018d1namo` | 10.1016/j.imu.2018.09.003 | Crossref; Zenodo D1NAMO record (10.5281/zenodo.5651217) | None | Open multi-modal wearable dataset (ECG-centred, with CGM glucose) from 20 healthy subjects and 9 with T1D, built for non-invasive glycemic-event detection. Source of GlucoBench's `dubosson` cohort. |
| 30 | `hall2018glucotypes` | 10.1371/journal.pbio.2005143 | Crossref; PubMed 30040822 | None | CGM-defined "glucotypes": even people normoglycemic by standard tests spend 15% / 2% of time in prediabetic / diabetic ranges. Source of GlucoBench's `hall` cohort. |

## UNVERIFIED

None of the Task 2 works.

The reviewer cards also name works outside Task 2. These were **not verified and
not added**, so do not cite them unless they are verified later:
- Aleppo et al., J Endocr Soc 2017 (trend arrows, adults)
- Laffel et al., J Endocr Soc 2017 (trend arrows, youth)
- Kudva et al., J Endocr Soc 2018
- Basu et al., Diabetes 2013
- Mensh et al., JDST 2013
- Danne et al., Diabetes Care 2017
- Collins et al., BMJ 2024 (TRIPOD+AI)
- Perdomo et al., ICML 2020
- 21 CFR 862.1355
- any source for ordinal label smoothing or the distance-weighted cross-entropy penalty (S23)

## Notes for writers

- **Page numbers.** The three IEEE/CVF conference entries carry the IEEE DOI,
  so they use the DOI record's pages. The CVF open-access pages the reviewers
  quoted differ by a few pages.
- **Cohort mapping.** The cohort-to-trial mapping (CITY, DLCP3, PEDAP,
  REPLACE-BG, RT-CGM, SENCE, WISDM) matches the `citation` fields in
  `paper/results/diadata_audit_r1.json`.
- **DiaData README.** `../DiaData/README.md` gives the same Jaeb URL
  (`public.jaeb.org/dataset/537`) for both SENCE and the Severe Hypoglycemia
  (Weinstock) dataset. Do not copy that URL for SENCE without checking it.
- **Label-noise wording (R16).** If a writer cites `facchinetti2014sensor`, the
  sentence should say the paper's own perturbation was independent white noise,
  whereas the published model is autoregressive.
