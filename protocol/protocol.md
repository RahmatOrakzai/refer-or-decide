# Evaluation protocol (frozen 2026-10-15, before any model results)

This file fixes how every number in the paper is produced. It must not change after results are seen. If a change is unavoidable, it is logged at the bottom with the date and the reason.

## Data

- **Recordings.** 473 verified Pitt Cookie Theft recordings: 233 dementia and 240 control, from 246 participants (147 dementia, 99 control).
  - Transcript 022-1 is missing, so text-based models use 472 recordings.
- **Label.** 1 if the recording is in the verified Dementia folder, 0 if it is in the verified Control folder.
- **Covariates per visit.** Age at visit, years of education, sex, MMSE and CDR, from `derived/pitt_master_recordings.csv`.
- **Text input.** Participant (PAR) turns only. Interviewer (INV) turns are never given to a classifier as input. They are used only to build examiner covariates for adjustment and for the interviewer-only negative control.

## Splits (participant-level; one person never spans two partitions)

| Partition | Participants | Recordings | Use |
|---|---|---|---|
| Locked test | 50 (20 control, 30 dementia) | 95 | Touched once, at the end, for the final table. Defined in `derived/pitt_splits.csv` (`split == test`). |
| Development | 196 | 378 | All model selection, cross-validation and calibration. |

- **Discrimination experiments.** Use `StratifiedGroupKFold(5)` on the development set, repeated with seeds 0–9. The fold assignment is saved in `derived/protocol_folds.csv`.
- **Referral experiments.** Use 200 random participant-level resplits of the full data (seeds 0–199) into train, calibration and test, at 50% / 25% / 25% of participants. This gives the distribution of realised risk. The final referral numbers come from: train on development minus calibration, calibrate on a 40% participant subset of development, then evaluate once on the locked test.
- **Exchangeability.** Calibration and test sets for the referral experiments use **one randomly chosen visit per participant**. All-visit results are reported as secondary.

## Models

| ID | Input | Classifier |
|---|---|---|
| F | 16 participant speech features from the CHAT codes and gold word timings | L2 logistic regression |
| F-adj | F, residualised on age, education, sex and 6 interviewer covariates (residualisation fitted on training controls only) | same |
| T | Frozen sentence encoder over PAR text with pause tokens (Yuan et al. 2020 bins: `,` < 0.5 s, `.` 0.5–2 s, `...` > 2 s), mean-pooled | L2 logistic regression |
| T-adj | T, with its training out-of-fold logit residualised on the same covariates | same |
| FT | Fine-tuned DistilRoBERTa on PAR text with pause tokens (single seed, reference only) | — |
| F+T(-adj) | Late fusion of the F and T logits | logistic regression |

- **Hyperparameters.** The logistic regression C is chosen by inner grouped CV over {0.01, 0.1, 1, 10}. There is no other tuning.

## Negative controls (shortcut ledger)

Each is trained and evaluated exactly like model F:

| Control | Input |
|---|---|
| Demographics | age, education, sex |
| Interviewer | 6 INV covariates |
| Length | participant speech span in seconds |
| Leaky split | model F with recording-level (not participant-level) folds |
| Silence-only audio | acoustic statistics from non-speech frames (Liu et al. 2024). Pending audio access. |

## Metrics

- **Discrimination.**
  - AUC with 95% participant-level bootstrap CI (2000 resamples).
  - Balanced accuracy, and sensitivity and specificity at the threshold chosen on training data (Youden).
  - Paired comparisons by participant-level bootstrap of the AUC difference.
- **Calibration.** Brier score and ECE (10 bins) after Platt scaling fitted on training folds.
- **Strata.** Age at visit (≥ 70 vs < 70) and education (≤ 12 vs > 12 years). Every metric is also reported per stratum.
- **Referral (target α = 0.10, δ = 0.10, plus a curve over α ∈ {0.05, 0.10, 0.15, 0.20}).**
  - (a) Class-conditional and Mondrian (age × education) conformal sets. Refer if the set is not a singleton. Report per-class and per-stratum coverage.
  - (b) Learn-then-Test selective classification. Two thresholds (t_lo, t_hi): control if p < t_lo, dementia if p > t_hi, refer otherwise. Chosen so that error among automatic decisions is ≤ α with probability ≥ 1 − δ (binomial tail p-values, fixed-sequence testing). Done overall and separately per stratum.
  - Report: automatic-decision rate, error among automatic decisions, and sensitivity and specificity among automatic decisions, each per stratum. Also report the fraction of the 200 resplits in which realised risk exceeded α.

## Change log
All changes below were made before any model result was computed.

- 2026-10-15: **T-adj.** T-adj residualises each embedding dimension on the covariates, using the same procedure as F-adj, instead of residualising the logit. This keeps one adjustment method across models.
- 2026-10-15: **Calibration.** No separate Platt step. Logistic regression already outputs a fitted logistic score, and Brier score and ECE are reported on those probabilities.
- 2026-10-15: **Encoders.** Two frozen encoders are reported for T: bert-base-uncased and roberta-base. Each is run with and without pause tokens, as a pause-token ablation.
- 2026-10-15 (after the first referral run, logged as a post-hoc correction): **LTT testing rule.** The first implementation skipped failed tests while fewer than 10 calibration cases were above the threshold. That skip breaks the familywise error guarantee of fixed-sequence testing, so those numbers are discarded. Replaced by a fixed, data-independent threshold grid λ ∈ {0.95, 0.90, …, 0.55} tested either (i) in strict fixed sequence from high to low, stopping at the first non-rejection, or (ii) all at once with Bonferroni (δ/9). Both are valid. Bonferroni is primary because a strict sequence stops at the high thresholds, where few cases fall. The test-set risk at each grid threshold is also saved, so the calibration size that certification would need can be reported.

## Addendum A (2026-10-15, written before any step 4–5 result): minimum-speech rule and external corpora

### Step 4: minimum-speech rule (Pitt only)
- **Inputs.** Participant text truncated to its first k words, k ∈ {10, 20, 30, 50, 75, 100, all}. A transcript shorter than k words is used whole.
- **Model.** T-adj (RoBERTa-base, mean-pooled, with the step-3 residualisation and PCA(64)). It is trained and calibrated on k-word prefixes, so each k is a fixed-length deployment option.
- **Evaluation.** The same 200 resplits as step 3. For each k, report:
  - test AUC;
  - Learn-then-Test (Bonferroni over the λ grid) automatic-decision rate and realised risk at α ∈ {0.10, 0.15, 0.20}.
- **Joint guarantee across k.** The minimum-speech rule picks the smallest k whose certified automatic rate is at least 90% of the full-text rate. Because k is then chosen from the same calibration data, the guarantee across k uses δ/(7·9) (Bonferroni over k and λ). Results at δ/9 per k are also reported.

### Step 5: external corpora (never used for training or model selection)
- **Lu** (DementiaBank English). 28 dementia and 26 control participants, one Cookie Theft recording each. This is the primary external dementia test.
- **Delaware** (DementiaBank English). Participants are MCI or control, across several visits. Only the Cookie Theft segment of each recording is used (from its `@G: Cookie` header to the next `@G`). The labels are the folder labels (MCI = 1). This tests transfer to a milder condition, not the same task.
- **WLS.** There are no diagnosis labels in the available files, so it is not used.
- **Model trained on Pitt.** The primary external model is **T (plain text, no pause tokens)**, trained on the whole Pitt development set.
  - Plain text: Lu has no word-level timings, and the step-2 ablation showed pause tokens add nothing.
  - No adjustment: Lu has no education data.
  - T-adj is secondary, on Delaware only. Delaware education codes are mapped to years: 0→0, 1→8, 2→10, 3→12, 4→14, 5→13, 6→14, 7→16, 8→18, 9→19, 10→20.
- **Referral transfer.** λ is certified on Pitt calibration data (40% of development participants, one visit each), then applied unchanged to the external corpus. Report the realised error among automatic decisions, the automatic rate, and the per-class sensitivity and specificity.
- **Recalibration.** Separately, λ is re-certified on a random half of the external participants and evaluated on the other half, over 200 splits.
- **Reported throughout.** AUC with a participant bootstrap CI. Delaware uses one visit per participant as primary and all visits as secondary.
- **Locked Pitt test.** Still not opened.

## Addendum B (2026-10-16, written before any of these analyses were run): revision for a top-journal submission

### B1. Automatic transcription (ASR) pipeline
- **ASR model.** Whisper via faster-whisper, model `small.en`, int8 on CPU, beam size 5, word timestamps on. Run on all 473 Pitt recordings, resampled to 16 kHz mono.
- **Speaker attribution, two conditions:**
  - (a) *oracle diarization*: an ASR word is assigned to the participant if its midpoint falls inside a PAR utterance's time span in the CHAT file;
  - (b) *no diarization*: every ASR word is used. This is the realistic, fully automatic setting.
- **Text and models.** The same text cleaning as the main-tier text (lower case; fillers kept when Whisper produces them). Models T and T-adj (RoBERTa-base, mean-pooled, PCA(64), logistic regression) are evaluated with exactly the step-2 protocol: 10 × participant-grouped 5-fold CV on development.
- **Referral.** Bonferroni LTT is evaluated with the step-3 protocol (200 resplits).
- **Locked test.** It was opened once already (2026-10-16) for the pre-specified manual-transcript model. The ASR models are evaluated on it once, as a pre-specified secondary analysis, with exactly the same training and calibration procedure and no choices made after seeing locked-test results. This second look is disclosed in the paper.
- **Transcript quality.** WER of ASR against the CHAT main tier (PAR words only, oracle condition) is reported.

### B2. Stronger baseline
- **Model.** Fine-tuned RoBERTa-base on manual participant text (main tier, no pause tokens): max 256 tokens, 6 epochs, learning rate 2e-5, batch 8, weight decay 0.01, no early stopping.
- **Protocol.** One repeat of the same participant-grouped 5-fold CV on development (seed 0, `fold_r0`), so it is compared on identical folds with T. CPU limits prevent ten repeats; this is disclosed.

### B3. ADReSS-comparable result
- **If the ADReSS 2020 train/test speaker lists are available in the user's files:** train on the ADReSS training speakers and test on the ADReSS test speakers, with no tuning on the test speakers, and report accuracy and AUC for T and T-adj next to published ADReSS results.
- **Otherwise:** report that the comparison is not possible, rather than approximating the split.

### B4. Silence-only audio control
- **Features.** Statistics of non-speech frames: frames below an energy threshold of −40 dB relative to the file peak. These are mean, standard deviation and spectral centroid of the silent frames, and the fraction of silence.
- **Classifier and protocol.** Logistic regression with the step-2 protocol, reported as a negative control.

### B5. Figures
Figures show only results already produced by the analyses above:
- risk versus automatic-decision rate curves;
- the calibration size required;
- AUC and certified rate against the number of words;
- external transfer.
- 2026-10-16 (B1 amendment, before any classification on ASR text): **ASR decoding settings.** With default settings, Whisper skipped most of the speech of quiet participants: the interviewer is about 25 dB louder in some recordings. Four decoding settings were compared on 6 control recordings using WER against the manual transcript only, with no diagnostic outcome looked at. The chosen setting is:
  - ffmpeg `speechnorm=e=50:r=0.0001:l=1` loudness normalisation;
  - Whisper's no-speech, log-probability and compression-ratio skip thresholds disabled;
  - no VAD.

  On the 6 test files this gave a WER of 0.41, against 0.47 with the defaults. Silence statistics (B4) are computed on the un-normalised audio.

### B6 (2026-10-16, added before running): separating the transcription shift from the MCI shift
- **Question.** The Delaware transcripts are automatic. Is the Delaware failure caused by MCI or by the transcription?
- **Design.** Rerun the step-5 external analysis with the Pitt training text replaced by Pitt's own Whisper transcripts (B1 oracle condition: participant words only). Everything else is unchanged, including Lu, which stays on its manual transcripts.
- **Interpretation.**
  - If the Delaware AUC and transfer do not improve, the failure is attributed to the population shift (MCI).
  - If they improve, part of the failure is attributed to the transcription mismatch.

## Addendum C (2026-11-02, written AFTER the results above were seen — declared post hoc)

The analyses in this addendum were **not** pre-specified. They were added after the development and
locked-test results were known, in response to a methodological review, and every result from them is
reported as post hoc and hypothesis-generating rather than confirmatory. No model, threshold, split or
preprocessing choice anywhere else in the paper was changed in response to them.

### C1. Severity stratification (post hoc)
Discrimination is reported within MMSE bands (24-30, 20-23, 10-19, <10) and CDR bands (0.5-1, >=2) of
the dementia cases, each scored against all development controls, using the already-computed
out-of-fold predictions. MMSE is available for 396 of 473 recordings and CDR for 391; recordings
without a score are omitted from the corresponding stratum and counted in the caption.

### C2. Prevalence-adjusted predictive values (post hoc)
Sensitivity and specificity at the 0.5 threshold are converted to positive and negative predictive
value across prevalences from 2% to 49% by Bayes' rule. This is a deterministic transformation of
already-reported quantities and introduces no new fitting.

### C3. Class-specific risk control (post hoc)
In addition to the pre-specified pooled error budget, two thresholds are certified against two
separate budgets: the false-omission rate among automatic negative decisions at alpha_FN, and the
false-discovery rate among automatic positive decisions at alpha_FP. Each uses the same fixed lambda
grid and the same Bonferroni LTT construction as the pre-specified analysis, with delta split over the
two families (delta/(2*9)). Resplits, calibration sizes and the one-visit rule are unchanged.
Implemented in `data/derived/scripts/run_referral_classwise.py`.

### C4. Delaware primary analysis (resolved 2026-11-03)
An earlier draft of the manuscript reported the all-recording Delaware AUC as primary, reversing
Addendum A. This was corrected on 2026-11-03: the manuscript now reports one visit per participant as
primary, as Addendum A specifies, with all recordings as secondary. Primary (one visit, after the
identifier correction in C5): 0.622 [0.559, 0.685] for T, 0.579 [0.517, 0.640] for T-adj, 0.596
[0.534, 0.662] for the Whisper-trained model. Secondary (all recordings): 0.615, 0.581, 0.603. No
conclusion depends on the choice.

### C5. Delaware participant identifiers (data-handling correction)
`participant_id` was built as `"D" + <numeric filename prefix>`, which ignores the Control/MCI folder.
23 numeric prefixes occur in both folders, so 22 identifiers collapsed two different people into one
grouping unit, affecting one-visit sampling and the participant bootstrap. Identifiers now include the
folder (`DC...` / `DM...`). Recording-level labels were always folder-correct, so all-recording AUCs
are unchanged; one-visit figures and bootstrap intervals are regenerated.
