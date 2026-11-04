# Refer or Decide: code, splits and protocol

Code for "Refer or Decide: What Risk Certification Can and Cannot Establish for Speech-Based Dementia
Screening". It covers participant-grouped evaluation with shortcut controls, linear nuisance adjustment,
Learn-then-Test (LTT) referral, the calibration-size analysis, the external corpora and the Whisper
pipeline.

**No data is included.** DementiaBank (TalkBank) audio and transcripts are available to registered
members only. Get the Pitt, Lu and Delaware corpora from https://dementia.talkbank.org under its terms
of use. This repository holds only recording identifiers, split assignments, scripts and the protocol.

## Contents
- `protocol/protocol.md`: the pre-registered evaluation protocol. It includes the change log and
  Addenda A–C; Addendum C is declared post hoc.
- `splits/pitt_splits.csv`, `splits/protocol_folds.csv`: participant-level split and fold assignments
  per recording (TalkBank media id). `locked_test` marks the 50 held-out participants, and
  `fold_r0..r9` give the ten repeats of grouped 5-fold CV. Diagnosis labels are left out; they come
  from the corpus metadata.
- `scripts/`: every script used for the paper (see the pipeline below).

## Pipeline (run from a folder holding the derived tables)
1. `build_master.py`, `audio_qc.py`, `group_balance.py`, `make_splits.py`, `make_protocol_folds.py`:
   verified recording table, QC and splits.
2. `extract_chat_features.py`: participant features (F) and interviewer counts from CHAT.
   `make_par_text.py`, `make_main_tier_text.py`: participant text (with and without pause tokens,
   prefixes, main tier). `build_mci_table.py`: Delaware table.
3. `embed_text.py`, `embed_columns.py`: frozen RoBERTa/BERT mean-pooled embeddings.
4. `run_baselines.py`: negative controls, F, F-adj, T, T-adj and fusion (Table I).
   `run_text_cv.py`: the same CV protocol for any extra input. Set `COVARS=demo` to adjust for age,
   education and sex only.
5. `run_referral.py <derived> <out> <model> [--emb=npz:key] [--tag=name] [--kdiv=k] [--covars=demo]`:
   200 participant resplits; conformal and LTT (Bonferroni, fixed sequence, per-education).
   `--reps=a:b` runs a chunk of resplits; `merge_referral_chunks.py <out> <tag>` merges the chunks.
   `run_referral_classwise.py`: separate FOR/FDR budgets (post hoc).
6. `calibration_size.py`: power-based planning approximation. `calibration_size_sim.py`: the same
   with a random number of calibration cases above each threshold and a bootstrap of the risk curve.
7. `run_asr.py`, `make_asr_text.py`: Whisper transcription and ASR text (oracle and no diarisation).
8. `run_external.py`: Lu and Delaware discrimination and certificate transfer.
9. `run_locked_test.py`: one-time locked-test evaluation. It refuses to run twice without `--force`.
   It also writes per-recording predictions and the certified thresholds.
   `locked_and_threshold_analyses.py`: bootstrap intervals, visit-gap analysis and predictive values
   of the certified threshold (post hoc).
10. `severity_prevalence.py`: severity bands and prevalence-adjusted predictive values (post hoc).
    `review_analyses.py`: resplit distributions and participant-level statistics.
11. `make_figures.py`: paper figures from the saved result files.

Seeds: resplit r uses `numpy.random.default_rng(r)`. The locked-test split uses seed 2026, and
bootstraps use seed 0.
