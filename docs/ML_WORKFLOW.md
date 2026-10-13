# ML workflow

## Load and clean

**Drawings:** `build_manifest()` finds labeled image paths, SHA-256 hashes and removes exact duplicate contents. Labels follow folders; `V##` filenames provide provisional groups.

**Pen traces:** `build_dataset()` scans two local `hw_dataset` trees, parses point rows, hashes duplicate recordings, uses folder labels and derives candidate groups from filenames. Recorded audit: 80 files, 40 unique recordings, 110 task samples, 25 provisional groups.

**Tappy:** `load_profiles()` reads the user ZIP; `build_participant_dataset()` processes 622 event files. It audits profile matching, UserKey conflicts, duplicates, timing validity, dates, timestamps and event order. Recorded exclusions include 63 unmatched files/301,987 rows, 72 mismatched-key rows, 178,394 duplicate rows and 592 invalid timing rows. Invalid date/time and order are reported without dropping otherwise valid timing. Negative hold/latency are excluded; negative flight is retained.

## Features and unit of analysis

Tappy target: `0=control`, `1=self-reported Parkinson's`. One row is one participant. Each month and each hold/latency/flight measure gets mean, population standard deviation, median, q90, IQR and coefficient of variation. Timing summaries are also made for L/R current-key events. L/R/space (`S`) and all nine direction codes become fractions. Monthly values are median-aggregated per participant, so each month has equal weight. Timings use milliseconds; fractions and CV are dimensionless. IDs, target, timestamps, profile covariates, event count and session count are not predictors.

Drawing inputs are 32x32 normalized grayscale values. Pen features summarize path geometry, step lengths, turns, pressure, grip angle, z and task ID. Exact feature code is in `src/train_handwriting_models.py`.

## Split and preprocessing

Tappy seed 42 creates stratified 75/25 participant split: 162 train (41 control,121 PD), 55 test (14,41); five-fold stratified CV uses training only. At least 2,000 events yields 131 participants, split 98/33. No participant crosses partitions.

Drawing partitions contain 114/42/48 images and 19/5/4 candidate groups; grouped 3-fold CV uses training data. Pen partitions contain 66/22/22 task samples and 16/5/4 candidate groups, with grouped 3-fold CV. Its group IDs are not validated participant identities.

Tappy's sklearn `Pipeline` fits median imputation with missingness indicators and (for linear/SVM candidates) standard scaling inside each training fold. This prevents validation data from determining medians or scale. Trees omit scaling. Drawing and pen similarly place feature processing in sklearn pipeline/column-transformer workflows. See source for each fit/refit boundary.

## Models and selection

Tappy compares class-weighted logistic regression, RBF SVM and random forest with unweighted gradient boosting, plus a prior-only baseline. Search grids are saved in JSON configuration. Mean training-fold ROC-AUC chooses the candidate; balanced accuracy and average precision are also reported. Test data are evaluated only after selection. Drawing and pen compare logistic regression, RBF SVM and random forest; their metrics record separate validation selection.

## Evaluation and outputs

ROC-AUC describes ranking across thresholds; average precision summarizes precision/recall and should be compared to positive prevalence. Accuracy may hide class imbalance. Balanced accuracy averages sensitivity and specificity. A confusion matrix gives counts (true labels by rows). Tappy records 2,000 stratified participant bootstrap percentile intervals; these do not include model-selection or external-cohort uncertainty. Scores are not calibrated probabilities.

Tappy writes a joblib model bundle, config and metrics JSON, candidate and feature-definition CSVs, test-prediction CSV, ROC/PR and confusion/CV plots, and Markdown report. `predict_participant_features()` loads the bundle, orders required features, and returns class plus ranking score. Raw event conversion is a separate call to `build_unlabeled_participant_features()`.
