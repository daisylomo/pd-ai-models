# Learning guide

## Samples, labels and units

In the drawing experiment one image is an input row, but images may be related by person. The `V##` name token is only a candidate group. Group separation reduces an obvious repeated-image leak but cannot prove independent-participant evaluation.

In the virtual typing experiment, each raw line represents a key press/release event. The loader summarizes each excerpt and then median-aggregates excerpts to one participant row before splitting. This prevents excerpts from the same participant crossing into train and test. The target is the clinical CSV `Group` status, `Control` or `PD`.

## Features

The drawing baseline resizes an image to 32×32 grayscale pixels. That is a simple pixel representation but squares non-square wave drawings. It uses static image appearance and does not encode true movement speed, pauses or temporal smoothness.
Typing features summarize hold durations, between-press intervals, inter-key gaps, long pauses and documented normalized pressure. The features represent performance in the recorded task, not disease-specific biomarkers.

## Keeping related observations together

Cross-validation and the held-out test should separate the independent unit, not merely rows. Typing uses participant IDs. Drawings use class-scoped filename candidates because verified participant identifiers are unavailable; that limitation narrows the claim. The original source `training/` and `testing/` folders overlap on candidate prefixes.

## Preprocessing and model selection

The drawing baseline uses a grouped train/validation/test partition and keeps image vectorization and classifier fitting in its sklearn pipeline. The separate representation-comparison experiment has been removed from the active workflow; its saved reports are historical records only.
Typing uses median imputation with missingness indicators and scaling for linear models, inside a pipeline searched with participant-grouped CV. The held-out test participants do not influence imputation or scaling. Model selection uses training-fold ROC-AUC; the seven-person test set is evaluated once in the isolated rerun.

## Reading metrics

Accuracy is the fraction correctly classified and may hide imbalance. Balanced accuracy averages recall across classes. Precision answers how many predicted positives were positive; recall answers how many actual positives were found; F1 is their harmonic mean. ROC-AUC measures ranking over thresholds and is unstable when test support is tiny. Confusion matrices use true labels as rows and predictions as columns. Scores from these small, unmatched cohorts are exploratory, not clinical performance estimates.

## Claims

The experiments classify existing diagnosis/group labels. They do not predict future disease, establish early detection, diagnose patients, or validate a clinical screening tool. Drawing source identity and licensing remain unresolved; typing has only 33 participants and no external cohort. See [DATASETS](DATASETS.md), [drawing audit](DRAWING_DATA_QUALITY.md), and [EXPERIMENTS](EXPERIMENTS.md).
