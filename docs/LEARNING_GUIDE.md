# Learning guide

The Tappy path shows the full journey from event data to a saved estimator. The other two pipelines follow the same broad pattern with different input features.

## Examples, labels and cleaning

A raw Tappy row is one recorded key event. Millions of events are not millions of independent people. The loader summarizes each month, then aggregates to one row per participant. This is appropriate for a participant-level question and prevents a participant's months being spread across partitions.

The target is `0=control`, `1=self-reported Parkinson's`. Model features are timing summaries and key/hand fractions. User IDs, target, dates, timestamps, event counts, and profile demographics/medication are excluded as predictors.

The parser counts duplicate rows and rejects invalid timing entries. Negative hold and latency are rejected; negative flight is retained. Invalid timestamps are audited without discarding an otherwise valid timing row. Data cleaning is a documented rule, not a blanket deletion of imperfect records.

## Feature summaries

Mean is the arithmetic average; standard deviation measures spread; median is the middle observation; q90 is the 90th percentile; IQR is q75 minus q25; coefficient of variation is standard deviation divided by mean. These summarize typical timing and variability. Hand/direction fractions summarize event composition. They are measurements of uncontrolled typing, not disease-specific biological markers.

## Keeping participants separate

If one person's records appear in both training and testing, the model could recognize a person rather than generalize. Tappy aggregates to participants before a fixed-seed stratified 75/25 split. Training-only cross-validation (CV) compares settings; the held-out test does not select them. The primary split is 162 train and 55 test participants.

## Preprocessing

Median imputation fills missing features using a median learned from the current training fold; missingness indicators record which values were absent. Standard scaling recenters/rescales numeric features, helping logistic regression and SVM. sklearn `Pipeline` ensures held-out CV rows do not influence these learned values.

## Model intuition

- **Logistic regression:** combines weighted inputs and passes the sum through a sigmoid-shaped function to make a score. Regularization parameter `C` controls penalty strength; smaller C means stronger constraint.
- **RBF SVM:** finds a separating boundary that can curve according to similarity between examples. `C` sets the penalty for errors; `gamma` controls how local each example's influence is.
- **Random forest:** averages votes from many decision trees trained on resampled rows/features. Tree count, features per split, and minimum leaf size affect complexity.
- **Gradient boosting:** adds shallow trees sequentially to correct earlier errors. Learning rate shrinks each correction; number/depth of trees and minimum leaf size control flexibility.
- **Prior baseline:** predicts the most common class, exposing performance that comes from imbalance alone.

## Metrics and interpretation

Confusion matrix `[[2,12],[6,35]]` means 2 controls correctly called control, 12 controls called positive, 6 PD participants called control, and 35 PD participants called positive. Sensitivity is 35/41=.854; specificity is 2/14=.143, so many controls are incorrectly flagged.

Accuracy is fraction correct and can mislead on imbalanced data. Balanced accuracy averages sensitivity and specificity. ROC-AUC measures score ranking over thresholds; .5 is chance-like. Average precision summarizes precision/recall and should be compared with positive prevalence (.745 here). F1 combines positive precision and recall but omits true negatives. Tappy score is not a calibrated probability: .8 does not mean an 80% chance. Bootstrap intervals resample held-out participants within classes; they do not cover external-population or model-selection uncertainty.

## Reading artifacts

JSON files record split counts, candidates, metrics, configuration and limitations. Candidate CSVs show model comparisons. Tappy test prediction CSV has one row per held-out participant with true/predicted class, score and error flag. PNG plots show ROC/PR, confusion matrix, and fold scores. The joblib bundle stores the fitted pipeline and required feature order. Reproducible commands and expected result summaries appear in the README and [EXPERIMENTS](EXPERIMENTS.md).

The inference helper returns a DataFrame with `prediction` and `decision_score`, plus `user_key` when that column was supplied. A synthetic all-zero feature row can verify that the serialized model loads and produces a prediction; it is only a software smoke check, not a meaningful example or a measured participant result.

## Observations and hypotheses

**Observed:** primary Tappy held-out ROC-AUC .423, balanced accuracy .498, specificity .143; ROC and average precision (.710) do not exceed the simple prevalence baseline. Training fold performance was much stronger than validation for the chosen boosting model.

**Hypotheses only:** confounding, typing/device differences, sparse controls, or cohort size may contribute. These experiments do not isolate those explanations. Testing them requires verified labels, a larger external cohort and predeclared analyses.

## Glossary

| Term | Meaning |
|---|---|
| Feature | A measured/calculated input to a model |
| Target | The outcome the model is asked to predict |
| Leakage | Held-out information influencing model fitting or selection |
| Imputation | Filling missing values using a training-learned rule |
| Hyperparameter | Model setting selected outside ordinary fitting |
| Cross-validation | Repeated train/validation folds within training data |
| Holdout | Data reserved for final evaluation |
| Class imbalance | Unequal numbers of examples per class |
| ROC-AUC | Threshold-independent ranking metric |
| Average precision | Summary of precision/recall behavior |
| Calibration | Whether predicted probabilities match observed frequencies |
| Generalization | Performance on new participants or settings |
