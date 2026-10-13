# Tappy Keystroke participant-level experiment

**Scope:** Exploratory participant-level classification of self-reported Parkinsons status from naturalistic typing; not prospective risk prediction or clinical diagnosis

## Cohort and data quality

- Participants: 131 (33 controls; 98 self-reported Parkinson's).
- Monthly recordings: 622 files; {'control': 104, 'parkinson': 340, 'unlabeled': 0}
- Valid matched events used: 8,835,813.
- Excluded duplicate rows: 178,394; UserKey conflicts: 72; invalid timing rows: 592.
- Timestamp audit: invalid dates 61; invalid times 234; filename-month mismatches 0; ordering reversals 9,256.
- Unit of analysis: participant. Each monthly recording is summarized first; a participant feature is the median across their monthly recordings. No participant appears in more than one split.
- Usage counts and profile demographics/medication fields are audit-only and excluded from model predictors.

## Split and model selection

- Split: seed 42, 98 training participants and 33 held-out participants.
- Training labels: 25 controls and 73 Parkinson's; test labels: 8 controls and 25 Parkinson's.
- Model selected by training-only 5-fold CV ROC-AUC: **logistic_regression**, {'classifier__C': 10.0}.
- Test data were not used for feature or hyperparameter selection.

| Candidate | Train-fold ROC-AUC | Validation-fold ROC-AUC (SD) | Validation-fold PR-AUC | Balanced accuracy | Parameters |
| --- | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.976 | 0.498 (0.077) | 0.783 | 0.481 | `{"classifier__C": 10.0}` |
| rbf_svm | 0.996 | 0.444 (0.162) | 0.759 | 0.469 | `{"classifier__C": 10.0, "classifier__gamma": "scale"}` |
| gradient_boosting | 0.997 | 0.416 (0.116) | 0.732 | 0.497 | `{"classifier__learning_rate": 0.1, "classifier__max_depth": 1, "classifier__min_samples_leaf": 3, "classifier__n_estimators": 100}` |
| random_forest | 1.000 | 0.349 (0.051) | 0.700 | 0.479 | `{"classifier__max_features": 0.5, "classifier__min_samples_leaf": 2, "classifier__n_estimators": 200}` |

## Held-out performance

- ROC-AUC: 0.390 (95% bootstrap interval 0.155 to 0.655).
- PR-AUC (average precision): 0.689 (test positive prevalence 0.758; compare to this prevalence baseline).
- Balanced accuracy 0.425; sensitivity 0.600; specificity 0.250; precision 0.714; F1 0.652.
- Confusion matrix, rows=true [control, Parkinson's], columns=predicted [control, Parkinson's]: `[[2, 6], [10, 15]]`.
- All-positive majority baseline: ROC-AUC 0.500; average precision 0.758; balanced accuracy 0.500; specificity 0.000.
- Stratified bootstrap intervals: balanced accuracy 0.260 to 0.610; sensitivity 0.400 to 0.800; specificity 0.000 to 0.503 (2,000 participant resamples).
- ROC-AUC assesses ranking; PR-AUC emphasizes positive-class precision/recall and is sensitive to prevalence. Balanced accuracy gives both classes equal weight. Sensitivity/specificity are included as case/control rates, not clinical performance claims.

## Interpretation and limitations

Labels are self-reported, typing is uncontrolled, and the cohort is modest and imbalanced. Profile variables such as age, gender, disease severity/duration, tremor and medication can confound associations; they are excluded from model input. Usage volume can also reflect health, computer use, or compliance and is excluded as a predictor. This experiment does not establish diagnosis or prospective risk detection.

## Reproduction

```powershell
python -m src.train_tappy_models
python -m unittest discover -s tests -v
```
