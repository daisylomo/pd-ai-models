# Tappy Keystroke participant-level experiment

**Scope:** Exploratory participant-level classification of self-reported Parkinsons status from naturalistic typing; not prospective risk prediction or clinical diagnosis

## Cohort and data quality

- Participants: 217 (55 controls; 162 self-reported Parkinson's).
- Monthly recordings: 622 files; {'control': 131, 'parkinson': 427, 'unlabeled': 0}
- Valid matched events used: 8,835,813.
- Excluded duplicate rows: 178,394; UserKey conflicts: 72; invalid timing rows: 592.
- Timestamp audit: invalid dates 61; invalid times 234; filename-month mismatches 0; ordering reversals 9,256.
- Unit of analysis: participant. Each monthly recording is summarized first; a participant feature is the median across their monthly recordings. No participant appears in more than one split.
- Usage counts and profile demographics/medication fields are audit-only and excluded from model predictors.

## Split and model selection

- Split: seed 42, 162 training participants and 55 held-out participants.
- Training labels: 41 controls and 121 Parkinson's; test labels: 14 controls and 41 Parkinson's.
- Model selected by training-only 5-fold CV ROC-AUC: **gradient_boosting**, {'classifier__learning_rate': 0.1, 'classifier__max_depth': 2, 'classifier__min_samples_leaf': 8, 'classifier__n_estimators': 100}.
- Test data were not used for feature or hyperparameter selection.

| Candidate | Train-fold ROC-AUC | Validation-fold ROC-AUC (SD) | Validation-fold PR-AUC | Balanced accuracy | Parameters |
| --- | ---: | ---: | ---: | ---: | --- |
| gradient_boosting | 1.000 | 0.540 (0.095) | 0.778 | 0.463 | `{"classifier__learning_rate": 0.1, "classifier__max_depth": 2, "classifier__min_samples_leaf": 8, "classifier__n_estimators": 100}` |
| random_forest | 0.998 | 0.470 (0.073) | 0.763 | 0.479 | `{"classifier__max_features": 0.5, "classifier__min_samples_leaf": 5, "classifier__n_estimators": 200}` |
| logistic_regression | 0.683 | 0.445 (0.089) | 0.730 | 0.450 | `{"classifier__C": 0.01}` |
| rbf_svm | 0.708 | 0.439 (0.092) | 0.743 | 0.500 | `{"classifier__C": 0.1, "classifier__gamma": "scale"}` |

## Held-out performance

- ROC-AUC: 0.423 (95% bootstrap interval 0.244 to 0.599).
- PR-AUC (average precision): 0.710 (test positive prevalence 0.745; compare to this prevalence baseline).
- Balanced accuracy 0.498; sensitivity 0.854; specificity 0.143; precision 0.745; F1 0.795.
- Confusion matrix, rows=true [control, Parkinson's], columns=predicted [control, Parkinson's]: `[[2, 12], [6, 35]]`.
- All-positive majority baseline: ROC-AUC 0.500; average precision 0.745; balanced accuracy 0.500; specificity 0.000.
- Stratified bootstrap intervals: balanced accuracy 0.390 to 0.606; sensitivity 0.732 to 0.951; specificity 0.000 to 0.357 (2,000 participant resamples).
- ROC-AUC assesses ranking; PR-AUC emphasizes positive-class precision/recall and is sensitive to prevalence. Balanced accuracy gives both classes equal weight. Sensitivity/specificity are included as case/control rates, not clinical performance claims.

## Interpretation and limitations

Labels are self-reported, typing is uncontrolled, and the cohort is modest and imbalanced. Profile variables such as age, gender, disease severity/duration, tremor and medication can confound associations; they are excluded from model input. Usage volume can also reflect health, computer use, or compliance and is excluded as a predictor. This experiment does not establish diagnosis or prospective risk detection.

## Reproduction

```powershell
python -m src.train_tappy_models
python -m unittest discover -s tests -v
```
