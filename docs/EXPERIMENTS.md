# Experiments and recorded results

Values below come from existing metrics JSON and reports. They are not freshly estimated in this document pass. Units and source populations differ, so this is not a cross-modal leaderboard.

## Cohorts

| Source | Local subset | Target/unit | Caveat |
|---|---|---|---|
| Drawings | 204 unique images, 28 candidate groups | Healthy vs Parkinson's; image | `V##` group token provisional; four test groups |
| Pen traces | 40 unique recordings, 110 task samples, 25 candidate groups | Control vs Parkinson's; task sample | Folder labels and filename-derived groups; test includes one PD group |
| Tappy | 217 profiled participants, 558 files, 8,835,813 valid matched events | Control vs self-reported PD; participant | Naturalistic typing, not standardized tapping; UPDRS mostly unavailable |

No shared participant crosswalk, equivalent diagnostic criteria, or equivalent distributions have been established.

## Results

| Run | Selected/CV | Held-out result |
|---|---|---|
| Drawing | Logistic regression C=.01; grouped CV ROC-AUC .671 (SD .022), validation ROC .718 | 48 images/four candidate groups; ROC .494, BA .428, accuracy .396, sensitivity .556, specificity .300; confusion `[[9,21],[8,10]]`. Majority ROC/BA=.5. |
| Pen | Logistic regression C=10; grouped CV ROC .824, validation ROC 1.000 | 22 task samples/four candidate groups; ROC .920, BA .929; confusion `[[8,0],[2,12]]`. Only one PD group in test. |
| Tappy primary | Gradient boosting: learning rate .1, depth 2, leaf 8, 100 estimators; train-fold ROC 1.000, validation .540 (SD .095), AP .778, BA .463 | 55 participants (14 control,41 PD); ROC .423 (95% bootstrap .244-.599), AP .710 vs prevalence .745, accuracy .673, BA .498, sensitivity .854, specificity .143, precision .745, F1 .796; confusion `[[2,12],[6,35]]`. |
| Tappy >=2,000 events | Logistic regression C=10; train-fold ROC .976, validation .498 (SD .077), AP .783, BA .481 | 33 participants (8 control,25 PD); ROC .390 (95% bootstrap .155-.655), AP .689 vs prevalence .758, accuracy .515, BA .425, sensitivity .600, specificity .250, precision .714, F1 .652; confusion `[[2,6],[10,15]]`. |

Tappy's majority baseline predicts every test participant positive. In the primary set its accuracy is .745, balanced accuracy and ROC-AUC .5, average precision .745 and specificity zero. The selected model has near-chance BA, ROC below .5 and AP below baseline. The train/validation gap indicates overfit. The sensitivity run changes cohort composition and is not a correction for confounding.

## Candidate models and search settings

Across all trainers, logistic regression learns a regularized linear boundary, RBF SVM a curved similarity-based boundary, and random forest an average of decision trees. Tappy additionally uses gradient boosting, which adds shallow trees sequentially. Tappy grids: logistic `C=[.01,.1,1,10]`; RBF SVM `C=[.1,1,10]`, `gamma='scale'`; random forest 200 trees, `max_features=['sqrt',.5]`, `min_samples_leaf=[2,5]`; gradient boosting `n_estimators=[100,250]`, `learning_rate=[.03,.1]`, `max_depth=[1,2]`, `min_samples_leaf=[3,8]`. Logistic/SVM use balanced class weights; forest uses `balanced_subsample`; boosting is unweighted. Tappy selects by training-fold mean ROC-AUC.

Drawing uses logistic `C=[.01,.1,1,10]`, RBF SVM `C=[.1,1,10]` and `gamma='scale'`, and a 300-tree balanced-subsample forest with `max_features=['sqrt',.25]`, leaf sizes `[2,4]`. Pen uses the same logistic/SVM grids and a 300-tree balanced-subsample forest with `max_features=['sqrt',.75]`, leaf sizes `[2,4]`. These trainers select using their grouped validation ROC-AUC, with balanced accuracy as configured in their selection rules. All use fixed random seeds in the recorded runs.

## Limits

Tappy labels are self-reported; age, medication, tremor, severity, device, typing habits and exposure may confound. Bootstrap intervals cover resampling of the test participants at fixed class counts, not selection or external uncertainty. Small held-out groups make image/pen results especially unstable. No external validation, prospective outcomes, calibrated probabilities, confounder adjustment, or clinical assessment is available. Do not make diagnostic, clinical sensitivity, or cross-dataset generalization claims.

## Artifact paths

- Drawing: `models/drawing_image_model.joblib`, `results/drawing_training_metrics.json`, `results/drawing_candidate_results.csv`.
- Pen: `models/handwriting_trajectory_model.joblib`, `results/handwriting_training_metrics.json`, `results/handwriting_candidate_results.csv`.
- Tappy primary: model `models/tappy_keystroke_model.joblib`, artifacts `results/tappy_*`.
- Tappy sensitivity: model `models/tappy_keystroke_min2000_model.joblib`, artifacts in `results/tappy_min_events_2000/`.
