# Active experiment results

All runs are retrospective classification of existing labels. No experiment predicts future diagnosis or establishes clinical validity. Local datasets, labels and populations are not harmonized; results are not a cross-modality leaderboard.

## Drawing comparison

The preserved pixel baseline was rerun to `results/finalization_baseline/drawing/` and matched the canonical root metrics. It selects logistic regression on a separate grouped validation partition, then evaluates one 48-image test set. Its candidate groups are filename-derived and unverified.

The controlled comparison uses seed 42, the same candidate-group train/validation/test assignments for every representation/task, and three-fold `StratifiedGroupKFold` on train+validation groups. Candidate choice uses CV ROC-AUC, with balanced accuracy as tie-breaker. No test score is used to choose a model or representation. Because multiple feature/model candidates are compared on these same CV folds, the winning CV mean is selection-optimistic. The grouped test has four candidate groups total (one healthy, three Parkinson's). Since identity is unverified, this is not confirmed unseen-participant evaluation.

An earlier broader draft search was stopped for runtime before completing or producing result metrics. The bounded search documented below completed; no results from the stopped run are reported.

| Evaluation | Representation | Best CV model | CV ROC-AUC, mean ± SD | CV balanced accuracy | Test ROC-AUC | Test balanced accuracy | Test support (healthy, PD) |
|---|---|---|---:|---:|---:|---:|---:|
| Original combined baseline | 32×32 grayscale pixels | Logistic regression (validation-selected) | 0.671 ± 0.022 | 0.625 | 0.494 | 0.428 | 30, 18 |
| Spiral | 32×32 grayscale pixels | RBF SVM | 0.788 ± 0.088 | 0.650 | — | — | — |
| Spiral | Aspect-preserving 64×64 pixels | RBF SVM | 0.749 ± 0.119 | 0.644 | — | — | — |
| Spiral | Hand-crafted image descriptors | RBF SVM | 0.876 ± 0.076 | 0.753 | **0.733** | 0.644 | 15, 9 |
| Spiral | Combined pixels + descriptors | Logistic regression | 0.767 ± 0.043 | 0.692 | — | — | — |
| Wave | 32×32 grayscale pixels | Logistic regression | 0.711 ± 0.087 | 0.644 | — | — | — |
| Wave | Aspect-preserving 64×64 pixels | Logistic regression | 0.616 ± 0.151 | 0.547 | — | — | — |
| Wave | Hand-crafted image descriptors | Random forest | 0.716 ± 0.071 | 0.689 | **0.444** | 0.422 | 15, 9 |
| Wave | Combined pixels + descriptors | Logistic regression | 0.660 ± 0.171 | 0.608 | — | — | — |
| Combined tasks | 32×32 grayscale pixels | Logistic regression | 0.688 ± 0.077 | 0.654 | — | — | — |
| Combined tasks | Aspect-preserving 64×64 pixels | Logistic regression | 0.658 ± 0.102 | 0.624 | — | — | — |
| Combined tasks | Hand-crafted image descriptors | RBF SVM | 0.750 ± 0.063 | 0.681 | **0.554** | 0.467 | 30, 18 |
| Combined tasks | Combined pixels + descriptors | Logistic regression | 0.679 ± 0.094 | 0.654 | — | — | — |

Only the CV-selected candidate for each task was evaluated on the final comparison test set; dashes mean no separate test metric was generated. On this split the combined handcrafted result is numerically higher than the preserved baseline on ROC-AUC (.554 vs .494) and balanced accuracy (.467 vs .428), but balanced accuracy remains below .5 and the test contains only four unverified groups. This is not reliable evidence of improvement. Spiral is higher on this small split; wave is below chance ROC-AUC. Candidate-specific metrics, all CV scores, split assignments and test predictions are in `results/drawing_representation_comparison/`.

## Virtual touchscreen typing

The i-PROGNOSIS pipeline uses one participant-level row after excerpt aggregation. Seed 42 gives 26 training and seven held-out participants; grouped five-fold CV operates only on training participants. Logistic regression and random forest are selected by grouped-CV ROC-AUC. The model is random forest with `max_features=sqrt`, `min_samples_leaf=2`.

| Measure | Verified value |
|---|---:|
| Cohort | 33 participants: 15 controls, 18 PD |
| Excerpts | 338 |
| Best grouped-CV ROC-AUC | Logistic regression 0.689; random forest 0.733 |
| Test support | 3 controls, 4 PD (7 participants total) |
| Accuracy | 0.857 |
| Balanced accuracy | 0.875 |
| PD precision / recall / F1 | 1.000 / 0.750 / 0.857 |
| ROC-AUC | 1.000 |
| Confusion matrix (true rows; control, PD columns) | `[[3, 0], [1, 3]]` |

The held-out result is reproduced but very unstable at seven participants. It does not establish external validation, prospective prediction, clinical readiness or robust generalization. The source cohort is single-study and potential confounders remain.

## Reproducibility and artifact paths

- Drawing pixel baseline: `models/finalization_baseline/drawing_image_model.joblib`; `results/finalization_baseline/drawing/drawing_training_metrics.json` and `drawing_candidate_results.csv`.
- Drawing comparison: selected per-task bundles under `models/drawing_representation_comparison/`; metrics, CV comparison, fold/split assignments, test predictions and report under `results/drawing_representation_comparison/`.
- Typing rerun: `models/finalization_baseline/typing_iprognosis_model.joblib`; participant features, excerpt audit, split assignments, grouped-CV table, test predictions, config, metrics and report under `results/finalization_baseline/typing/`.

Seeds and participant/candidate-group split identities are recorded in the JSON/CSV outputs. The typing features and participant IDs are local derived data and should not be redistributed casually. Existing pre-finalization result and model files were preserved.
