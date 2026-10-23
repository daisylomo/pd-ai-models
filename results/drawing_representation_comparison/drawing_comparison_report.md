# Drawing representation comparison

Exploratory results only. Filename-derived groups are unverified participant candidates.
The grouped test set contains four candidate groups; do not infer reliable participant-independent generalization.

## Cross-validation comparisons

| Task | Representation | Model | Grouped CV ROC-AUC (mean ± SD) | Balanced accuracy | F1 | Selected |
|---|---|---|---:|---:|---:|---|
| spiral | pixels32_baseline | logistic_regression | 0.766 ± 0.060 | 0.694 | 0.687 |  |
| spiral | pixels32_baseline | rbf_svm | 0.788 ± 0.088 | 0.650 | 0.692 |  |
| spiral | pixels32_baseline | random_forest | 0.724 ± 0.019 | 0.672 | 0.699 |  |
| spiral | aspect_pixels64 | logistic_regression | 0.746 ± 0.030 | 0.706 | 0.707 |  |
| spiral | aspect_pixels64 | rbf_svm | 0.749 ± 0.119 | 0.644 | 0.700 |  |
| spiral | aspect_pixels64 | random_forest | 0.638 ± 0.053 | 0.600 | 0.625 |  |
| spiral | handcrafted | logistic_regression | 0.852 ± 0.026 | 0.783 | 0.794 |  |
| spiral | handcrafted | rbf_svm | 0.876 ± 0.076 | 0.753 | 0.790 | yes |
| spiral | handcrafted | random_forest | 0.844 ± 0.042 | 0.781 | 0.806 |  |
| spiral | combined64 | logistic_regression | 0.767 ± 0.043 | 0.692 | 0.680 |  |
| spiral | combined64 | rbf_svm | 0.761 ± 0.105 | 0.686 | 0.736 |  |
| spiral | combined64 | random_forest | 0.714 ± 0.072 | 0.706 | 0.764 |  |
| wave | pixels32_baseline | logistic_regression | 0.711 ± 0.087 | 0.644 | 0.625 |  |
| wave | pixels32_baseline | rbf_svm | 0.669 ± 0.077 | 0.567 | 0.579 |  |
| wave | pixels32_baseline | random_forest | 0.650 ± 0.078 | 0.619 | 0.649 |  |
| wave | aspect_pixels64 | logistic_regression | 0.616 ± 0.151 | 0.547 | 0.574 |  |
| wave | aspect_pixels64 | rbf_svm | 0.548 ± 0.113 | 0.497 | 0.531 |  |
| wave | aspect_pixels64 | random_forest | 0.553 ± 0.093 | 0.564 | 0.605 |  |
| wave | handcrafted | logistic_regression | 0.637 ± 0.164 | 0.611 | 0.640 |  |
| wave | handcrafted | rbf_svm | 0.663 ± 0.164 | 0.631 | 0.679 |  |
| wave | handcrafted | random_forest | 0.716 ± 0.071 | 0.689 | 0.713 | yes |
| wave | combined64 | logistic_regression | 0.660 ± 0.171 | 0.608 | 0.647 |  |
| wave | combined64 | rbf_svm | 0.562 ± 0.118 | 0.511 | 0.549 |  |
| wave | combined64 | random_forest | 0.558 ± 0.078 | 0.547 | 0.629 |  |
| combined | pixels32_baseline | logistic_regression | 0.688 ± 0.077 | 0.654 | 0.619 |  |
| combined | pixels32_baseline | rbf_svm | 0.669 ± 0.037 | 0.597 | 0.616 |  |
| combined | pixels32_baseline | random_forest | 0.627 ± 0.028 | 0.575 | 0.587 |  |
| combined | aspect_pixels64 | logistic_regression | 0.658 ± 0.102 | 0.624 | 0.649 |  |
| combined | aspect_pixels64 | rbf_svm | 0.615 ± 0.115 | 0.576 | 0.601 |  |
| combined | aspect_pixels64 | random_forest | 0.609 ± 0.044 | 0.619 | 0.660 |  |
| combined | handcrafted | logistic_regression | 0.740 ± 0.044 | 0.662 | 0.675 |  |
| combined | handcrafted | rbf_svm | 0.750 ± 0.063 | 0.681 | 0.720 | yes |
| combined | handcrafted | random_forest | 0.747 ± 0.078 | 0.681 | 0.707 |  |
| combined | combined64 | logistic_regression | 0.679 ± 0.094 | 0.654 | 0.672 |  |
| combined | combined64 | rbf_svm | 0.643 ± 0.119 | 0.590 | 0.616 |  |
| combined | combined64 | random_forest | 0.605 ± 0.055 | 0.581 | 0.603 |  |

## Selected test evaluations

Metrics and supports are saved in `drawing_comparison_metrics.json`.

- **spiral:** handcrafted + rbf_svm; test accuracy 0.667, balanced accuracy 0.644, ROC-AUC 0.733; support {'healthy': 15, 'parkinson': 9}; confusion `[[11, 4], [4, 5]]`.
- **wave:** handcrafted + random_forest; test accuracy 0.500, balanced accuracy 0.422, ROC-AUC 0.444; support {'healthy': 15, 'parkinson': 9}; confusion `[[11, 4], [8, 1]]`.
- **combined:** handcrafted + rbf_svm; test accuracy 0.542, balanced accuracy 0.467, ROC-AUC 0.554; support {'healthy': 30, 'parkinson': 18}; confusion `[[23, 7], [15, 3]]`.

The final test partition was not used in model or representation selection. These evaluations remain candidate-group holdouts, not verified participant holdouts.
