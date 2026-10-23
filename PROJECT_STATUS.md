# Project status

## Active research scope

The active experiments are retrospective classification of existing labels for (1) spiral drawings, (2) wave drawings, and (3) i-PROGNOSIS virtual touchscreen typing. The codebase is a research/evaluation repository, not an application or clinical system. None of these experiments predicts future disease onset.

## Verified local results

| Task | Selected configuration | Verified evaluation | Main limitation |
|---|---|---|---|
| Preserved drawing pixel baseline | Logistic regression, 32×32 grayscale | 48-image candidate-group test: accuracy .396, balanced accuracy .428, PD precision .323, recall .556, F1 .408, ROC-AUC .494; confusion `[[9,21],[8,10]]` | Four unverified candidate groups; only one healthy group in test |
| Spiral representation comparison | Hand-crafted descriptors, RBF SVM | Grouped CV ROC-AUC .876±.076; 24-image test ROC-AUC .733, balanced accuracy .644; confusion `[[11,4],[4,5]]` | Candidate identities unverified; 4 groups per task test |
| Wave representation comparison | Hand-crafted descriptors, random forest | Grouped CV ROC-AUC .716±.071; 24-image test ROC-AUC .444, balanced accuracy .422; confusion `[[11,4],[8,1]]` | Weak test result; candidate identities unverified |
| Combined drawing comparison | Hand-crafted descriptors, RBF SVM | Grouped CV ROC-AUC .750±.063; 48-image test ROC-AUC .554, balanced accuracy .467; confusion `[[23,7],[15,3]]` | No test improvement over original pixel baseline; candidate identities unverified |
| Virtual touchscreen typing | Random forest, min leaf 2 | 33 participants (15 controls, 18 PD), 338 excerpts; 7-person test accuracy .857, balanced accuracy .875, PD P/R/F1 1.000/.750/.857, ROC-AUC 1.000; confusion `[[3,0],[1,3]]` | Very small held-out participant set; single study; retrospective labels |

CV and test results above come from isolated reruns in `results/finalization_baseline/` and `results/drawing_representation_comparison/`. The typing dataset default path is corrected to the local directory under `virtual keyboard data/`.

## Evaluation and data limits

- Drawing `V##` tokens are not verified participant IDs. Their overlap across the original `training/` and `testing/` folders prevents treating those folder names as independent-person partitions. Candidate-group results are exploratory.
- Drawing dataset source, licence, citation and clinical acquisition protocol were not found locally.
- Typing splits by linked participant ID, but seven test participants cannot support a stable generalization estimate. Its local ReadMe documents CC BY 4.0 and a required paper citation.
- There is no verified participant crosswalk or external validation. No multimodal fusion or clinical claim is supported.

## Repository contents

Active trainers: `src/train_drawing_models.py`, `src/train_drawing_comparisons.py`, and `src/train_typing_models.py`. Active tests cover drawing features/splits and i-PROGNOSIS parsing/linkage/splits. Historical pen/Tappy source trainers were removed from active source; their saved models/results and raw local data remain preserved. Exploratory notebooks remain historical and are not imported by active training code.

## Verification

The preserved drawing baseline and typing pipeline were rerun with isolated result/model paths. The drawing representation comparison generated per-task models, folds, assignments, predictions, metrics and report. Focused tests and full-suite verification are recorded in the final repository report and README after all changes are checked.
