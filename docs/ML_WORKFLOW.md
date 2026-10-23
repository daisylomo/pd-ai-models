# Active machine-learning workflow

## Spiral and wave drawings

`src.train_drawing_models` builds a labeled image manifest, checks duplicate hashes, derives class-scoped `V##` candidate groups and retains the original 32×32 pixel baseline. It partitions groups into train/validation/test and selects among logistic regression, RBF SVM and random forest using grouped CV plus validation. The preserved baseline test remains 48 images from four candidate groups.

`src.train_drawing_comparisons` compares that pixel representation with aspect-preserving 64×64 grayscale, image-derived foreground/shape/radial/orientation descriptors, and a combined descriptor. It evaluates spiral, wave and combined-task subsets on the same outer candidate-group assignments. CV ROC-AUC chooses model/representation with balanced accuracy as tie-breaker; test is used only after selection. Fixed per-image transforms are cached, while `StandardScaler` is learned inside each CV fit. Since `V##` identity and cross-task mapping are not source-verified, all findings remain candidate-group exploratory results.

## Virtual touchscreen typing

`src.iprognosis_data` parses the documented `Press,<ms>,Release <ms>,<pressure>` event lines; joins S## folders and S##_TEX## names to the clinical CSV `Subject ID`; maps `Control=0` and `PD=1`; audits malformed records, reversed timestamps and pressure values outside the documented range; and extracts excerpt timing, pause and pressure summaries. Excerpt features are median-aggregated to a single participant row. Event and excerpt counts are audit-only.

`src.train_typing_models` uses a fixed-seed stratified participant holdout (26 train, 7 test) and five-fold grouped CV on the training participants. Logistic regression and random forest are selected by CV ROC-AUC. Imputation, missingness indicators and logistic-regression scaling are inside sklearn pipelines, so fold-validation and test rows do not fit preprocessing. The held-out set is too small for stable estimates.

## Outputs and interpretation

Drawing comparison artifacts include CV candidate scores, fold/split assignments, selected test predictions, metrics JSON, report and serialized pipelines. Typing outputs include participant and excerpt audit tables, split/CV assignments, test predictions, model config, metrics, report and pipeline. Use a new output directory for each run; never overwrite the preserved baseline artifacts when conducting another comparison.

Both tasks classify existing group status. Neither evaluates future disease conversion, clinical screening, or prospective risk. There is no external validation, and no cross-modal fusion is justified.
