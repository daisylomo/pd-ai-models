# Project status

## Research goal and current scope

The long-term goal is early Parkinson's disease risk research. Current experiments are retrospective known-status classifications using drawing images, pen trajectories, and naturalistic Tappy Keystroke records. No current dataset/model establishes a pre-diagnostic outcome, prospective horizon, calibrated risk, or clinical validity. Pipelines remain independent; no verified participant crosswalk or harmonized target supports integration.

## Repository audit

- Three runnable model trainers and two shared source modules exist. The EDA notebook inventories or inspects gait, motion, drawing, pen and Tappy materials; only the three trainer-specific source subsets enter model fitting.
- Raw local data include image/pen data, Tappy monthly records/profile archive, gait C3D/cycle/workbook data, and motion/UDysRS-UPDRS exports. Other data need schema and provenance review before use.
- Tappy splits at participant level and fits imputation/scaling in training folds. Drawing/pen partitions are group-disjoint by filename-derived provisional IDs that require source verification.
- Eight automated tests cover Tappy data, image vectorization, pen feature summaries and group-disjoint splits. No CI or package metadata exists. Dependencies are pinned in `requirements.txt` and verified in the current Windows/Python 3.14.6 environment.
- Git has commit `145ceae initial models`; the worktree was clean at audit. Raw data are ignored.

See [repository audit](docs/REPOSITORY_AUDIT.md) and [dataset inventory](docs/DATASETS.md) for full findings and proposed organization.

## Existing verified results

| Experiment | Selected model | Held-out result |
|---|---|---|
| Drawing | Logistic regression, C=.01 | 48 images, 4 provisional groups; ROC-AUC .494, balanced accuracy .428; confusion `[[9,21],[8,10]]` |
| Pen | Logistic regression, C=10 | 22 task samples, 4 provisional groups; ROC-AUC .920, balanced accuracy .929; confusion `[[8,0],[2,12]]`; only one PD group in test |
| Tappy primary | Gradient boosting (.1 learning rate, depth 2, leaf 8, 100 trees) | 55 participants (14 control,41 self-reported PD); ROC-AUC .423 (95% bootstrap .244-.599), AP .710 vs prevalence .745, balanced accuracy .498, sensitivity .854, specificity .143; confusion `[[2,12],[6,35]]` |
| Tappy >=2,000 events | Logistic regression, C=10 | 33 participants (8 control,25 PD); ROC-AUC .390 (.155-.655), AP .689 vs prevalence .758, balanced accuracy .425, sensitivity .600, specificity .250; confusion `[[2,6],[10,15]]` |

The primary Tappy model is weak on held-out data and does not beat the all-positive baseline on ROC-AUC, average precision or balanced accuracy. The pen result is unstable because only one PD candidate group appears in test. Results are not directly comparable across modalities.

## Changes in this audit pass

- Added `docs/DATASETS.md` and `docs/REPOSITORY_AUDIT.md` for source inventory, data-flow, limitations and a proposed minimal structure.
- Updated README framing to distinguish long-term risk research from current known-status classification.
- Corrected stale Git-state text.
- Added `tests/test_model_features.py` for image conversion, pen geometry, group-token parsing and group-disjoint splits.
- Left model implementations, raw datasets, split rules and canonical model/results untouched.

## Remaining work

1. Verify image and pen group tokens against source metadata; check participant overlap across derivative copies.
2. Consider a consistent experiment metadata and per-test-row output format without changing modality-specific protocols.
3. Audit gait/motion schemas, labels, timing and participant links before deciding whether another pipeline is feasible.
4. Obtain prospective, verified follow-up data and define a risk horizon before describing any model as early-risk prediction.

No shared model, multimodal fusion, external validation or clinical evaluation is currently supported.

## Verification and reproducibility

All eight tests pass (`python -m unittest discover -s tests -v`); `python -m compileall -q src tests` completes; all three trainer `--help` commands work. The drawing, pen and primary Tappy experiments were rerun to isolated output paths under `results/audit_repro/`, with model bundles under `models/audit_repro/`. Selected models, hyperparameters, held-out metrics and (for Tappy) bootstrap intervals exactly matched their canonical JSON summaries. The Tappy run took several minutes to parse the local event records. The optional >=2,000-event sensitivity run was not repeated in this pass.
