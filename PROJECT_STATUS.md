# Project status

## Scope

The repository contains three separate exploratory Parkinson's research classification experiments: drawing images, pen trajectories, and naturalistic Tappy Keystroke records. No verified participant crosswalk or harmonized target supports merging. No results establish diagnosis or future risk prediction.

## Completed work

- Preserved existing drawing and pen pipelines, notebooks, outputs, and metrics.
- Implemented Tappy event parsing, quality audits, participant-level feature aggregation, a participant holdout, model comparison, evaluation, and inference helper.
- Recorded an optional minimum-2,000-event sensitivity experiment.
- Added a documentation package describing the actual code and artifacts.

## Recorded findings

| Experiment | Selected model | Held-out result |
|---|---|---|
| Drawing | Logistic regression, C=.01 | 48 images, 4 provisional groups; ROC-AUC .494, balanced accuracy .428; confusion `[[9,21],[8,10]]` |
| Pen | Logistic regression, C=10 | 22 task samples, 4 provisional groups; ROC-AUC .920, balanced accuracy .929; confusion `[[8,0],[2,12]]`; only one PD group in test |
| Tappy primary | Gradient boosting (.1 learning rate, depth 2, leaf 8, 100 trees) | 55 participants (14 controls,41 self-reported PD); ROC-AUC .423 (95% bootstrap .244-.599), AP .710 vs prevalence .745, BA .498, sensitivity .854, specificity .143; confusion `[[2,12],[6,35]]` |
| Tappy >=2,000 events | Logistic regression, C=10 | 33 participants (8 controls,25 PD); ROC-AUC .390 (.155-.655), AP .689 vs prevalence .758, BA .425, sensitivity .600, specificity .250; confusion `[[2,6],[10,15]]` |

The Tappy model is weak on held-out data and does not outperform the all-positive baseline on ROC-AUC, average precision, or balanced accuracy. Image and pen group tokens are provisional filename-derived identifiers; results are highly uncertain.

## Limitations and remaining work

- Tappy profile status is self-reported. UPDRS is unavailable/unknown for all controls and 154/162 PD profiles.
- Tappy is uncontrolled naturalistic typing, not standardized finger tapping. Age, medication, tremor, severity, device and typing exposure can confound.
- Small held-out cohorts; no external validation, prospective follow-up, calibrated probabilities, fairness assessment, or confounder adjustment.
- No verified shared participants or compatible targets across sources.
- Only `tests/test_tappy_data.py` exists (three tests); drawing and pen trainers lack unit tests.
- `data/*` is Git-ignored. The local inventory referenced `Archived-Data.zip`, which is absent; expanded event files and profile archive exist.
- Git has no commits and project files are untracked in this checkout.

## Reproducibility

Recorded Tappy primary and sensitivity commands completed, with model bundles, metrics, candidate tables, feature definitions, test predictions, plots and reports. In this documentation pass, all three Tappy unit tests passed, `compileall` completed without errors, all three trainer `--help` commands were checked, and a saved-model smoke prediction on a synthetic all-zero 66-feature row returned class `1`. That synthetic check confirms artifact loading and code execution only; full feature extraction over all raw records was not completed during this pass. Documentation verification commands are listed below. Do not rerun against default output paths if preserving current artifacts matters; choose fresh paths.

Environment observed: Python 3.14.6, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, Matplotlib 3.11.2, Pillow 12.3.0. `requirements.txt` pins dependencies; broader compatibility is not verified.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q src tests
.venv\Scripts\python.exe -m src.train_tappy_models --model-path models/verify_tappy.joblib --results-dir results/verify_tappy
```

See [README](README.md) for setup, data layout and all experiment commands.
