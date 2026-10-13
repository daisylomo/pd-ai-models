# Parkinson's Research ML Lab

The long-term research motivation is early Parkinson's disease (PD) risk assessment. The current code does **not** estimate early or future risk: it explores retrospective known-status classification using drawing images, pen trajectories, and naturalistic Tappy Keystroke records. The pipelines are independent because populations, labels, measurements, and participant identities have not been shown to align. Results do not establish diagnosis, calibrated risk, or clinical validity.

See [dataset inventory and boundaries](docs/DATASETS.md), [repository audit and proposed structure](docs/REPOSITORY_AUDIT.md), [architecture](docs/ARCHITECTURE.md), [code walkthrough](docs/CODEBASE_WALKTHROUGH.md), [ML workflow](docs/ML_WORKFLOW.md), [experiments](docs/EXPERIMENTS.md), and [learning guide](docs/LEARNING_GUIDE.md).

## Repository map

```text
src/       dataset loaders, feature extraction, and three independent trainers
tests/     Tappy data, image transformation, pen feature, and split tests
data/      local raw data (ignored by Git)
models/    fitted joblib model bundles
results/   experiment metrics, candidate tables, plots, reports, predictions
notebooks/ exploratory repository and data analysis
docs/      dataset guide, architecture, walkthrough, workflow, results, audit
```

## Datasets and research questions

| Dataset/pipeline | Question currently studied | Input representation | Main limitation |
|---|---|---|---|
| Spiral/wave drawings | Separate known healthy vs Parkinson's labels | 32x32 grayscale pixels; image rows | `V##` filename groups are provisional; small held-out set |
| Pen trajectories | Separate control vs Parkinson's folder labels | Geometry, timing-independent path, pressure/grip/z summaries plus task category | Filename group IDs are provisional; only one PD candidate group in test |
| Tappy Keystroke | Associate naturalistic typing summaries with self-reported PD status | Per-person medians of monthly timing, hand and direction summaries | Self-report, confounding, small imbalanced holdout; not standardized tapping |

The local repository also includes gait/C3D/cycle and motion/UPDRS exports, plus derivative drawing/pen folders. Current trainers do not fit models on those materials; see the [dataset inventory](docs/DATASETS.md) before considering them. No verified cross-dataset participant links or harmonized labels support a shared or multimodal model.

Each pipeline compares its own logistic regression, RBF SVM and random forest candidates; Tappy also compares gradient boosting. Their feature spaces and protocols differ. These are retrospective known-status classifiers, not early-risk predictions. Prospective follow-up and a prespecified outcome horizon are needed for the long-term risk objective.

## Setup

The checked environment is Windows with Python 3.14.6. Dependencies are pinned in `requirements.txt`; other Python/platform combinations are unverified.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked, use `.venv\Scripts\python.exe` directly.

## Data

Raw files are local and ignored by Git. Keep originals unchanged. Default paths are defined in source modules.

- Tappy: event files and `Archived-users.zip` under `data/Parkinson's Datasets/tappy-keystroke-data-1.0.0/` (the exact default is `DEFAULT_TAPPY_ROOT` in `src/tappy_data.py`). Files need participant keys and `YYMM` suffixes; parser supports tab/comma rows.
- Drawings: `data/Parkinson's Datasets/Parkinson's drawings dataset/drawings`; override via `--image-root`.
- Pen: `data/Parkinson's UCI dataset/hw_dataset` and `data/Park's UCI kaggle dataset/hw_dataset`.

Only these source subsets enter the existing pipelines. Unused materials under `data/` are not automatically trained.

## Train

Run at repository root:

```powershell
python -m src.train_drawing_models
python -m src.train_handwriting_models
python -m src.train_tappy_models
```

Tappy sensitivity run with a minimum of 2,000 events:

```powershell
python -m src.train_tappy_models --min-events 2000 --model-path models/tappy_keystroke_min2000_model.joblib --results-dir results/tappy_min_events_2000
```

Existing default output paths are overwritten on rerun. Use distinct model/results paths to retain a comparison run. See each command's `--help` for options.

All three commands train and evaluate their own model; evaluation is written alongside the model. For example, a drawing run writes `models/drawing_image_model.joblib`, `results/drawing_training_metrics.json`, and a candidate CSV. Handwriting writes its model bundle and metrics/candidate CSV. Tappy writes a bundle plus metrics/config JSON, candidate and feature-definition CSVs, held-out participant predictions, plots, and a Markdown report. A Tappy sensitivity run writes the analogous files in its selected results directory.

## Recorded results and metric reading

| Experiment | Selected model | Held-out result |
|---|---|---|
| Drawing | Logistic regression, C=.01 | ROC-AUC .494, balanced accuracy .428 on 48 images/four provisional groups |
| Pen | Logistic regression, C=10 | ROC-AUC .920, balanced accuracy .929 on 22 task samples/four provisional groups; only one PD group in test |
| Tappy primary | Gradient boosting, learning rate .1, depth 2, leaf 8, 100 trees | ROC-AUC .423, average precision .710 vs prevalence .745, balanced accuracy .498, sensitivity .854, specificity .143 on 55 participants |
| Tappy >=2,000 events | Logistic regression, C=10 | ROC-AUC .390, average precision .689 vs prevalence .758, balanced accuracy .425 on 33 participants |

ROC-AUC describes ranking over thresholds; average precision summarizes precision/recall and should be compared with positive prevalence. Balanced accuracy averages sensitivity and specificity, useful when class counts differ. Sensitivity and specificity show positive and control recall at the chosen threshold. Tappy bootstrap intervals and confusion matrices are in the metrics JSON and [experiment guide](docs/EXPERIMENTS.md). The results are weak or unstable and do not establish clinical performance.

## Tests and evaluation outputs

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The test suite has eight tests covering Tappy parsing/aggregation, image vectorization, pen feature summaries, and group-disjoint splits. Results are in `results/*_training_metrics.json`; Tappy also writes candidate and feature tables, test predictions, plots, and a Markdown report. Exact recorded results and caveats are in [EXPERIMENTS](docs/EXPERIMENTS.md).

## Tappy inference

```python
from pathlib import Path
from src.tappy_data import build_unlabeled_participant_features
from src.train_tappy_models import predict_participant_features

features, audit = build_unlabeled_participant_features()
print(audit["participants"])
predictions = predict_participant_features(
    Path("models/tappy_keystroke_model.joblib"),
    features.drop(columns=["target"], errors="ignore"),
)
print(predictions.head())
```

The helper checks expected feature columns and returns integer prediction plus an uncalibrated ranking score. `1` means the fitted model predicts the self-reported Parkinson's class; the score is not a disease probability.

## Interpretation

Primary Tappy held-out ROC-AUC was .423, balanced accuracy .498, and specificity .143. The >=2,000-event analysis had ROC-AUC .390 and balanced accuracy .425. Labels are self-reported, test samples are small, and no external validation or clinical evaluation exists. Drawing and pen participant tokens are provisional filename-derived groups. No integrated or multimodal model is supported.
