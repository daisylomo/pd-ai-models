# Parkinson's Research ML Lab

Exploratory machine-learning experiments using drawing images, pen trajectories, and naturalistic Tappy Keystroke records. The pipelines are independent: participants, populations, measurement methods, labels, and feature spaces have not been shown to align. Results do not establish diagnosis or prospective risk prediction.

See [architecture](docs/ARCHITECTURE.md), [code walkthrough](docs/CODEBASE_WALKTHROUGH.md), [ML workflow](docs/ML_WORKFLOW.md), [experiments](docs/EXPERIMENTS.md), and [learning guide](docs/LEARNING_GUIDE.md).

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

## Tests and evaluation outputs

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The current three tests cover Tappy data processing only. Results are in `results/*_training_metrics.json`; Tappy also writes candidate and feature tables, test predictions, plots, and a Markdown report. Exact recorded results and caveats are in [EXPERIMENTS](docs/EXPERIMENTS.md).

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
