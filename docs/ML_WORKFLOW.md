# Active machine-learning workflow

Use the notebooks as the interactive entry points for the active experiments. They expose dataset checks, intermediate tables, output paths, and the training call in separate cells so you can inspect state and rerun a cell while debugging. Shared parsing, feature extraction, split, and model code remains in `src/`; the notebooks import these tested Python functions rather than duplicating reusable logic in notebook cells.

## 1. Prepare the environment

From PowerShell, open the repository root and create the pinned environment once:

```powershell
Set-Location C:\Users\daisy\Projects\AI\pd-ml-lab
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is disabled, launch Jupyter with `.venv\Scripts\jupyter.exe lab` or run `.venv\Scripts\python.exe -m jupyter lab`. Select the `.venv` Python kernel. Open the repository in JupyterLab and run each notebook from top to bottom. Run training only after the data inspection cell succeeds. Training cells perform preprocessing, CV/model selection, held-out evaluation, and artifact writing; wait for each to finish before moving on.

## 2. Confirm local datasets

Raw data are not included in Git. The notebooks use the defaults in the Python data modules and print/check the resolved paths. Drawing files must have `spiral/` and `wave/` task folders, each with `training/` and `testing/` and `healthy/`/`parkinson/` class folders. Typing data must include `Data/` and `Demographics_Clinical_Characteristics.csv` beneath the i-PROGNOSIS root. See [DATASETS.md](DATASETS.md) and [DRAWING_DATA_QUALITY.md](DRAWING_DATA_QUALITY.md) for exact source details and caveats. The drawing filename prefix such as `V01` is only a provisional grouping token, not a verified participant identifier.

## 3. Run the experiments

Run each notebook independently; no experiment requires a `.joblib` file created by another notebook.

1. **Drawing pixel baseline:** open [`drawing_pixel_baseline.ipynb`](../notebooks/drawing_pixel_baseline.ipynb). Inspect the manifest counts, then run the training cell. The default outputs are `models/finalization_baseline/drawing_image_model.joblib` and `results/finalization_baseline/drawing/`.
2. **i-PROGNOSIS typing:** open [`typing_iprognosis.ipynb`](../notebooks/typing_iprognosis.ipynb). Inspect participant features and the data audit, then run the training cell. Outputs go to `models/finalization_baseline/typing_iprognosis_model.joblib` and `results/finalization_baseline/typing/`.

Use the notebook's path variables to select a different dataset or output location. Baseline and typing defaults can replace files from earlier runs at their output paths; change those variables first if you need to preserve existing artifacts.

## 4. Review completed runs

The final notebook cells list the model and result files. Review the generated metrics JSON and experiment report, followed by candidate/CV tables, split assignments, and held-out predictions for detail. [EXPERIMENTS.md](EXPERIMENTS.md) describes preserved results and limitations. Serialized `.joblib` pipelines are outputs for later reuse, not prerequisites for training. Only load model files from trusted sources.

The drawing evaluation uses unverified filename-derived candidate groups and are exploratory; they do not establish performance on unseen people. Typing is participant-disjoint, but the held-out test contains only seven participants, so its estimates are unstable.

## What the notebooks call

- `src.train_drawing_models` builds the labeled image manifest, checks duplicate hashes, creates grouped partitions, and trains the original 32×32 pixel baseline.
- `src.iprognosis_data` parses events, links participant folders to clinical IDs, audits anomalies, and aggregates excerpt features to one participant row. `src.train_typing_models` performs participant holdout and training-only grouped CV with preprocessing inside sklearn pipelines.

The Python implementations are retained as the reusable source of truth and are also available for command-line runs. Keep model/data logic there when it should be shared by a notebook, tests, or scripts; use notebook cells for interactive inspection, parameter changes, and experiment orchestration.
