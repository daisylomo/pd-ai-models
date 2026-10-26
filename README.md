# Parkinson’s Digital Motor Research Lab

This repository is a research and evaluation codebase for investigating digital motor characteristics associated with Parkinson’s disease (PD). Its experiments classify existing diagnostic-group labels. It is not a deployed application, diagnostic system, or prospective risk model.

## Research scope

The active work covers two experiments in independent pipelines:

1. Spiral drawing image analysis.
2. Virtual touchscreen typing analysis.

The drawing pipeline trains the preserved 32×32 pixel baseline on spiral and wave images. Typing uses the i-PROGNOSIS touchscreen dataset in a separate participant-level pipeline. No verified cross-dataset participant links support multimodal fusion.

## Current experiment status

| Task | Dataset and representation | Selected model | Verified result | Main limitation |
|---|---|---|---|---|
| Preserved drawing pixel baseline | 204 spiral/wave PNGs; direct 32×32 grayscale pixels | Logistic regression | 48-image test: accuracy .396, balanced accuracy .428, PD precision .323, recall .556, F1 .408, ROC-AUC .494; confusion `[[9,21],[8,10]]` | Four filename-derived candidate groups in test; identity is unverified |
| Virtual touchscreen typing | i-PROGNOSIS; participant medians of timing, pause and pressure features | Random forest | 33 participants (15 controls, 18 PD), 338 excerpts; seven-person test accuracy .857, balanced accuracy .875, PD precision/recall/F1 1.000/.750/.857, ROC-AUC 1.000; confusion `[[3,0],[1,3]]` | Very small single-study test set; retrospective labels |

The drawing test assignment contains four filename-derived candidate groups (one healthy and three PD); `V##` tokens are not verified participants, so its score does not establish generalization to unseen people. The typing result is participant-disjoint, but seven held-out participants make its metrics unstable. All results are exploratory; no external validation or clinical readiness is claimed. Details are in [experiment results](docs/EXPERIMENTS.md).

## Repository structure

```text
src/
  image_features.py               preserved 32x32 pixel transformer
  iprognosis_data.py              typing event parsing and participant aggregation
  train_drawing_models.py         preserved drawing pixel baseline
  train_typing_models.py          i-PROGNOSIS participant-level experiment
tests/
  test_model_features.py
  test_iprognosis_data.py
docs/
  DATASETS.md
  DRAWING_DATA_QUALITY.md
  EXPERIMENTS.md
  ARCHITECTURE.md
  ML_WORKFLOW.md
  CODEBASE_WALKTHROUGH.md
  REPOSITORY_AUDIT.md
  LEARNING_GUIDE.md
models/                            fitted active models and preserved historical artifacts
results/                           metrics, assignments, CV tables, predictions and reports
data/                              local raw datasets; ignored by Git
notebooks/                         interactive entry points for the active experiments
requirements.txt                   pinned Python dependencies
```

The former drawing representation-comparison trainer and its tests were removed. Its saved tables and report remain under `results/drawing_representation_comparison/` as historical records, not active workflow outputs. Out-of-scope pen-trajectory and Tappy source trainers were also removed from active `src/` and tests; their old artifacts remain for reference.

## Dataset documentation

See [dataset inventory](docs/DATASETS.md) for source paths, labels, counts, event format, attribution and limitations, and the [drawing identity audit](docs/DRAWING_DATA_QUALITY.md) for image properties, duplicate checks and the unverified `V##` group tokens.

- **Drawings:** `data/Parkinson's drawings dataset/`. There are 204 PNG images, balanced between healthy and PD folder labels for each task. The local tree inspected contains no licence, citation, clinical metadata or participant crosswalk. Do not infer redistribution rights.
- **Typing:** `data/virtual keyboard data/iprognosis_typing_dev/`. The local ReadMe describes a custom keyboard on an LG Nexus 5X (Android 7.0), with up to 11 excerpts per participant. It documents CC BY 4.0 and the requested citation to Iakovakis et al. (2018), *Scientific Reports* 8, 7663, DOI [10.1038/s41598-018-25999-0](https://doi.org/10.1038/s41598-018-25999-0). Ten participants reportedly did not complete the initial shared excerpt; event files do not contain excerpt text to verify completeness.

Raw and participant-level data remain local and are not committed. The typing participant feature and prediction files under `results/` are derived participant-level research data; treat them with the same care as the source cohort.

## Environment setup

The verified environment is Windows, Python 3.14.6, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.4 and Pillow 12.3.0. Dependencies are pinned in `requirements.txt`; other environments have not been verified.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is disabled, invoke `.venv\Scripts\python.exe` directly.

## Tests and verification

Run the full test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
```

The test suite covers the active baseline image pipeline and typing event parsing, clinical label linkage, participant aggregation, participant splitting, and metrics. Run these commands after code changes to check the current suite:

## Running experiments

For interactive training and debugging, use the notebooks in `notebooks/` and follow [the step-by-step workflow](docs/ML_WORKFLOW.md). Each notebook inspects its data before training, then calls shared implementation in `src/`.

The active experiments can also be run from PowerShell with the Python entry points below. Use experiment-specific output paths for reruns so preserved artifacts remain untouched.

```powershell
# Preserved 32x32 drawing baseline
.\.venv\Scripts\python.exe -m src.train_drawing_models `
  --model-path models/finalization_baseline/drawing_image_model.joblib `
  --results-dir results/finalization_baseline/drawing `
  --seed 42 --image-size 32


# i-PROGNOSIS virtual touchscreen typing
.\.venv\Scripts\python.exe -m src.train_typing_models `
  --model-path models/finalization_baseline/typing_iprognosis_model.joblib `
  --results-dir results/finalization_baseline/typing `
  --seed 42
```


## Interpretation and limitations

- The direct 32×32 baseline distorts non-square wave scans. Its test set has only four unverified filename-derived candidate groups, so the estimate is unstable. Static images cannot reveal true speed, pauses or temporal smoothness.
- Typing splits on verified participant IDs, but the held-out test contains only seven people. Its ROC-AUC of 1.000 is not a stable estimate. The label is current PD/control group, not future conversion.
- Drawing provenance/licensing and participant identity remain unknown. Typing is a small single-study cohort and has possible confounding. Neither pipeline has external validation; neither supports clinical use or prospective disease-risk claims.

## Future work

Priorities are to obtain source-confirmed drawing participant identifiers and licensing/provenance; validate drawing features on independent participants; test whether drawing representations replicate in a larger cohort; and evaluate typing in a larger external participant cohort. Any future multimodal experiment needs a verified shared participant crosswalk and compatible outcomes. Prospective risk research additionally requires follow-up and a prespecified outcome horizon.

## References and acknowledgements

The i-PROGNOSIS local ReadMe requests citation of Iakovakis et al. (2018), “Touchscreen typing-pattern analysis for detecting fine motor skills decline in early-stage Parkinson’s disease,” *Scientific Reports* 8, 7663, DOI [10.1038/s41598-018-25999-0](https://doi.org/10.1038/s41598-018-25999-0). The local ReadMe documents CC BY 4.0. No drawing dataset citation or licence was found in the local files inspected; attribution and terms for that source remain unresolved.
