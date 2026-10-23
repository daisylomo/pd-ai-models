# Parkinson’s Digital Motor Research Lab

This repository is a research and evaluation codebase for investigating digital motor characteristics associated with Parkinson’s disease (PD). Its experiments classify existing diagnostic-group labels. It is not a deployed application, diagnostic system, or prospective risk model.

## Research scope

The active work covers three tasks in two independent pipelines:

1. Spiral drawing image analysis.
2. Wave drawing image analysis.
3. Virtual touchscreen typing analysis.

Spiral and wave are evaluated separately and together by the drawing pipeline. Typing uses the i-PROGNOSIS touchscreen dataset in a separate participant-level pipeline. No verified cross-dataset participant links support multimodal fusion.

## Current experiment status

| Task | Dataset and representation | Selected model | Verified result | Main limitation |
|---|---|---|---|---|
| Preserved drawing pixel baseline | 204 spiral/wave PNGs; direct 32×32 grayscale pixels | Logistic regression | 48-image test: accuracy .396, balanced accuracy .428, PD precision .323, recall .556, F1 .408, ROC-AUC .494; confusion `[[9,21],[8,10]]` | Four filename-derived candidate groups in test; identity is unverified |
| Spiral drawing | Same drawing dataset; foreground/shape/radial/gradient descriptors | RBF SVM | Grouped CV ROC-AUC .876±.076; 24-image candidate-group test ROC-AUC .733, balanced accuracy .644; confusion `[[11,4],[4,5]]` | Four unverified candidate groups; exploratory estimate |
| Wave drawing | Same drawing dataset; foreground/shape/radial/gradient descriptors | Random forest | Grouped CV ROC-AUC .716±.071; 24-image candidate-group test ROC-AUC .444, balanced accuracy .422; confusion `[[11,4],[8,1]]` | Weak held-out result; four unverified groups |
| Combined spiral and wave | Same dataset; descriptors | RBF SVM | Grouped CV ROC-AUC .750±.063; 48-image test ROC-AUC .554, balanced accuracy .467; confusion `[[23,7],[15,3]]` | Numerically above the pixel baseline on this split, but based on four unverified groups and not reliable evidence of improvement |
| Virtual touchscreen typing | i-PROGNOSIS; participant medians of timing, pause and pressure features | Random forest | 33 participants (15 controls, 18 PD), 338 excerpts; seven-person test accuracy .857, balanced accuracy .875, PD precision/recall/F1 1.000/.750/.857, ROC-AUC 1.000; confusion `[[3,0],[1,3]]` | Very small single-study test set; retrospective labels |

The preserved drawing pixel baseline and the new comparison use the same outer grouped test assignment (48 images, one healthy and three PD candidate groups). Spiral and wave rows contain 24 images each with those same four candidate groups. `V##` filename tokens have not been verified as participants, so these scores do not establish generalization to unseen people. The typing result is participant-disjoint, but seven held-out participants make its metrics unstable. All results are exploratory; no external validation or clinical readiness is claimed. Details and complete CV tables are in [experiment results](docs/EXPERIMENTS.md).

## Repository structure

```text
src/
  drawing_features.py             deterministic drawing representations
  image_features.py               preserved 32x32 pixel transformer
  iprognosis_data.py              typing event parsing and participant aggregation
  train_drawing_models.py         preserved drawing pixel baseline
  train_drawing_comparisons.py    spiral, wave and combined comparison
  train_typing_models.py          i-PROGNOSIS participant-level experiment
tests/
  test_drawing_features.py
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
notebooks/                         exploratory analysis, not imported by trainers
requirements.txt                   pinned Python dependencies
```

Out-of-scope pen-trajectory and Tappy source trainers were removed from active `src/` and tests. Their existing model/result artifacts, raw datasets and exploratory notebook history remain preserved; they are not active experiments.

## Dataset documentation

See [dataset inventory](docs/DATASETS.md) for source paths, labels, counts, event format, attribution and limitations, and the [drawing identity audit](docs/DRAWING_DATA_QUALITY.md) for image properties, duplicate checks and the unverified `V##` group tokens.

- **Drawings:** `data/Parkinson's Datasets/Parkinson's drawings dataset/drawings/`. There are 204 PNG images, balanced between healthy and PD folder labels for each task. The local tree inspected contains no licence, citation, clinical metadata or participant crosswalk. Do not infer redistribution rights.
- **Typing:** `data/Parkinson's Datasets/virtual keyboard data/iprognosis_typing_dev/`. The local ReadMe describes a custom keyboard on an LG Nexus 5X (Android 7.0), with up to 11 excerpts per participant. It documents CC BY 4.0 and the requested citation to Iakovakis et al. (2018), *Scientific Reports* 8, 7663, DOI [10.1038/s41598-018-25999-0](https://doi.org/10.1038/s41598-018-25999-0). Ten participants reportedly did not complete the initial shared excerpt; event files do not contain excerpt text to verify completeness.

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

The verified suite has 14 passing tests covering image representations, group isolation, event parsing, clinical label linkage, participant aggregation, participant splitting and metrics. Both commands completed successfully in the current environment.

## Running experiments

Use experiment-specific output paths for reruns so the preserved baseline artifacts remain untouched.

```powershell
# Preserved 32x32 drawing baseline
.\.venv\Scripts\python.exe -m src.train_drawing_models `
  --model-path models/finalization_baseline/drawing_image_model.joblib `
  --results-dir results/finalization_baseline/drawing `
  --seed 42 --image-size 32

# Spiral, wave and combined representation comparison
.\.venv\Scripts\python.exe -m src.train_drawing_comparisons --seed 42

# i-PROGNOSIS virtual touchscreen typing
.\.venv\Scripts\python.exe -m src.train_typing_models `
  --model-path models/finalization_baseline/typing_iprognosis_model.joblib `
  --results-dir results/finalization_baseline/typing `
  --seed 42
```

The drawing comparison writes selected model bundles under `models/drawing_representation_comparison/` and CV scores, grouped split/fold assignments, held-out predictions, metrics and a report under `results/drawing_representation_comparison/`. The typing rerun writes its fitted pipeline under `models/finalization_baseline/typing_iprognosis_model.joblib` and participant/excerpt features, split assignments, CV results, test predictions, configuration, metrics and report under `results/finalization_baseline/typing/`. Drawing baseline defaults point to canonical output paths; use the isolated paths above when preserving those artifacts matters.

## Interpretation and limitations

- The drawing comparison selects among multiple candidates on grouped CV, so winning CV estimates are selection-optimistic. Its held-out unit is an unverified filename-derived group, with only four groups in the overall test set. Images within a group are correlated.
- The direct 32×32 baseline distorts non-square wave scans. The comparison adds aspect-preserving pixels and static foreground/shape/radial/gradient descriptors. Static images cannot reveal true speed, pauses or temporal smoothness. On the combined test set the descriptor model is numerically higher than the pixel baseline, but four candidate groups cannot establish a repeatable improvement.
- Typing splits on verified participant IDs, but the held-out test contains only seven people. Its ROC-AUC of 1.000 is not a stable estimate. The label is current PD/control group, not future conversion.
- Drawing provenance/licensing and participant identity remain unknown. Typing is a small single-study cohort and has possible confounding. Neither pipeline has external validation; neither supports clinical use or prospective disease-risk claims.

## Future work

Priorities are to obtain source-confirmed drawing participant identifiers and licensing/provenance; validate drawing features on independent participants; test whether drawing representations replicate in a larger cohort; and evaluate typing in a larger external participant cohort. Any future multimodal experiment needs a verified shared participant crosswalk and compatible outcomes. Prospective risk research additionally requires follow-up and a prespecified outcome horizon.

## References and acknowledgements

The i-PROGNOSIS local ReadMe requests citation of Iakovakis et al. (2018), “Touchscreen typing-pattern analysis for detecting fine motor skills decline in early-stage Parkinson’s disease,” *Scientific Reports* 8, 7663, DOI [10.1038/s41598-018-25999-0](https://doi.org/10.1038/s41598-018-25999-0). The local ReadMe documents CC BY 4.0. No drawing dataset citation or licence was found in the local files inspected; attribution and terms for that source remain unresolved.
