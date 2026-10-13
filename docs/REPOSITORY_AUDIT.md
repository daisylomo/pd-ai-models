# Repository audit and proposed structure

## Current repository state

The current checkout has one commit (`145ceae initial models`) and a clean Git worktree. Raw data are ignored by `.gitignore`; source, documentation, experiment JSON/CSV/PNG outputs, and four joblib model bundles are tracked. The tracked artifacts support result inspection, but binaries from joblib should only be loaded when trusted. The status text that previously claimed there were no commits and all files were untracked is stale.

There are three runnable trainers: `src/train_drawing_models.py`, `src/train_handwriting_models.py`, and `src/train_tappy_models.py`. Dataset loading and feature code live in `src/tappy_data.py` and `src/image_features.py`; pen and image loading/features are currently in their respective trainer modules. Before this audit pass, automated tests covered only Tappy data. This pass adds synthetic image, pen-feature, and group-split tests; the suite now has eight tests. Both notebooks are exploratory/support notebooks; the EDA covers inventory, handwriting sensor streams, drawing images, gait/body-motion exports and prior model results. No notebook is imported by training scripts. `first_experiment.ipynb` is a small environment check.

## Data flow and experiment integrity

- Drawing: labeled folder paths -> hash deduplication -> provisional class-scoped filename groups -> grouped train/validation/test and grouped CV -> grayscale pixels -> three classifiers -> validation selection -> refit on train plus validation -> held-out image metrics.
- Pen: two `hw_dataset` trees -> parse and hash deduplication -> task features -> provisional filename groups -> grouped partitions/CV -> three classifiers -> validation selection -> train+validation refit -> held-out task-sample metrics.
- Tappy: profile ZIP plus expanded monthly event logs -> matching/validation/audit -> timing and category summaries per month -> equal-month median per participant -> stratified participant holdout -> train-only five-fold CV and pipeline selection -> one held-out participant evaluation, baseline and bootstrap interval -> model and reports.

Tappy's participant split and in-fold preprocessing avoid the main repeated-measurement and preprocessing leakage paths. Drawing/pen use group-disjoint partitions, but the group mapping is not source-verified. Tappy CV performance is also used to select among several model families/settings, so those CV estimates are selection-optimistic; the held-out score remains the relevant final estimate but is imprecise. The pen test result uses only one PD candidate group. No cross-dataset performance comparison uses matched splits, and the scores should not be compared as if they were one benchmark.

## Code quality and reproducibility findings

- All three module `--help` commands import successfully; the eight-test suite passes and `compileall` completes.
- Hashing and metric calculation patterns are duplicated in drawing and pen. Their data schemas, group split logic and metric key formats differ. Extracting a general framework now would risk hiding important dataset distinctions; reuse only a small helper after tests demonstrate identical semantics.
- Experiment settings are constants and CLI arguments, not separate config files. There is no package metadata (`pyproject.toml`), CI setup, or environment lock beyond the pinned `requirements.txt`. The lock includes notebook dependencies and is validated only in the observed Windows/Python 3.14.6 environment.
- Tappy outputs are relatively complete; image and pen outputs are less uniform (no equivalent per-sample test prediction CSV or plots). This is a useful consistency improvement, but should be added as a small reporting layer without changing model selection.
- The EDA notebook repeats some inventory/hash logic, as expected for exploration. Keep it unless repeated calculations become a source of conflicting results.
- There is no data augmentation implementation. No augmentation should be added without a modality-specific research rationale and train-only fit boundary.

## Proposed target structure

Keep the existing source paths and CLI entry points to protect working commands. Add organization only where it improves traceability:

```text
.
├── configs/                         # optional versioned experiment settings, if extracted
├── data/                            # local raw datasets; ignored by Git
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CODEBASE_WALKTHROUGH.md
│   ├── DATASETS.md
│   ├── EXPERIMENTS.md
│   ├── LEARNING_GUIDE.md
│   ├── ML_WORKFLOW.md
│   └── REPOSITORY_AUDIT.md
├── models/                          # fitted artifacts
├── notebooks/                       # exploratory analysis
├── results/                         # run-specific outputs
├── src/
│   ├── image_features.py
│   ├── tappy_data.py
│   ├── train_drawing_models.py
│   ├── train_handwriting_models.py
│   └── train_tappy_models.py
├── tests/                            # Tappy, image, pen-feature and split tests
├── PROJECT_STATUS.md
├── README.md
└── requirements.txt
```

This is intentionally close to the working tree. A future `src/common/` should be added only for a proven reusable responsibility, such as output metadata or a metric function with matching label and threshold semantics. Do not move all modules into a generic framework or force shared preprocessing. A common CLI/report envelope can record run ID, dataset, target, sample unit, split policy, seed, selected estimator, artifact paths and metrics while leaving metric computation and evaluation protocol dataset-specific.

## Staged implementation recommendation

1. Preserve current code and document dataset provenance, schema and unused raw data (done in `DATASETS.md`).
2. Verify filename-derived groups against source metadata before treating image/pen estimates as participant-generalization results.
3. Add synthetic-fixture tests for image vectorization, pen feature construction and group non-overlap (completed in this pass).
4. Reproduce each existing primary experiment to isolated output directories; compare summaries and artifact schemas without overwriting saved runs (completed; selected configurations and evaluation summaries matched exactly).
5. Standardize result metadata and test-prediction outputs, keeping existing metric definitions and selection protocols unchanged.
6. Only then consider extracting shared evaluation/output helpers, with no result changes beyond formatting.
7. For the early-risk goal, first locate or collect prospective data with a verified baseline/follow-up design. Do not relabel current case/control classifiers as risk models.
