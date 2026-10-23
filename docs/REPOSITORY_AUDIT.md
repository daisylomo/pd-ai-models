# Repository consolidation audit

## Active code

- Drawing: `src/image_features.py`, `src/drawing_features.py`, `src/train_drawing_models.py`, and `src/train_drawing_comparisons.py`.
- Typing: `src/iprognosis_data.py` and `src/train_typing_models.py`.
- Tests: focused parsing/linkage/feature/split/metric tests under `tests/`.
- Documentation: dataset inventory, drawing identity audit, experiments, architecture, workflow, this audit, and learning guide.

The repository also contains exploratory notebooks, historical saved models/results, and local datasets. These do not enter the active training paths unless explicitly invoked. Out-of-scope Tappy and pen-trajectory trainer source was removed after checking imports; existing historical model/result artifacts and raw data were left in place.

## Evaluation findings

The drawing baseline correctly uses pipeline preprocessing and group-disjoint candidate partitions, but filename-derived `V##` identities are unverified. Its four-group test composition (one healthy and three Parkinson's groups) makes results unstable, and does not establish unseen-person performance. The original `training`/`testing` folders themselves share candidate tokens.

The comparison experiment uses deterministic per-image representations with fold-local scaling and model fitting. A shared fixed candidate-group partition allows representation comparisons on matching groups, but the groups are not known participants. The reserved test set is never used for selection. Its metrics must remain exploratory.

The typing loader aggregates to participants before partitioning and joins labels through clinical subject IDs. Grouped training CV keeps the seven held-out participants unseen and fits imputation/scaling within folds. The test has only seven participants, so even the reproduced high ROC-AUC is imprecise.

## Reproducibility

Seed 42 is recorded, as are partition/fold assignments, candidate CV tables, selected configurations and test predictions. Raw data are excluded from Git by `.gitignore`. The existing pinned `requirements.txt` is retained; no dependency was added for drawing features.

## Remaining limits

Drawing participant IDs, provenance, citation and licence remain unknown. The typing cohort is small and single-study; data completeness limitations are noted in its local source ReadMe. Neither modality has external validation, and neither evaluates prospective risk. See [dataset documentation](DATASETS.md), [drawing audit](DRAWING_DATA_QUALITY.md), and [experiment results](EXPERIMENTS.md).
