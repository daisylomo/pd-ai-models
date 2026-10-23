# Active code walkthrough

## Data flow

`src/train_drawing_models.py` creates a labeled image manifest, removes exact duplicate content, generates grouped train/validation/test partitions, performs grouped-CV model searches, selects using validation ROC-AUC, and writes the original pixel-baseline outputs.

`src/drawing_features.py` implements deterministic aspect-preserving grayscale pixels and static image descriptors. `src/train_drawing_comparisons.py` evaluates pixels, descriptors and their combination for spiral, wave and combined-task subsets. It reserves a fixed candidate-group test partition and keeps standardization/model fitting within grouped CV.

`src/iprognosis_data.py` reads the documented touchscreen event lines, audits malformed or anomalous events, links S## files to clinical `Subject ID` and `Group`, and aggregates excerpt features into one participant row. `src/train_typing_models.py` performs a participant holdout, training-only grouped CV and pipeline selection; it writes split identities, participant features, test predictions, metrics and a serialized pipeline.

## Tests and data

- `tests/test_drawing_features.py`: new drawing representations, deterministic output, grouped CV isolation and metric schema.
- `tests/test_model_features.py`: preserved 32×32 image vectorizer and drawing split test.
- `tests/test_iprognosis_data.py`: event parsing, feature extraction, clinical linkage, aggregation and participant split.
- `data/`: local raw files ignored by Git. Exact active paths and data caveats are in [DATASETS](DATASETS.md).
- `models/` and `results/`: preserved canonical and historical artifacts plus the isolated finalization reruns.

The notebook `notebooks/parkinsons_eda.ipynb` is exploratory and may contain older analyses outside the active scope; active trainers do not import it. Historical pen/Tappy trainers were removed from `src/`; their old result/model artifacts are retained for reference.

## Reproducibility

Commands and pinned environment setup are documented in the root README. Run from repository root. Fixed seed 42, group/participant assignments, feature configuration, CV scores and test predictions are saved alongside each experiment. Only load trusted joblib artifacts.
