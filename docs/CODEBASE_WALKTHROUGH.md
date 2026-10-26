# Active code walkthrough

## Data flow

`src/train_drawing_models.py` creates a labeled image manifest, removes exact duplicate content, generates grouped train/validation/test partitions, performs grouped-CV model searches, selects using validation ROC-AUC, and writes the original pixel-baseline outputs.

`src/iprognosis_data.py` reads the documented touchscreen event lines, audits malformed or anomalous events, links S## files to clinical `Subject ID` and `Group`, and aggregates excerpt features into one participant row. `src/train_typing_models.py` performs a participant holdout, training-only grouped CV and pipeline selection; it writes split identities, participant features, test predictions, metrics and a serialized pipeline.

## Tests and data

- `tests/test_model_features.py`: preserved 32×32 image vectorizer and drawing split test.
- `tests/test_iprognosis_data.py`: event parsing, feature extraction, clinical linkage, aggregation and participant split.
- `data/`: local raw files ignored by Git. Exact active paths and data caveats are in [DATASETS](DATASETS.md).
- `models/` and `results/`: preserved canonical and historical artifacts plus the isolated finalization reruns.

The interactive entry points are `notebooks/drawing_pixel_baseline.ipynb` and `notebooks/typing_iprognosis.ipynb`. Each notebook previews its inputs, exposes run paths and parameters, and calls the reusable implementations in `src/`. The former drawing comparison trainer and notebook have been removed; its saved results are archived. Historical pen/Tappy trainers were removed from `src/`; their old result/model artifacts are retained for reference.

## Reproducibility

Commands and pinned environment setup are documented in the root README. Run from repository root. Fixed seed 42, group/participant assignments, feature configuration, CV scores and test predictions are saved alongside each experiment. Only load trusted joblib artifacts.
