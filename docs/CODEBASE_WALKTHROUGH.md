# Codebase walkthrough

## Structure

- `src/`: shared image transformer, Tappy loader, three trainers.
- `tests/test_tappy_data.py`: three Tappy data tests.
- `data/`: local raw files, ignored by Git.
- `models/`: serialized estimators; `results/`: metrics, tables, plots and reports.
- `notebooks/parkinsons_eda.ipynb`: exploratory analysis; `first_experiment.ipynb`: package/environment check.
- `requirements.txt`: pinned dependencies; `AGENTS.md`: research rules; `PROJECT_STATUS.md`: current state.

## Execution order

Running `python -m src.train_tappy_models` imports dependencies and local loader functions, then the module guard calls `main()`. `main()` parses arguments and calls `run_experiment()`. That function loads/aggregates participants, splits IDs, searches pipelines in training CV, selects a model, evaluates test participants, bootstraps intervals, saves artifacts and prints JSON/table summaries. For inference, build feature rows, load the bundle through `predict_participant_features()`, check expected columns, then predict class and score.

## Tappy loader: `src/tappy_data.py`

- `load_profiles(root)` reads user profiles from ZIP into a key-indexed mapping.
- `_valid_date()` and `_valid_time()` validate date and clock strings.
- `_session_statistics(values, prefix)` calculates numeric distribution summaries.
- `_session_features(events)` summarizes timing, hand and direction for one monthly file.
- `_combine_sessions(user_key, target, bucket)` median-aggregates monthly records into a participant record.
- `_profile_audit(users, profiles)` counts labels and profile-field completeness.
- `build_participant_dataset(root, min_events)` returns labeled participant rows plus quality audit.
- `build_unlabeled_participant_features(root)` prepares participant features for inference. Known profiles may carry a `target` column; drop it before prediction.

## Tappy trainer: `src/train_tappy_models.py`

- `_pipeline(classifier, scale)` chains median imputation, missingness indicators, optional standard scaling and classifier. sklearn refits each step inside CV folds.
- `_score_values()` returns decision scores where supported, otherwise positive-class probabilities; neither guarantees calibration.
- `_evaluate()` calculates metrics and fixed-label confusion matrix.
- `_bootstrap_intervals()` resamples test participants separately within each class.
- `_feature_definitions()` creates a definition/unit table; `_plot_results()` writes figures; `_write_report()` renders Markdown.
- `run_experiment()` orchestrates all training, selection, test evaluation and writing. `predict_participant_features()` performs saved-bundle inference.
- `parse_args()` and `main()` form the CLI; the `__main__` guard prevents training on ordinary imports.

## Drawing code

`src/image_features.py` defines `GrayscaleVectorizer`. `fit()` returns itself (the sklearn transformer convention); `transform()` EXIF-corrects images, converts to grayscale, resizes to 32x32 with Pillow, normalizes pixels to [0,1], and flattens them.

In `src/train_drawing_models.py`, `sha256_file()` fingerprints contents; `build_manifest()` gathers paths/labels/groups; `_has_both_classes()` and `_choose_group_fold()` support grouped folds; `make_grouped_splits()` creates partitions; `make_preprocessor()` builds image/task processing; `decision_scores()` and `evaluate()` score a model. `run_experiment()` searches candidates, selects on validation ROC-AUC (balanced accuracy tie-break), refits train plus validation, evaluates test and writes artifacts. `parse_args()`/`main()` expose options.

## Pen code

In `src/train_handwriting_models.py`, `sha256_file()` identifies duplicate contents; `participant_group()` parses provisional filename group tokens; `summarize_task()` calculates trace and sensor summaries; `build_dataset()` parses/deduplicates files and returns feature rows plus audit. `has_both_classes()`, `choose_group_fold()` and `make_splits()` create group-aware partitions. `make_preprocessor()` configures numeric/categorical transforms. `decision_scores()` and `evaluate()` calculate metrics. `run_experiment()` compares candidates, selects, tests and saves outputs; `parse_args()`/`main()` provide CLI.

## Python and sklearn syntax

`Path` provides platform-aware paths. Type hints such as `dict[str, Any]` document expected values without runtime enforcement. `Pipeline` chains transforms and estimator so preprocessing is trained with each fold. `GridSearchCV` evaluates a settings grid across folds and refits the best configuration. `train_test_split(..., stratify=y)` preserves approximate class proportions. `joblib.dump/load` serializes/restores objects (only load trusted files). The `if __name__ == "__main__"` condition runs CLI only when invoked as a module/script. `**` unpacks a mapping into named arguments.

See [ML workflow](ML_WORKFLOW.md), [experiments](EXPERIMENTS.md), and [learning guide](LEARNING_GUIDE.md) for model intuition, metric definitions, exact results and caveats.
