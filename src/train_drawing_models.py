"""Leakage-aware case-control pipeline for spiral and wave drawing images.

Run from the repository root with:
    python -m src.train_drawing_models

The filename-derived V## token is treated as a provisional participant key and
namespaced by class because healthy and Parkinson's IDs may be numbered
independently. Confirm that convention against the source documentation before
using the scores as evidence of generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.image_features import GrayscaleVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "Parkinson's drawings dataset"
)
LABEL_TO_TARGET = {"healthy": 0, "parkinson": 1}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RANDOM_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(image_root: Path) -> pd.DataFrame:
    """Read labels/tasks from directories and grouping keys from filenames."""
    records = []
    for task in ("spiral", "wave"):
        for split in ("training", "testing"):
            for label in LABEL_TO_TARGET:
                directory = image_root / task / split / label
                if not directory.is_dir():
                    continue
                for path in sorted(directory.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    match = re.match(r"^(V\d+)", path.stem, flags=re.IGNORECASE)
                    if match is None:
                        raise ValueError(f"Cannot derive a conservative group token from filename: {path.name}")
                    records.append(
                        {
                            "path": str(path.resolve()),
                            "task": task,
                            "source_split": split,
                            "label": label,
                            "target": LABEL_TO_TARGET[label],
                            "group": f"{label}:{match.group(1).upper()}",
                            "sha256": sha256_file(path),
                        }
                    )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise FileNotFoundError(f"No supported labeled spiral/wave images found under {image_root}")

    conflicting_hashes = manifest.groupby("sha256")["target"].nunique().gt(1)
    if conflicting_hashes.any():
        raise ValueError("Identical image bytes have conflicting class labels; resolve this before training")
    duplicate_hashes = manifest[manifest.duplicated("sha256", keep=False)]
    if not duplicate_hashes.empty:
        groups_per_image = duplicate_hashes.groupby("sha256")["group"].nunique()
        if groups_per_image.gt(1).any():
            raise ValueError("An identical image maps to multiple participant groups; resolve this before splitting")
        manifest = manifest.drop_duplicates("sha256", keep="first").reset_index(drop=True)

    observed_labels = set(manifest["label"])
    if observed_labels != set(LABEL_TO_TARGET):
        raise ValueError(f"Expected both labels {set(LABEL_TO_TARGET)}, found {observed_labels}")
    return manifest


def _has_both_classes(y: np.ndarray) -> bool:
    return set(np.unique(y)) == {0, 1}


def _choose_group_fold(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first deterministic stratified fold with both classes."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    features = np.zeros((len(y), 1), dtype=np.uint8)
    for train_indices, holdout_indices in splitter.split(features, y, groups):
        if _has_both_classes(y[train_indices]) and _has_both_classes(y[holdout_indices]):
            return train_indices, holdout_indices
    raise ValueError(f"Could not form a {n_splits}-fold split with both classes in each partition")


def make_grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create approximately 60/20/20 train/validation/test participant splits."""
    groups_per_class = pd.DataFrame({"target": y, "group": groups}).drop_duplicates()
    minimum_groups = int(groups_per_class.groupby("target")["group"].nunique().min())
    if minimum_groups < 5:
        raise ValueError("At least five independent participant groups per class are required for this split")

    train_validation, test = _choose_group_fold(y, groups, n_splits=5, seed=seed)
    inner_train, validation_relative = _choose_group_fold(
        y[train_validation], groups[train_validation], n_splits=4, seed=seed + 1
    )
    train = train_validation[inner_train]
    validation = train_validation[validation_relative]

    partitions = {"train": train, "validation": validation, "test": test}
    for name, indices in partitions.items():
        if not _has_both_classes(y[indices]):
            raise ValueError(f"The {name} partition does not contain both classes")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(groups[partitions[first]]) & set(groups[partitions[second]])
        if overlap:
            raise AssertionError(f"Participant-group leakage between {first} and {second}")
    return train, validation, test


def make_preprocessor(image_size: int) -> ColumnTransformer:
    image_pipeline = Pipeline(
        [
            ("vectorize", GrayscaleVectorizer(size=image_size)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("pixels", image_pipeline, ["path"]),
            ("task", OneHotEncoder(handle_unknown="ignore"), ["task"]),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def decision_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def evaluate(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(X)
    scores = decision_scores(model, X)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else 0.0
    return {
        "n_images": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "sensitivity": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_healthy_parkinson": matrix.tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["healthy", "parkinson"],
            output_dict=True,
            zero_division=0,
        ),
    }


def run_experiment(
    image_root: Path,
    model_path: Path,
    results_dir: Path,
    seed: int = RANDOM_SEED,
    image_size: int = 32,
) -> dict[str, Any]:
    manifest = build_manifest(image_root)
    X = manifest[["path", "task"]]
    y = manifest["target"].to_numpy(dtype=np.int64)
    groups = manifest["group"].to_numpy(dtype=object)
    train_indices, validation_indices, test_indices = make_grouped_splits(y, groups, seed)

    X_train, y_train, groups_train = X.iloc[train_indices], y[train_indices], groups[train_indices]
    X_validation, y_validation = X.iloc[validation_indices], y[validation_indices]
    X_test, y_test = X.iloc[test_indices], y[test_indices]

    cv_group_counts = pd.DataFrame({"target": y_train, "group": groups_train}).drop_duplicates().groupby("target")["group"].nunique()
    cv_folds = min(3, int(cv_group_counts.min()))
    if cv_folds < 2:
        raise ValueError("Training partition does not contain enough groups per class for grouped CV")
    cv = list(
        StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 2).split(
            X_train, y_train, groups_train
        )
    )
    for train_fold, validation_fold in cv:
        if set(groups_train[train_fold]) & set(groups_train[validation_fold]):
            raise AssertionError("Participant-group leakage inside cross-validation")

    baseline_pipeline = Pipeline(
        [("features", make_preprocessor(image_size)), ("classifier", DummyClassifier(strategy="prior"))]
    )
    baseline_cv = cross_validate(
        baseline_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    baseline = {
        "cv_train_roc_auc_mean": float(baseline_cv["train_roc_auc"].mean()),
        "cv_roc_auc_mean": float(baseline_cv["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(baseline_cv["test_roc_auc"].std(ddof=0)),
        "cv_roc_auc_gap": float(baseline_cv["train_roc_auc"].mean() - baseline_cv["test_roc_auc"].mean()),
        "cv_train_balanced_accuracy_mean": float(baseline_cv["train_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_mean": float(baseline_cv["test_balanced_accuracy"].mean()),
    }
    baseline_validation = clone(baseline_pipeline).fit(X_train, y_train)
    baseline["validation_metrics"] = evaluate(baseline_validation, X_validation, y_validation)

    candidate_specs = {
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "rbf_svm": (
            SVC(kernel="rbf", class_weight="balanced", random_state=seed),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__gamma": ["scale"]},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
            {
                "classifier__max_features": ["sqrt", 0.25],
                "classifier__min_samples_leaf": [2, 4],
            },
        ),
    }

    fitted_searches = {}
    candidate_results = []
    for name, (estimator, parameter_grid) in candidate_specs.items():
        pipeline = Pipeline(
            [("features", make_preprocessor(image_size)), ("classifier", estimator)]
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"},
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted_searches[name] = search
        validation_metrics = evaluate(search.best_estimator_, X_validation, y_validation)
        best_index = int(search.best_index_)
        candidate_results.append(
            {
                "model": name,
                "cv_train_roc_auc_mean": float(search.cv_results_["mean_train_roc_auc"][best_index]),
                "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][best_index]),
                "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][best_index]),
                "cv_roc_auc_gap": float(
                    search.cv_results_["mean_train_roc_auc"][best_index]
                    - search.cv_results_["mean_test_roc_auc"][best_index]
                ),
                "cv_train_balanced_accuracy_mean": float(
                    search.cv_results_["mean_train_balanced_accuracy"][best_index]
                ),
                "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][best_index]),
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_sensitivity": validation_metrics["sensitivity"],
                "validation_specificity": validation_metrics["specificity"],
                "validation_precision": validation_metrics["precision"],
                "validation_f1": validation_metrics["f1"],
                "validation_false_positives": int(validation_metrics["confusion_matrix_labels_healthy_parkinson"][0][1]),
                "validation_false_negatives": int(validation_metrics["confusion_matrix_labels_healthy_parkinson"][1][0]),
                "validation_confusion_matrix": json.dumps(
                    validation_metrics["confusion_matrix_labels_healthy_parkinson"]
                ),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    candidate_table = pd.DataFrame(candidate_results).sort_values(
        ["validation_roc_auc", "validation_balanced_accuracy", "cv_roc_auc_mean"],
        ascending=False,
    )
    selected_name = str(candidate_table.iloc[0]["model"])
    selected_search = fitted_searches[selected_name]

    fit_indices = np.concatenate([train_indices, validation_indices])
    final_model = clone(selected_search.best_estimator_)
    final_model.fit(X.iloc[fit_indices], y[fit_indices])
    test_metrics = evaluate(final_model, X_test, y_test)
    final_baseline = clone(baseline_pipeline).fit(X.iloc[fit_indices], y[fit_indices])
    baseline_test_metrics = evaluate(final_baseline, X_test, y_test)

    split_summary = {}
    for name, indices in (("train", train_indices), ("validation", validation_indices), ("test", test_indices)):
        class_groups = pd.DataFrame({"target": y[indices], "group": groups[indices]}).drop_duplicates()
        split_summary[name] = {
            "images": int(len(indices)),
            "participant_groups": int(pd.Series(groups[indices]).nunique()),
            "healthy_images": int(np.sum(y[indices] == 0)),
            "parkinson_images": int(np.sum(y[indices] == 1)),
            "healthy_groups": int(class_groups.loc[class_groups["target"] == 0, "group"].nunique()),
            "parkinson_groups": int(class_groups.loc[class_groups["target"] == 1, "group"].nunique()),
        }

    summary = {
        "study_scope": "Exploratory healthy-versus-known-Parkinson's image classification; not prospective early-risk prediction",
        "seed": int(seed),
        "image_size": int(image_size),
        "label_mapping": LABEL_TO_TARGET,
        "group_key": "class-scoped filename prefix V## (provisional participant token; verify with source documentation)",
        "input_root": str(image_root.resolve()),
        "n_images_after_exact_deduplication": int(len(manifest)),
        "n_unique_candidate_groups": int(pd.Series(groups).nunique()),
        "split_summary": split_summary,
        "grouped_cv_folds": int(cv_folds),
        "majority_baseline_cv": baseline,
        "majority_baseline_test_metrics": baseline_test_metrics,
        "candidate_results": candidate_results,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "model_selection": "validation ROC-AUC, with validation balanced accuracy as a tie-breaker; test metrics are not used to select the model",
        "motion_fusion": "not performed: no verified participant crosswalk or labeled healthy-control motion cohort is available",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_model,
            "label_mapping": LABEL_TO_TARGET,
            "group_key_note": summary["group_key"],
            "image_size": image_size,
        },
        model_path,
    )
    candidate_table.to_csv(results_dir / "drawing_candidate_results.csv", index=False)
    with (results_dir / "drawing_training_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print("Grouped split summary:")
    print(pd.DataFrame(split_summary).T.to_string())
    print(f"\nMajority baseline CV ROC-AUC: {baseline['cv_roc_auc_mean']:.3f} +/- {baseline['cv_roc_auc_std']:.3f}")
    print("Candidate model CV and validation results:")
    print(candidate_table.to_string(index=False))
    print(f"\nSelected on validation ROC-AUC: {selected_name}")
    print("Untouched grouped test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved pipeline: {model_path}")
    print(f"Saved metrics: {results_dir / 'drawing_training_metrics.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "drawing_image_model.joblib")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--image-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run_experiment(
        image_root=arguments.image_root,
        model_path=arguments.model_path,
        results_dir=arguments.results_dir,
        seed=arguments.seed,
        image_size=arguments.image_size,
    )


if __name__ == "__main__":
    main()
