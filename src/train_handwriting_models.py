"""Participant-grouped case-control models for labeled pen-trajectory recordings.

Run from any working directory with:
    python src/train_handwriting_models.py

This dataset contains X/Y/Z, pressure, grip-angle, timestamp, and task-code
samples. The participant key is provisionally inferred from C_#### and the
repeated P_#### filename prefix; verify that convention in the source metadata.
The resulting experiment is a separate cohort from the spiral/wave image model.
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
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "Parkinson's Datasets"
SENSOR_ROOTS = [
    DATA_ROOT / "Parkinson's UCI dataset" / "hw_dataset",
    DATA_ROOT / "Park's UCI kaggle dataset" / "hw_dataset",
]
LABEL_TO_TARGET = {"control": 0, "parkinson": 1}
SENSOR_COLUMNS = ["X", "Y", "Z", "Pressure", "GripAngle", "Timestamp", "Test ID"]
RANDOM_SEED = 42
NUMERIC_FEATURES = [
    "n_points",
    "path_length",
    "net_displacement",
    "path_efficiency",
    "x_range",
    "y_range",
    "aspect_ratio",
    "step_mean",
    "step_std",
    "step_q90",
    "step_cv",
    "turn_abs_mean",
    "turn_abs_std",
    "turn_abs_q90",
    "sharp_turn_fraction",
    "pressure_mean",
    "pressure_std",
    "pressure_min",
    "pressure_max",
    "pressure_nonzero_fraction",
    "grip_angle_mean",
    "grip_angle_std",
    "z_mean",
    "z_std",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def participant_group(path: Path, label: str) -> str:
    if label == "control":
        match = re.match(r"^(C_\d+)", path.stem, flags=re.IGNORECASE)
        token = match.group(1).upper() if match else None
    else:
        match = re.match(r"^P_(\d{4})\d*", path.stem, flags=re.IGNORECASE)
        token = f"P_{match.group(1)}" if match else None
    if token is None:
        raise ValueError(f"Cannot derive a participant-group candidate from filename: {path.name}")
    return f"{label}:{token}"


def summarize_task(points: pd.DataFrame) -> dict[str, float]:
    points = points.dropna(subset=["X", "Y"]).reset_index(drop=True)
    x = points["X"].to_numpy(dtype=float)
    y = points["Y"].to_numpy(dtype=float)
    if len(points) < 2:
        raise ValueError("A task recording must contain at least two valid X/Y points")

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)
    headings = np.arctan2(dy, dx)
    turns = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    turn_magnitudes = np.abs(turns)
    path_length = float(step.sum())
    net_displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    step_mean = float(step.mean())
    pressure = points["Pressure"].to_numpy(dtype=float)
    grip_angle = points["GripAngle"].to_numpy(dtype=float)
    z_values = points["Z"].to_numpy(dtype=float)

    return {
        "n_points": float(len(points)),
        "path_length": path_length,
        "net_displacement": net_displacement,
        "path_efficiency": net_displacement / path_length if path_length > 0 else 0.0,
        "x_range": float(np.ptp(x)),
        "y_range": float(np.ptp(y)),
        "aspect_ratio": float(np.ptp(x) / max(np.ptp(y), 1e-9)),
        "step_mean": step_mean,
        "step_std": float(step.std()),
        "step_q90": float(np.quantile(step, 0.9)),
        "step_cv": float(step.std() / step_mean) if step_mean > 0 else 0.0,
        "turn_abs_mean": float(turn_magnitudes.mean()) if len(turn_magnitudes) else 0.0,
        "turn_abs_std": float(turn_magnitudes.std()) if len(turn_magnitudes) else 0.0,
        "turn_abs_q90": float(np.quantile(turn_magnitudes, 0.9)) if len(turn_magnitudes) else 0.0,
        "sharp_turn_fraction": float((turn_magnitudes > np.pi / 2).mean()) if len(turn_magnitudes) else 0.0,
        "pressure_mean": float(np.nanmean(pressure)),
        "pressure_std": float(np.nanstd(pressure)),
        "pressure_min": float(np.nanmin(pressure)),
        "pressure_max": float(np.nanmax(pressure)),
        "pressure_nonzero_fraction": float((pressure > 0).mean()),
        "grip_angle_mean": float(np.nanmean(grip_angle)),
        "grip_angle_std": float(np.nanstd(grip_angle)),
        "z_mean": float(np.nanmean(z_values)),
        "z_std": float(np.nanstd(z_values)),
    }


def build_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    unique_recordings: dict[str, dict[str, Any]] = {}
    file_count = 0
    for root in SENSOR_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.txt")):
            if path.name.lower() == "readme.txt" or path.name.startswith("._"):
                continue
            path_parts = {part.lower() for part in path.parts}
            label = next((name for name in LABEL_TO_TARGET if name in path_parts), None)
            if label is None:
                continue
            file_count += 1
            content_hash = sha256_file(path)
            if content_hash in unique_recordings:
                if unique_recordings[content_hash]["label"] != label:
                    raise ValueError(f"Identical recordings have conflicting labels: {path}")
                continue

            rows = []
            malformed_lines = 0
            with path.open("r", encoding="utf-8-sig", errors="replace") as source:
                for line in source:
                    if not line.strip():
                        continue
                    values = [value.strip() for value in line.strip().split(";")]
                    if len(values) != len(SENSOR_COLUMNS):
                        malformed_lines += 1
                        continue
                    rows.append(values)
            if malformed_lines:
                raise ValueError(f"{malformed_lines} malformed nonblank rows in {path}")
            if not rows:
                raise ValueError(f"No sensor rows found in {path}")

            points = pd.DataFrame(rows, columns=SENSOR_COLUMNS)
            for column in SENSOR_COLUMNS:
                points[column] = pd.to_numeric(points[column], errors="coerce")
            points = points.dropna(subset=["X", "Y", "Test ID"])
            for task_id, task_points in points.groupby("Test ID", sort=True):
                unique_recordings[f"{content_hash}:{task_id}"] = {
                    "label": label,
                    "target": LABEL_TO_TARGET[label],
                    "group": participant_group(path, label),
                    "task": str(int(task_id)),
                    "features": summarize_task(task_points),
                    "content_hash": content_hash,
                }

    records = list(unique_recordings.values())
    if not records:
        raise FileNotFoundError("No labeled, parseable sensor recordings found in the configured roots")
    dataset = pd.DataFrame(
        [
            {
                "label": record["label"],
                "target": record["target"],
                "group": record["group"],
                "task": record["task"],
                **record["features"],
            }
            for record in records
        ]
    )
    if set(dataset["label"]) != set(LABEL_TO_TARGET):
        raise ValueError("Both control and Parkinson's recordings are required")
    audit = {
        "source_files_scanned": file_count,
        "unique_recording_contents": len({record["content_hash"] for record in records}),
        "recording_task_samples": len(dataset),
        "candidate_groups_by_label": dataset.drop_duplicates(["group", "label"]).groupby("label")["group"].nunique().to_dict(),
        "malformed_nonblank_lines": 0,
        "missing_feature_values": int(dataset[NUMERIC_FEATURES].isna().sum().sum()),
    }
    return dataset, audit


def has_both_classes(y: np.ndarray) -> bool:
    return set(np.unique(y)) == {0, 1}


def choose_group_fold(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    features = np.zeros((len(y), 1), dtype=np.uint8)
    for train_indices, holdout_indices in splitter.split(features, y, groups):
        if has_both_classes(y[train_indices]) and has_both_classes(y[holdout_indices]):
            return train_indices, holdout_indices
    raise ValueError(f"Could not create a {n_splits}-fold participant split with both classes")


def make_splits(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = pd.DataFrame({"target": y, "group": groups}).drop_duplicates()
    minimum_groups = int(unique_groups.groupby("target")["group"].nunique().min())
    if minimum_groups < 5:
        raise ValueError("At least five independent participant groups per class are required")
    train_validation, test = choose_group_fold(y, groups, 5, seed)
    train_relative, validation_relative = choose_group_fold(
        y[train_validation], groups[train_validation], 4, seed + 1
    )
    train = train_validation[train_relative]
    validation = train_validation[validation_relative]
    partitions = {"train": train, "validation": validation, "test": test}
    for name, indices in partitions.items():
        if not has_both_classes(y[indices]):
            raise ValueError(f"{name} partition is missing a class")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[partitions[first]]) & set(groups[partitions[second]]):
            raise AssertionError(f"Group leakage between {first} and {second}")
    return train, validation, test


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("task", OneHotEncoder(handle_unknown="ignore"), ["task"]),
        ]
    )


def decision_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def evaluate(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(X)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "n_recording_task_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, decision_scores(model, X))),
        "sensitivity": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else 0.0,
        "precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_control_parkinson": matrix.tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["control", "parkinson"],
            output_dict=True,
            zero_division=0,
        ),
    }


def run_experiment(model_path: Path, results_dir: Path, seed: int = RANDOM_SEED) -> dict[str, Any]:
    dataset, audit = build_dataset()
    feature_columns = [*NUMERIC_FEATURES, "task"]
    X = dataset[feature_columns]
    y = dataset["target"].to_numpy(dtype=np.int64)
    groups = dataset["group"].to_numpy(dtype=object)
    train_indices, validation_indices, test_indices = make_splits(y, groups, seed)
    X_train, y_train, groups_train = X.iloc[train_indices], y[train_indices], groups[train_indices]
    X_validation, y_validation = X.iloc[validation_indices], y[validation_indices]
    X_test, y_test = X.iloc[test_indices], y[test_indices]

    cv_group_counts = pd.DataFrame({"target": y_train, "group": groups_train}).drop_duplicates().groupby("target")["group"].nunique()
    cv_folds = min(3, int(cv_group_counts.min()))
    if cv_folds < 2:
        raise ValueError("Training split has too few participant groups for cross-validation")
    cv = list(
        StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 2).split(
            X_train, y_train, groups_train
        )
    )
    for fold_train, fold_validation in cv:
        if set(groups_train[fold_train]) & set(groups_train[fold_validation]):
            raise AssertionError("Participant-group leakage inside cross-validation")

    baseline = Pipeline([("features", make_preprocessor()), ("classifier", DummyClassifier(strategy="prior"))])
    baseline_scores = cross_validate(
        baseline,
        X_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    baseline_summary = {
        "cv_train_roc_auc_mean": float(baseline_scores["train_roc_auc"].mean()),
        "cv_roc_auc_mean": float(baseline_scores["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(baseline_scores["test_roc_auc"].std(ddof=0)),
        "cv_roc_auc_gap": float(baseline_scores["train_roc_auc"].mean() - baseline_scores["test_roc_auc"].mean()),
        "cv_train_balanced_accuracy_mean": float(baseline_scores["train_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_mean": float(baseline_scores["test_balanced_accuracy"].mean()),
    }
    baseline_validation = clone(baseline).fit(X_train, y_train)
    baseline_summary["validation_metrics"] = evaluate(baseline_validation, X_validation, y_validation)

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
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=1),
            {"classifier__max_features": ["sqrt", 0.75], "classifier__min_samples_leaf": [2, 4]},
        ),
    }

    searches = {}
    rows = []
    for name, (estimator, parameter_grid) in candidate_specs.items():
        pipeline = Pipeline([("features", make_preprocessor()), ("classifier", estimator)])
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
        searches[name] = search
        validation_metrics = evaluate(search.best_estimator_, X_validation, y_validation)
        best = int(search.best_index_)
        rows.append(
            {
                "model": name,
                "cv_train_roc_auc_mean": float(search.cv_results_["mean_train_roc_auc"][best]),
                "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][best]),
                "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][best]),
                "cv_roc_auc_gap": float(
                    search.cv_results_["mean_train_roc_auc"][best]
                    - search.cv_results_["mean_test_roc_auc"][best]
                ),
                "cv_train_balanced_accuracy_mean": float(
                    search.cv_results_["mean_train_balanced_accuracy"][best]
                ),
                "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][best]),
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_sensitivity": validation_metrics["sensitivity"],
                "validation_specificity": validation_metrics["specificity"],
                "validation_precision": validation_metrics["precision"],
                "validation_f1": validation_metrics["f1"],
                "validation_false_positives": int(validation_metrics["confusion_matrix_labels_control_parkinson"][0][1]),
                "validation_false_negatives": int(validation_metrics["confusion_matrix_labels_control_parkinson"][1][0]),
                "validation_confusion_matrix": json.dumps(
                    validation_metrics["confusion_matrix_labels_control_parkinson"]
                ),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    candidates = pd.DataFrame(rows).sort_values(
        ["validation_roc_auc", "validation_balanced_accuracy", "cv_roc_auc_mean"], ascending=False
    )
    selected_name = str(candidates.iloc[0]["model"])
    selected = searches[selected_name]
    fit_indices = np.concatenate([train_indices, validation_indices])
    final_model = Pipeline(selected.best_estimator_.steps)
    final_model.fit(X.iloc[fit_indices], y[fit_indices])
    test_metrics = evaluate(final_model, X_test, y_test)
    final_baseline = clone(baseline).fit(X.iloc[fit_indices], y[fit_indices])
    baseline_test_metrics = evaluate(final_baseline, X_test, y_test)

    split_summary = {}
    for name, indices in (("train", train_indices), ("validation", validation_indices), ("test", test_indices)):
        class_groups = pd.DataFrame({"target": y[indices], "group": groups[indices]}).drop_duplicates()
        split_summary[name] = {
            "recording_task_samples": int(len(indices)),
            "participant_groups": int(pd.Series(groups[indices]).nunique()),
            "control_samples": int(np.sum(y[indices] == 0)),
            "parkinson_samples": int(np.sum(y[indices] == 1)),
            "control_groups": int(class_groups.loc[class_groups["target"] == 0, "group"].nunique()),
            "parkinson_groups": int(class_groups.loc[class_groups["target"] == 1, "group"].nunique()),
        }

    summary = {
        "study_scope": "Exploratory control-versus-known-Parkinson's pen-trajectory classification; not prospective early-risk prediction",
        "seed": int(seed),
        "label_mapping": LABEL_TO_TARGET,
        "group_key": "class-scoped C_#### / P_#### filename convention (provisional; verify in source metadata)",
        "target_features": NUMERIC_FEATURES,
        "task_feature": "Test ID one-hot encoded; 0 static spiral, 1 dynamic spiral, 2 circular motion per readme",
        "excluded_features": ["filename", "source path", "group ID", "raw timestamp", "release name"],
        "dataset_audit": audit,
        "split_summary": split_summary,
        "grouped_cv_folds": int(cv_folds),
        "majority_baseline_cv": baseline_summary,
        "majority_baseline_test_metrics": baseline_test_metrics,
        "candidate_results": rows,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "model_selection": "validation ROC-AUC, with validation balanced accuracy as a tie-breaker; test metrics are not used to select the model",
        "motion_fusion": "not performed: this pen-trajectory cohort has no verified participant mapping to the image, C3D, or UPDRS cohorts",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_model, "label_mapping": LABEL_TO_TARGET, "group_key_note": summary["group_key"]}, model_path)
    candidates.to_csv(results_dir / "handwriting_candidate_results.csv", index=False)
    with (results_dir / "handwriting_training_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print("Dataset audit:")
    print(json.dumps(audit, indent=2))
    print("Grouped split summary:")
    print(pd.DataFrame(split_summary).T.to_string())
    print(f"\nMajority baseline CV ROC-AUC: {baseline_summary['cv_roc_auc_mean']:.3f} +/- {baseline_summary['cv_roc_auc_std']:.3f}")
    print("Candidate model CV and validation results:")
    print(candidates.to_string(index=False))
    print(f"\nSelected on validation ROC-AUC: {selected_name}")
    print("Untouched grouped test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved pipeline: {model_path}")
    print(f"Saved metrics: {results_dir / 'handwriting_training_metrics.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "handwriting_trajectory_model.joblib")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    run_experiment(arguments.model_path, arguments.results_dir, arguments.seed)


if __name__ == "__main__":
    main()
