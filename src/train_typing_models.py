"""Participant-level baseline for i-PROGNOSIS smartphone typing data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.iprognosis_data import DEFAULT_IPROGNOSIS_ROOT, build_participant_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
TEST_FRACTION = 0.20


def make_participant_split(
    labels: np.ndarray,
    participant_ids: np.ndarray,
    *,
    seed: int = RANDOM_SEED,
    test_fraction: float = TEST_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified participant holdout and assert IDs do not overlap."""
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=labels
    )
    if set(participant_ids[train_indices]) & set(participant_ids[test_indices]):
        raise AssertionError("Participant overlap between training and test partitions")
    if len(np.unique(labels[train_indices])) != 2 or len(np.unique(labels[test_indices])) != 2:
        raise ValueError("Both classes must be represented in train and test partitions")
    return train_indices, test_indices


def _pipeline(classifier: Any, *, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _positive_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def evaluate(model: Pipeline, features: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(features).astype(int)
    scores = _positive_scores(model, features)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "n_participants": int(len(labels)),
        "class_support": {"control": int(np.sum(labels == 0)), "pd": int(np.sum(labels == 1))},
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_pd": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_pd": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_pd": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix_labels_control_pd": matrix.tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["control", "pd"],
            output_dict=True, zero_division=0,
        ),
    }


def run_experiment(
    iprognosis_root: Path = DEFAULT_IPROGNOSIS_ROOT,
    model_path: Path = PROJECT_ROOT / "models" / "typing_iprognosis_model.joblib",
    results_dir: Path = PROJECT_ROOT / "results" / "typing_iprognosis",
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    participants, excerpt_features, data_audit = build_participant_dataset(iprognosis_root)
    excluded = {"participant_id", "target", "excerpt_count", "total_valid_events"}
    feature_columns = [column for column in participants.columns if column not in excluded]
    X = participants[feature_columns]
    y = participants["target"].to_numpy(dtype=np.int64)
    participant_ids = participants["participant_id"].to_numpy(dtype=object)
    train_indices, test_indices = make_participant_split(y, participant_ids, seed=seed)
    X_train, y_train = X.iloc[train_indices], y[train_indices]
    X_test, y_test = X.iloc[test_indices], y[test_indices]
    ids_train, ids_test = participant_ids[train_indices], participant_ids[test_indices]

    minimum_class_count = int(pd.Series(y_train).value_counts().min())
    cv_folds = min(5, minimum_class_count)
    if cv_folds < 2:
        raise ValueError("Insufficient training participants per class for grouped CV")
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 1)
    cv_splits = list(cv.split(X_train, y_train, groups=ids_train))
    cv_assignments: dict[str, int] = {}
    cv_participant_ids: list[dict[str, list[str]]] = []
    for fold, (fold_train, fold_validation) in enumerate(cv_splits, start=1):
        fold_train_ids = set(ids_train[fold_train])
        fold_validation_ids = set(ids_train[fold_validation])
        if fold_train_ids & fold_validation_ids:
            raise AssertionError("Participant overlap inside cross-validation")
        for participant_id in fold_validation_ids:
            cv_assignments[str(participant_id)] = fold
        cv_participant_ids.append({
            "fold": fold,
            "train_participant_ids": sorted(str(x) for x in fold_train_ids),
            "validation_participant_ids": sorted(str(x) for x in fold_validation_ids),
        })

    candidates = {
        "logistic_regression": (
            _pipeline(
                LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed), scale=True
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            _pipeline(
                RandomForestClassifier(
                    n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1
                ),
                scale=False,
            ),
            {"classifier__max_features": ["sqrt"], "classifier__min_samples_leaf": [2, 4]},
        ),
    }
    searches: dict[str, GridSearchCV] = {}
    cv_rows: list[dict[str, Any]] = []
    for name, (estimator, parameter_grid) in candidates.items():
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring={"roc_auc": "roc_auc", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1": "f1"},
            refit="roc_auc",
            cv=cv_splits,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        searches[name] = search
        for index, params in enumerate(search.cv_results_["params"]):
            cv_rows.append({
                "model": name,
                "parameters": json.dumps(params, sort_keys=True),
                "mean_validation_roc_auc": float(search.cv_results_["mean_test_roc_auc"][index]),
                "std_validation_roc_auc": float(search.cv_results_["std_test_roc_auc"][index]),
                "mean_validation_accuracy": float(search.cv_results_["mean_test_accuracy"][index]),
                "mean_validation_balanced_accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][index]),
                "mean_validation_f1_pd": float(search.cv_results_["mean_test_f1"][index]),
                "selected_for_model": bool(index == search.best_index_),
            })

    candidate_best = {
        name: float(search.best_score_) for name, search in searches.items()
    }
    selected_name = max(candidate_best, key=candidate_best.get)
    selected_search = searches[selected_name]
    final_model = selected_search.best_estimator_
    test_metrics = evaluate(final_model, X_test, y_test)
    test_predictions = final_model.predict(X_test).astype(int)
    test_scores = _positive_scores(final_model, X_test)

    assignments = []
    for index, participant_id in enumerate(participant_ids):
        partition = "test" if index in set(test_indices) else "train"
        assignments.append({
            "participant_id": str(participant_id),
            "partition": partition,
            "cv_validation_fold": cv_assignments.get(str(participant_id)),
            "target": int(y[index]),
        })
    split_assignments = pd.DataFrame(assignments)
    test_output = pd.DataFrame({
        "participant_id": ids_test,
        "true_target": y_test,
        "predicted_target": test_predictions,
        "pd_score_uncalibrated": test_scores,
    }).sort_values("participant_id")

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    participants.to_csv(results_dir / "typing_participant_features.csv", index=False)
    excerpt_features.to_csv(results_dir / "typing_excerpt_feature_audit.csv", index=False)
    split_assignments.to_csv(results_dir / "typing_split_assignments.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(results_dir / "typing_cv_results.csv", index=False)
    test_output.to_csv(results_dir / "typing_test_predictions.csv", index=False)

    config = {
        "study_scope": "Retrospective classification of known PD versus control status from smartphone touchscreen typing; not prospective risk prediction",
        "seed": int(seed),
        "test_fraction": TEST_FRACTION,
        "split_method": "stratified participant-level holdout",
        "cv_method": "StratifiedGroupKFold on training participants only",
        "cv_folds": int(cv_folds),
        "selection_metric": "mean grouped-CV ROC-AUC; test set excluded from selection",
        "decision_threshold": 0.5,
        "positive_class": "PD",
        "feature_columns": feature_columns,
        "excluded_from_predictors": sorted(excluded),
        "candidate_grids": {
            "logistic_regression": {"C": [0.1, 1.0, 10.0]},
            "random_forest": {"n_estimators": [200], "max_features": ["sqrt"], "min_samples_leaf": [2, 4]},
        },
        "class_weighting": {"logistic_regression": "balanced", "random_forest": "balanced"},
        "data_audit": data_audit,
        "split": {
            "train_participant_ids": sorted(str(x) for x in ids_train),
            "test_participant_ids": sorted(str(x) for x in ids_test),
            "cv_folds": cv_participant_ids,
            "participant_overlap": 0,
        },
        "cross_validation_best_roc_auc_by_model": candidate_best,
        "selected_model": selected_name,
        "selected_hyperparameters": selected_search.best_params_,
        "test_metrics": test_metrics,
    }
    (results_dir / "typing_training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = {
        "study_scope": config["study_scope"],
        "dataset_audit": data_audit,
        "seed": int(seed),
        "split": config["split"],
        "cross_validation_best_roc_auc_by_model": candidate_best,
        "selected_model": selected_name,
        "selected_hyperparameters": selected_search.best_params_,
        "test_metrics": test_metrics,
        "interpretation": "Small single-study cohort; held-out metrics are unstable and do not establish external validity, clinical readiness, or prospective risk prediction.",
    }
    (results_dir / "typing_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({
        "pipeline": final_model,
        "feature_columns": feature_columns,
        "label_mapping": {"Control": 0, "PD": 1},
        "selected_model": selected_name,
        "selected_hyperparameters": selected_search.best_params_,
    }, model_path)

    report = [
        "# i-PROGNOSIS smartphone typing baseline",
        "",
        f"**Scope:** {config['study_scope']}.",
        "",
        f"- Cohort: {data_audit['participants']} participants ({data_audit['participants_by_label']['control']} controls, {data_audit['participants_by_label']['pd']} PD); {data_audit['excerpt_files']} excerpt files.",
        f"- Event audit: {data_audit['audit_totals'].get('valid_events', 0)} valid events; {data_audit['audit_totals'].get('malformed_lines', 0)} malformed lines; {data_audit['audit_totals'].get('timestamp_order_reversals', 0)} timestamp reversals; {data_audit['audit_totals'].get('out_of_range_pressure_events', 0)} pressure values outside documented [0,1].",
        f"- Split: {len(ids_train)} training and {len(ids_test)} held-out participants; model selection used {cv_folds}-fold grouped CV on training participants only.",
        f"- Selected model: {selected_name}, parameters `{json.dumps(selected_search.best_params_, sort_keys=True)}`.",
        f"- Test support: {test_metrics['class_support']}; accuracy {test_metrics['accuracy']:.3f}; balanced accuracy {test_metrics['balanced_accuracy']:.3f}; PD precision {test_metrics['precision_pd']:.3f}; PD recall {test_metrics['recall_pd']:.3f}; PD F1 {test_metrics['f1_pd']:.3f}; ROC-AUC {test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else 'undefined'}.",
        f"- Confusion matrix, rows=true and columns=predicted [control, PD]: `{test_metrics['confusion_matrix_labels_control_pd']}`.",
        "- Class-level metrics: " + json.dumps(test_metrics["classification_report"], sort_keys=True),
        "- Estimates are unstable due to the small single-study cohort. No external validation or clinical readiness is claimed.",
        "- Dataset license: CC BY 4.0. Cite Iakovakis et al. (2018), Scientific Reports 8, 7663, doi:10.1038/s41598-018-25999-0.",
        "",
    ]
    (results_dir / "typing_experiment_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "participants": data_audit["participants"],
        "participants_by_label": data_audit["participants_by_label"],
        "excerpt_files": data_audit["excerpt_files"],
        "audit_totals": data_audit["audit_totals"],
        "train_participants": len(ids_train),
        "test_participants": len(ids_test),
        "selected_model": selected_name,
        "selected_hyperparameters": selected_search.best_params_,
        "test_metrics": test_metrics,
        "results_dir": str(results_dir),
        "model_path": str(model_path),
    }, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iprognosis-root", type=Path, default=DEFAULT_IPROGNOSIS_ROOT)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "typing_iprognosis_model.joblib")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "typing_iprognosis")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.iprognosis_root, args.model_path, args.results_dir, seed=args.seed)


if __name__ == "__main__":
    main()
