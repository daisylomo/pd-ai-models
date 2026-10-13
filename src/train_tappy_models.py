"""Reproducible participant-level Tappy Keystroke model comparison.

Run from the repository root with ``python -m src.train_tappy_models``. Model
selection uses cross-validation within training participants; the fixed held-out
participant test set is evaluated only after the candidate has been selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.tappy_data import DEFAULT_TAPPY_ROOT, DIRECTION_LEVELS, TIMING_COLUMNS, build_participant_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
TEST_FRACTION = 0.25
BOOTSTRAP_REPLICATES = 2000


def _pipeline(classifier: Any, *, scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _score_values(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _evaluate(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    predicted = model.predict(X)
    scores = _score_values(model, X)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "n_participants": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc_average_precision": float(average_precision_score(y, scores)),
        "sensitivity": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "confusion_matrix_control_parkinson": matrix.tolist(),
    }


def _bootstrap_intervals(
    y: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Stratified participant bootstrap; preserves test class counts per draw."""
    rng = np.random.default_rng(seed)
    by_class = {label: np.flatnonzero(y == label) for label in (0, 1)}
    values: dict[str, list[float]] = {
        "roc_auc": [], "pr_auc_average_precision": [], "balanced_accuracy": [],
        "sensitivity": [], "specificity": [], "precision": [], "f1": [],
    }
    for _ in range(replicates):
        sample = np.concatenate([
            rng.choice(by_class[label], size=len(by_class[label]), replace=True) for label in (0, 1)
        ])
        sample_y, sample_pred, sample_scores = y[sample], predictions[sample], scores[sample]
        matrix = confusion_matrix(sample_y, sample_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        values["roc_auc"].append(float(roc_auc_score(sample_y, sample_scores)))
        values["pr_auc_average_precision"].append(float(average_precision_score(sample_y, sample_scores)))
        values["balanced_accuracy"].append(float(balanced_accuracy_score(sample_y, sample_pred)))
        values["sensitivity"].append(float(tp / (tp + fn)) if tp + fn else 0.0)
        values["specificity"].append(float(tn / (tn + fp)) if tn + fp else 0.0)
        values["precision"].append(float(precision_score(sample_y, sample_pred, zero_division=0)))
        values["f1"].append(float(f1_score(sample_y, sample_pred, zero_division=0)))
    return {
        name: {
            "lower_95_percentile": float(np.quantile(samples, 0.025)),
            "upper_97_5_percentile": float(np.quantile(samples, 0.975)),
            "method": "stratified participant bootstrap percentile interval",
            "replicates": replicates,
        }
        for name, samples in values.items()
    }


def _feature_definitions(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for name in feature_columns:
        unit = "dimensionless fraction" if "fraction" in name or name.endswith("_cv_session_median") else "milliseconds"
        if name.endswith("_session_median"):
            source = name.removesuffix("_session_median")
            unit = "dimensionless fraction" if "fraction" in source or source.endswith("_cv") else "milliseconds"
            if source.endswith("_mean"):
                definition = "Median across participant-month files of the within-file mean"
            elif source.endswith("_std"):
                definition = "Median across participant-month files of the within-file population standard deviation"
            elif source.endswith("_median"):
                definition = "Median across participant-month files of the within-file median"
            elif source.endswith("_q90"):
                definition = "Median across participant-month files of the within-file 90th percentile"
            elif source.endswith("_iqr"):
                definition = "Median across participant-month files of within-file interquartile range"
            elif source.endswith("_cv"):
                definition = "Median across participant-month files of within-file standard deviation divided by mean"
            elif source.endswith("_fraction"):
                definition = "Median across participant-month files of the fraction of valid events in this category"
            else:
                definition = "Median of the corresponding within-file feature"
            if source.startswith("hold_time"):
                unit = "milliseconds" if not source.endswith("_cv") else "dimensionless"
            elif source.startswith(("latency_time", "flight_time")):
                unit = "milliseconds" if not source.endswith("_cv") else "dimensionless"
        else:
            definition = "Median across participant-month files of the within-file category fraction"
        rows.append({"feature": name, "definition": definition, "unit": unit})
    return pd.DataFrame(rows)


def _plot_results(
    output_dir: Path,
    candidate_table: pd.DataFrame,
    y_test: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    test_metrics: dict[str, Any],
    baseline_scores: np.ndarray,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, values, color in (("Selected model", scores, "#b34b4b"), ("Majority baseline", baseline_scores, "#687e91")):
        fpr, tpr, _ = roc_curve(y_test, values)
        precision, recall, _ = precision_recall_curve(y_test, values)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, values):.3f})", color=color)
        axes[1].plot(recall, precision, label=name, color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC curve")
    prevalence = float(np.mean(y_test))
    axes[1].axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence baseline={prevalence:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall curve", ylim=(0, 1.02))
    for axis in axes:
        axis.legend(loc="best", fontsize=8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    paths["test_roc_pr_plot"] = str(output_dir / "tappy_test_roc_pr.png")
    fig.savefig(paths["test_roc_pr_plot"], dpi=160)
    plt.close(fig)

    matrix = np.asarray(test_metrics["confusion_matrix_control_parkinson"])
    fig, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Control", "Parkinson's"],
             yticklabels=["Control", "Parkinson's"], xlabel="Predicted label", ylabel="True label",
             title="Held-out confusion matrix")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=13)
    fig.tight_layout()
    paths["confusion_matrix_plot"] = str(output_dir / "tappy_confusion_matrix.png")
    fig.savefig(paths["confusion_matrix_plot"], dpi=160)
    plt.close(fig)

    ordered = candidate_table.sort_values("cv_roc_auc_mean")
    positions = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(positions - 0.18, ordered["cv_train_roc_auc_mean"], height=0.35, label="Training folds", color="#a9bbc9")
    axis.barh(positions + 0.18, ordered["cv_roc_auc_mean"], height=0.35, label="Validation folds", color="#397da5")
    axis.axvline(0.5, linestyle="--", color="gray", linewidth=1)
    axis.set(yticks=positions, yticklabels=ordered["model"], xlabel="ROC-AUC", title="Training vs validation CV performance", xlim=(0, 1.05))
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    paths["cv_performance_plot"] = str(output_dir / "tappy_cv_performance.png")
    fig.savefig(paths["cv_performance_plot"], dpi=160)
    plt.close(fig)
    return paths


def _write_report(path: Path, summary: dict[str, Any], candidates: pd.DataFrame) -> None:
    test = summary["test_metrics"]
    baseline = summary["majority_baseline_test_metrics"]
    ci = summary["test_confidence_intervals"]
    split = summary["split"]
    lines = [
        "# Tappy Keystroke participant-level experiment",
        "",
        f"**Scope:** {summary['study_scope']}",
        "",
        "## Cohort and data quality",
        "",
        f"- Participants: {summary['dataset_audit']['participants']} ({summary['dataset_audit']['participants_by_label']['control']} controls; {summary['dataset_audit']['participants_by_label']['parkinson']} self-reported Parkinson's).",
        f"- Monthly recordings: {summary['dataset_audit']['event_files']} files; {summary['dataset_audit']['sessions_by_label']}",
        f"- Valid matched events used: {summary['dataset_audit']['matched_event_rows']:,}.",
        f"- Excluded duplicate rows: {summary['dataset_audit']['duplicate_rows_excluded']:,}; UserKey conflicts: {summary['dataset_audit']['filename_userkey_mismatch_rows']:,}; invalid timing rows: {summary['dataset_audit']['invalid_timing_rows']:,}.",
        f"- Timestamp audit: invalid dates {summary['dataset_audit']['invalid_date_rows']:,}; invalid times {summary['dataset_audit']['invalid_timestamp_rows']:,}; filename-month mismatches {summary['dataset_audit']['date_month_filename_mismatches']:,}; ordering reversals {summary['dataset_audit']['timestamp_order_reversals']:,}.",
        "- Unit of analysis: participant. Each monthly recording is summarized first; a participant feature is the median across their monthly recordings. No participant appears in more than one split.",
        "- Usage counts and profile demographics/medication fields are audit-only and excluded from model predictors.",
        "",
        "## Split and model selection",
        "",
        f"- Split: seed {summary['seed']}, {split['train_participants']} training participants and {split['test_participants']} held-out participants.",
        f"- Training labels: {split['train_control']} controls and {split['train_parkinson']} Parkinson's; test labels: {split['test_control']} controls and {split['test_parkinson']} Parkinson's.",
        f"- Model selected by training-only {summary['cross_validation']['folds']}-fold CV ROC-AUC: **{summary['selected_model']}**, {summary['selected_hyperparameters']}.",
        "- Test data were not used for feature or hyperparameter selection.",
        "",
        "| Candidate | Train-fold ROC-AUC | Validation-fold ROC-AUC (SD) | Validation-fold PR-AUC | Balanced accuracy | Parameters |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in candidates.sort_values("cv_roc_auc_mean", ascending=False).to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['cv_train_roc_auc_mean']:.3f} | {row['cv_roc_auc_mean']:.3f} ({row['cv_roc_auc_std']:.3f}) | {row['cv_average_precision_mean']:.3f} | {row['cv_balanced_accuracy_mean']:.3f} | `{row['best_params']}` |"
        )
    lines += [
        "",
        "## Held-out performance",
        "",
        f"- ROC-AUC: {test['roc_auc']:.3f} (95% bootstrap interval {ci['roc_auc']['lower_95_percentile']:.3f} to {ci['roc_auc']['upper_97_5_percentile']:.3f}).",
        f"- PR-AUC (average precision): {test['pr_auc_average_precision']:.3f} (test positive prevalence {split['test_parkinson'] / split['test_participants']:.3f}; compare to this prevalence baseline).",
        f"- Balanced accuracy {test['balanced_accuracy']:.3f}; sensitivity {test['sensitivity']:.3f}; specificity {test['specificity']:.3f}; precision {test['precision']:.3f}; F1 {test['f1']:.3f}.",
        f"- Confusion matrix, rows=true [control, Parkinson's], columns=predicted [control, Parkinson's]: `{test['confusion_matrix_control_parkinson']}`.",
        f"- All-positive majority baseline: ROC-AUC {baseline['roc_auc']:.3f}; average precision {baseline['pr_auc_average_precision']:.3f}; balanced accuracy {baseline['balanced_accuracy']:.3f}; specificity {baseline['specificity']:.3f}.",
        f"- Stratified bootstrap intervals: balanced accuracy {ci['balanced_accuracy']['lower_95_percentile']:.3f} to {ci['balanced_accuracy']['upper_97_5_percentile']:.3f}; sensitivity {ci['sensitivity']['lower_95_percentile']:.3f} to {ci['sensitivity']['upper_97_5_percentile']:.3f}; specificity {ci['specificity']['lower_95_percentile']:.3f} to {ci['specificity']['upper_97_5_percentile']:.3f} (2,000 participant resamples).",
        "- ROC-AUC assesses ranking; PR-AUC emphasizes positive-class precision/recall and is sensitive to prevalence. Balanced accuracy gives both classes equal weight. Sensitivity/specificity are included as case/control rates, not clinical performance claims.",
        "",
        "## Interpretation and limitations",
        "",
        "Labels are self-reported, typing is uncontrolled, and the cohort is modest and imbalanced. Profile variables such as age, gender, disease severity/duration, tremor and medication can confound associations; they are excluded from model input. Usage volume can also reflect health, computer use, or compliance and is excluded as a predictor. This experiment does not establish diagnosis or prospective risk detection.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python -m src.train_tappy_models",
        "python -m unittest discover -s tests -v",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    tappy_root: Path = DEFAULT_TAPPY_ROOT,
    model_path: Path = PROJECT_ROOT / "models" / "tappy_keystroke_model.joblib",
    results_dir: Path = PROJECT_ROOT / "results",
    *,
    min_events: int = 0,
    seed: int = RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    dataset, audit = build_participant_dataset(tappy_root, min_events=min_events)
    metadata_columns = {"user_key", "target", "event_count", "session_count"}
    feature_columns = [column for column in dataset.columns if column not in metadata_columns]
    X = dataset[feature_columns]
    y = dataset["target"].to_numpy(dtype=np.int64)
    indices = np.arange(len(dataset))
    train_ids, test_ids = train_test_split(indices, test_size=TEST_FRACTION, random_state=seed, stratify=y)
    X_train, y_train = X.iloc[train_ids], y[train_ids]
    X_test, y_test = X.iloc[test_ids], y[test_ids]
    train_user_keys = set(dataset.iloc[train_ids]["user_key"])
    test_user_keys = set(dataset.iloc[test_ids]["user_key"])
    if train_user_keys & test_user_keys:
        raise AssertionError("Participant overlap between training and test partitions")

    cv_folds = min(5, int(pd.Series(y_train).value_counts().min()))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 1)
    candidates = {
        "logistic_regression": (
            _pipeline(LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "rbf_svm": (
            _pipeline(SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__gamma": ["scale"]},
        ),
        "random_forest": (
            _pipeline(
                RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=1),
                scale=False,
            ),
            {"classifier__n_estimators": [200], "classifier__max_features": ["sqrt", 0.5],
             "classifier__min_samples_leaf": [2, 5]},
        ),
        "gradient_boosting": (
            _pipeline(GradientBoostingClassifier(random_state=seed), scale=False),
            {"classifier__n_estimators": [100, 250], "classifier__learning_rate": [0.03, 0.1],
             "classifier__max_depth": [1, 2], "classifier__min_samples_leaf": [3, 8]},
        ),
    }
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision", "balanced_accuracy": "balanced_accuracy"}
    searches: dict[str, GridSearchCV] = {}
    rows = []
    for name, (estimator, parameters) in candidates.items():
        search = GridSearchCV(
            estimator, parameters, scoring=scoring, refit="roc_auc", cv=cv, n_jobs=1,
            return_train_score=True, error_score="raise",
        )
        search.fit(X_train, y_train)
        searches[name] = search
        best = search.best_index_
        rows.append({
            "model": name,
            "cv_train_roc_auc_mean": float(search.cv_results_["mean_train_roc_auc"][best]),
            "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][best]),
            "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][best]),
            "cv_train_average_precision_mean": float(search.cv_results_["mean_train_average_precision"][best]),
            "cv_average_precision_mean": float(search.cv_results_["mean_test_average_precision"][best]),
            "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][best]),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        })

    candidate_table = pd.DataFrame(rows).sort_values(
        ["cv_roc_auc_mean", "cv_balanced_accuracy_mean"], ascending=False
    ).reset_index(drop=True)
    selected_name = str(candidate_table.iloc[0]["model"])
    selected_search = searches[selected_name]
    final_model = selected_search.best_estimator_  # refit on all training participants by GridSearchCV
    test_metrics = _evaluate(final_model, X_test, y_test)
    baseline = _pipeline(DummyClassifier(strategy="prior"), scale=False).fit(X_train, y_train)
    baseline_metrics = _evaluate(baseline, X_test, y_test)
    test_predictions = final_model.predict(X_test).astype(int)
    test_scores = _score_values(final_model, X_test)
    confidence_intervals = _bootstrap_intervals(
        y_test, test_predictions, test_scores, seed=seed + 100, replicates=bootstrap_replicates
    )

    split = {
        "method": "stratified participant-level holdout; all monthly recordings summarized before split",
        "train_participants": int(len(train_ids)),
        "test_participants": int(len(test_ids)),
        "train_control": int((y_train == 0).sum()),
        "train_parkinson": int((y_train == 1).sum()),
        "test_control": int((y_test == 0).sum()),
        "test_parkinson": int((y_test == 1).sum()),
        "participant_overlap": 0,
    }
    config = {
        "seed": int(seed), "test_fraction": TEST_FRACTION, "cv": "StratifiedKFold",
        "cv_folds": int(cv_folds), "selection_metric": "mean training-fold ROC-AUC",
        "test_used_for_selection": False, "minimum_events": int(min_events),
        "bootstrap_replicates": int(bootstrap_replicates),
        "class_weighting": {
            "logistic_regression": "balanced", "rbf_svm": "balanced",
            "random_forest": "balanced_subsample", "gradient_boosting": "none",
        },
        "candidate_grids": {
            name: {key: list(value) for key, value in params.items()}
            for name, (_, params) in candidates.items()
        },
    }
    summary = {
        "study_scope": "Exploratory participant-level classification of self-reported Parkinsons status from naturalistic typing; not prospective risk prediction or clinical diagnosis",
        "seed": int(seed),
        "label_mapping": {"control": 0, "self_reported_parkinson": 1},
        "feature_columns": feature_columns,
        "excluded_predictors": ["user_key", "target", "event_count", "session_count", "Date", "Timestamp",
                                "BirthYear", "Gender", "Tremors", "DiagnosisYear", "Sided", "UPDRS",
                                "Impact", "Levadopa", "DA", "MAOB", "Other"],
        "unit_of_analysis": "participant; participant-month summaries are median-aggregated within participant",
        "dataset_audit": audit,
        "split": split,
        "cross_validation": {"method": "StratifiedKFold within training participants", "folds": int(cv_folds)},
        "training_config": config,
        "candidate_results": rows,
        "selected_model": selected_name,
        "selected_hyperparameters": selected_search.best_params_,
        "selection_rule": "Highest mean training-fold ROC-AUC; test set is not used for selection",
        "majority_baseline_test_metrics": baseline_metrics,
        "test_metrics": test_metrics,
        "test_confidence_intervals": confidence_intervals,
        "test_positive_prevalence": float(y_test.mean()),
        "limitations": [
            "Labels are self-reported profile fields, not clinician-adjudicated diagnoses.",
            "Typing is naturalistic and uncontrolled; age, medication, tremor, disease severity, computer use and device differences may confound associations.",
            "Usage counts are excluded as predictors but reflect differing exposure/compliance; participant feature summaries are medians over monthly recordings.",
            "The held-out sample is small and from one cohort; no external validation or prospective follow-up is included.",
            "This dataset records general typing, not a standardized finger-tapping assessment.",
            "Bootstrap intervals quantify test-participant resampling uncertainty only and do not account for cohort/site shift or model-selection uncertainty.",
        ],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_model, "feature_columns": feature_columns, "training_config": config,
                 "selected_model": selected_name}, model_path)
    candidate_table.to_csv(results_dir / "tappy_candidate_results.csv", index=False)
    (results_dir / "tappy_training_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "tappy_training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _feature_definitions(feature_columns).to_csv(results_dir / "tappy_feature_definitions.csv", index=False)
    test_output = pd.DataFrame({
        "user_key": dataset.iloc[test_ids]["user_key"].to_numpy(),
        "true_label": y_test,
        "predicted_label": test_predictions,
        "decision_score": test_scores,
    })
    test_output["error"] = test_output["true_label"] != test_output["predicted_label"]
    test_output.to_csv(results_dir / "tappy_test_predictions.csv", index=False)
    plot_paths = _plot_results(results_dir, candidate_table, y_test, test_scores, test_predictions,
                               test_metrics, _score_values(baseline, X_test))
    summary["artifacts"] = {
        "model": str(model_path),
        "candidate_results": str(results_dir / "tappy_candidate_results.csv"),
        "training_config": str(results_dir / "tappy_training_config.json"),
        "feature_definitions": str(results_dir / "tappy_feature_definitions.csv"),
        "test_predictions": str(results_dir / "tappy_test_predictions.csv"),
        **plot_paths,
    }
    # Rewrite metrics after artifact paths have been recorded, then create report.
    (results_dir / "tappy_training_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(results_dir / "tappy_experiment_report.md", summary, candidate_table)

    print("Dataset audit:")
    print(json.dumps(audit, indent=2))
    print("Training CV candidate results:")
    print(candidate_table.to_string(index=False))
    print(f"Selected model: {selected_name} ({selected_search.best_params_})")
    print("Held-out test metrics:")
    print(json.dumps({"selected_model": test_metrics, "majority_baseline": baseline_metrics,
                      "confidence_intervals": confidence_intervals}, indent=2))
    print(f"Experiment report: {results_dir / 'tappy_experiment_report.md'}")
    return summary


def predict_participant_features(model_path: Path, features: pd.DataFrame) -> pd.DataFrame:
    """Return class and ranking score; the score is not a calibrated probability."""
    bundle = joblib.load(model_path)
    expected = bundle["feature_columns"]
    missing = sorted(set(expected) - set(features.columns))
    if missing:
        raise ValueError(f"Missing required Tappy feature columns: {missing}")
    X = features.loc[:, expected]
    model = bundle["pipeline"]
    result = pd.DataFrame({
        "prediction": model.predict(X).astype(int),
        "decision_score": _score_values(model, X),
    })
    if "user_key" in features:
        result.insert(0, "user_key", features["user_key"].to_numpy())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tappy-root", type=Path, default=DEFAULT_TAPPY_ROOT)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "tappy_keystroke_model.joblib")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--min-events", type=int, default=0, help="Optional participant event-count threshold")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.tappy_root, args.model_path, args.results_dir,
                   min_events=args.min_events, seed=args.seed,
                   bootstrap_replicates=args.bootstrap_replicates)


if __name__ == "__main__":
    main()
