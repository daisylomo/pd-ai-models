"""Controlled exploratory comparison of drawing image representations.

Run from the repository root with ``python -m src.train_drawing_comparisons``.
Filename-derived groups are provisional tokens, so this run cannot establish
generalization to verified unseen participants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.drawing_features import DrawingImageVectorizer
from src.image_features import GrayscaleVectorizer
from src.train_drawing_models import (
    DEFAULT_IMAGE_ROOT,
    LABEL_TO_TARGET,
    RANDOM_SEED,
    build_manifest,
    make_grouped_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("pixels32_baseline", "aspect_pixels64", "handcrafted", "combined64")
TASKS = ("spiral", "wave", "combined")


def _model_specs(seed: int) -> list[tuple[str, Any, list[dict[str, list[Any]]]]]:
    return [
        ("logistic_regression", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
         [{"classifier__C": [0.1, 1.0]}]),
        ("rbf_svm", SVC(kernel="rbf", class_weight="balanced", random_state=seed),
         [{"classifier__C": [1.0], "classifier__gamma": ["scale"]}]),
        ("random_forest", RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=seed, n_jobs=1),
         [{"classifier__max_features": ["sqrt"], "classifier__min_samples_leaf": [2]}]),
    ]


def _make_pipeline(n_features: int, *, include_task: bool) -> Pipeline:
    feature_columns = [f"feature_{index}" for index in range(n_features)]
    transformers = [("features", StandardScaler(), feature_columns)]
    if include_task:
        transformers.append(("task", OneHotEncoder(handle_unknown="ignore"), ["task"]))
    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

    return Pipeline([("preprocessing", preprocessor), ("classifier", LogisticRegression())])


def _extract_representations(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cache image-local transforms once; learned scaling remains inside CV."""
    paths = manifest["path"].tolist()
    return {
        "pixels32_baseline": GrayscaleVectorizer(size=32).fit_transform(paths),
        "aspect_pixels64": DrawingImageVectorizer(representation="aspect_pixels", size=64).fit_transform(paths),
        "handcrafted": DrawingImageVectorizer(representation="handcrafted", size=64).fit_transform(paths),
        "combined64": DrawingImageVectorizer(representation="combined", size=64).fit_transform(paths),
    }


def _vectorizer_for(representation: str) -> Any:
    if representation == "pixels32_baseline":
        return GrayscaleVectorizer(size=32)
    kind = {"aspect_pixels64": "aspect_pixels", "handcrafted": "handcrafted", "combined64": "combined"}[representation]
    return DrawingImageVectorizer(representation=kind, size=64)


def _model_input(matrix: np.ndarray, tasks: pd.Series, *, include_task: bool) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, columns=[f"feature_{index}" for index in range(matrix.shape[1])])
    if include_task:
        frame["task"] = tasks.to_numpy()
    return frame


def predict_from_bundle(bundle: dict[str, Any], image_paths: list[str | Path], tasks: list[str] | None = None) -> np.ndarray:
    """Apply a saved deterministic image extractor and fitted estimator bundle."""
    if bundle["include_task"] and (tasks is None or len(tasks) != len(image_paths)):
        raise ValueError("One task value per image is required for a combined model")
    features = bundle["feature_extractor"].transform([str(path) for path in image_paths])
    frame = pd.DataFrame(features, columns=bundle["feature_columns"])
    if bundle["include_task"]:
        assert tasks is not None
        frame["task"] = tasks
    return np.asarray(bundle["pipeline"].predict(frame), dtype=int)


def _scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _evaluate(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(model.predict(X), dtype=int)
    score = _scores(model, X)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "n_images": int(len(y)),
        "class_support": {"healthy": int(np.sum(y == 0)), "parkinson": int(np.sum(y == 1))},
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision_parkinson": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        "recall_parkinson": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "f1_parkinson": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix_healthy_parkinson": matrix.tolist(),
        "classification_report": classification_report(
            y, predicted, labels=[0, 1], target_names=["healthy", "parkinson"],
            output_dict=True, zero_division=0,
        ),
    }


def _group_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    group_counts = pd.DataFrame({"target": y, "group": groups}).drop_duplicates().groupby("target")["group"].nunique()
    folds = min(3, int(group_counts.min()))
    if folds < 2:
        raise ValueError("At least two candidate groups per class are needed for grouped CV")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups))
    for train_idx, validation_idx in splits:
        if set(groups[train_idx]) & set(groups[validation_idx]):
            raise AssertionError("Candidate-group overlap inside CV")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[validation_idx])) != 2:
            raise ValueError("A grouped CV fold lacks one or more classes")
    return splits


def run_comparisons(
    image_root: Path = DEFAULT_IMAGE_ROOT,
    results_dir: Path = PROJECT_ROOT / "results" / "drawing_representation_comparison",
    model_dir: Path = PROJECT_ROOT / "models" / "drawing_representation_comparison",
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    manifest = build_manifest(image_root)
    y_all = manifest["target"].to_numpy(dtype=int)
    groups_all = manifest["group"].to_numpy(dtype=object)
    train_idx, validation_idx, test_idx = make_grouped_splits(y_all, groups_all, seed)
    group_partition: dict[str, str] = {}
    for name, indices in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
        for group in groups_all[indices]:
            if group in group_partition and group_partition[group] != name:
                raise AssertionError("Candidate group assigned to multiple partitions")
            group_partition[str(group)] = name

    assignments = manifest[["path", "task", "label", "target", "group", "sha256"]].copy()
    assignments["row_id"] = np.arange(len(assignments))
    assignments["partition"] = assignments["group"].map(group_partition)
    if assignments["partition"].isna().any():
        raise AssertionError("Missing partition assignment")

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    assignments.assign(path=assignments["path"].map(lambda p: str(Path(p).relative_to(PROJECT_ROOT)))).to_csv(
        results_dir / "drawing_split_assignments.csv", index=False
    )
    # Image transforms are deterministic and use no cross-image statistics.
    # Cache them to avoid repeated file decoding in every CV estimator fit.
    representations = _extract_representations(manifest)

    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    final_eval: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    fold_assignments: list[dict[str, Any]] = []

    for task in TASKS:
        subset = assignments if task == "combined" else assignments[assignments["task"] == task]
        subset = subset.reset_index(drop=True)
        pool = subset[subset["partition"].isin(["train", "validation"])].reset_index(drop=True)
        test = subset[subset["partition"] == "test"].reset_index(drop=True)
        y_pool = pool["target"].to_numpy(dtype=int)
        groups_pool = pool["group"].to_numpy(dtype=object)
        pool_ids = pool["row_id"].to_numpy(dtype=int)
        cv_features = _model_input(representations["pixels32_baseline"][pool_ids], pool["task"], include_task=(task == "combined"))
        cv = _group_cv(cv_features, y_pool, groups_pool, seed=seed + 2)
        for fold, (tr, va) in enumerate(cv, start=1):
            for role, indices in (("train", tr), ("validation", va)):
                for row_index in indices:
                    fold_assignments.append({
                        "task": task,
                        "group": str(groups_pool[row_index]),
                        "fold": fold,
                        "role": role,
                    })

        candidates: list[tuple[float, float, str, str, GridSearchCV]] = []
        for representation in REPRESENTATIONS:
            matrix = representations[representation]
            X_pool = _model_input(matrix[pool_ids], pool["task"], include_task=(task == "combined"))
            for model_name, classifier, param_grid in _model_specs(seed):
                base = Pipeline([
                    ("preprocessing", _make_pipeline(matrix.shape[1], include_task=(task == "combined")).named_steps["preprocessing"]),
                    ("classifier", classifier),
                ])
                search = GridSearchCV(
                    base,
                    param_grid,
                    scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy", "f1": "f1"},
                    refit="roc_auc",
                    cv=cv,
                    n_jobs=1,
                    return_train_score=True,
                    error_score="raise",
                )
                search.fit(X_pool, y_pool)
                best_index = int(search.best_index_)
                cv_auc = float(search.cv_results_["mean_test_roc_auc"][best_index])
                cv_ba = float(search.cv_results_["mean_test_balanced_accuracy"][best_index])
                row = {
                    "task": task,
                    "representation": representation,
                    "model": model_name,
                    "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                    "cv_folds": len(cv),
                    "cv_roc_auc_mean": cv_auc,
                    "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][best_index]),
                    "cv_balanced_accuracy_mean": cv_ba,
                    "cv_f1_mean": float(search.cv_results_["mean_test_f1"][best_index]),
                    "cv_train_roc_auc_mean": float(search.cv_results_["mean_train_roc_auc"][best_index]),
                    "selected_by_cv": False,
                    "test_metrics": None,
                }
                rows.append(row)
                candidates.append((cv_auc, cv_ba, representation, model_name, search))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, representation, model_name, selected_search = candidates[0]
        selected_row = next(
            row for row in rows
            if row["task"] == task and row["representation"] == representation and row["model"] == model_name
        )
        selected_row["selected_by_cv"] = True
        selected_matrix = representations[representation]
        model_path = model_dir / f"{task}_{representation}_{model_name}.joblib"
        feature_extractor = _vectorizer_for(representation)
        feature_columns = [f"feature_{index}" for index in range(selected_matrix.shape[1])]
        joblib.dump({
            "pipeline": selected_search.best_estimator_,
            "feature_extractor": feature_extractor,
            "feature_columns": feature_columns,
            "include_task": task == "combined",
            "task": task,
            "representation": representation,
            "selected_model": model_name,
            "label_mapping": LABEL_TO_TARGET,
            "group_note": "Class-scoped V## candidate token; participant identity unverified.",
            "seed": seed,
        }, model_path)
        y_test = test["target"].to_numpy(dtype=int)
        X_test = _model_input(selected_matrix[test["row_id"].to_numpy(dtype=int)], test["task"], include_task=(task == "combined"))
        metrics = _evaluate(selected_search.best_estimator_, X_test, y_test)
        selected_row["test_metrics"] = json.dumps(metrics, sort_keys=True)
        selected[task] = {
            "representation": representation,
            "model": model_name,
            "best_parameters": selected_search.best_params_,
            "cv_roc_auc_mean": float(candidates[0][0]),
            "cv_balanced_accuracy_mean": float(candidates[0][1]),
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "test_metrics": metrics,
        }
        final_eval[task] = metrics
        predictions = selected_search.best_estimator_.predict(X_test).astype(int)
        scores = _scores(selected_search.best_estimator_, X_test)
        for idx, row in test.iterrows():
            prediction_rows.append({
                "task_evaluation": task,
                "candidate_group": row["group"],
                "task": row["task"],
                "true_target": int(row["target"]),
                "predicted_target": int(predictions[idx]),
                "decision_score": float(scores[idx]),
                "file_name": Path(row["path"]).name,
            })

    comparison_table = pd.DataFrame(rows)
    comparison_table.to_csv(results_dir / "drawing_cv_comparison.csv", index=False)
    pd.DataFrame(fold_assignments).drop_duplicates().sort_values(["task", "fold", "role", "group"]).to_csv(
        results_dir / "drawing_cv_group_assignments.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(results_dir / "drawing_test_predictions.csv", index=False)

    split_summary: dict[str, Any] = {}
    for partition in ("train", "validation", "test"):
        part = assignments[assignments["partition"] == partition]
        group_support = part[["target", "group"]].drop_duplicates().groupby("target")["group"].nunique()
        split_summary[partition] = {
            "images": int(len(part)),
            "candidate_groups": int(part["group"].nunique()),
            "group_support": {"healthy": int(group_support.get(0, 0)), "parkinson": int(group_support.get(1, 0))},
            "image_support": {"healthy": int((part["target"] == 0).sum()), "parkinson": int((part["target"] == 1).sum())},
        }
    summary = {
        "study_scope": "Exploratory retrospective classification of existing labels; not prospective risk prediction.",
        "identity_limit": "V## tokens are class-scoped candidate groups only; source documentation does not verify participant identity.",
        "seed": seed,
        "input_root": str(image_root.relative_to(PROJECT_ROOT)) if image_root.is_relative_to(PROJECT_ROOT) else str(image_root),
        "n_images": int(len(manifest)),
        "n_candidate_groups": int(manifest["group"].nunique()),
        "label_mapping": LABEL_TO_TARGET,
        "representations": {
            "pixels32_baseline": "Original direct 32x32 grayscale resize; 1,024 pixels.",
            "aspect_pixels64": "64x64 grayscale letterbox, preserving original aspect ratio.",
            "handcrafted": "Per-image Otsu foreground fraction, 4x4 occupancy, centroid/covariance, 12-bin radial occupancy and 4x4x8 unsigned gradient histogram.",
            "combined64": "Concatenated aspect-preserving 64x64 grayscale pixels and handcrafted descriptors.",
        },
        "partition_policy": "One deterministic class-scoped candidate-group partition reused across tasks and representations; original folder split ignored because groups cross its boundaries.",
        "model_selection": "Best grouped-CV ROC-AUC, balanced accuracy tie-breaker, across predefined representation/model/grid candidates within the train+validation groups. Test excluded from all selection.",
        "preprocessing_policy": "Fixed per-image transforms are cached once and use no cross-image statistics. StandardScaler and estimator are fitted within each GridSearchCV training fold; selected pipeline refits on train+validation only.",
        "split_summary": split_summary,
        "selected_candidates": selected,
        "test_metrics": final_eval,
        "claim_limit": "No verified participant-independent generalization; test results are exploratory candidate-group holdout results with four groups globally in test.",
    }
    (results_dir / "drawing_comparison_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_lines = [
        "# Drawing representation comparison",
        "",
        "Exploratory results only. Filename-derived groups are unverified participant candidates.",
        "The grouped test set contains four candidate groups; do not infer reliable participant-independent generalization.",
        "",
        "## Cross-validation comparisons",
        "",
        "| Task | Representation | Model | Grouped CV ROC-AUC (mean ± SD) | Balanced accuracy | F1 | Selected |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['task']} | {row['representation']} | {row['model']} | "
            f"{row['cv_roc_auc_mean']:.3f} ± {row['cv_roc_auc_std']:.3f} | "
            f"{row['cv_balanced_accuracy_mean']:.3f} | {row['cv_f1_mean']:.3f} | "
            f"{'yes' if row['selected_by_cv'] else ''} |"
        )
    report_lines.extend(["", "## Selected test evaluations", "", "Metrics and supports are saved in `drawing_comparison_metrics.json`.", ""])
    for task, result in selected.items():
        report_lines.append(
            f"- **{task}:** {result['representation']} + {result['model']}; "
            f"test accuracy {result['test_metrics']['accuracy']:.3f}, "
            f"balanced accuracy {result['test_metrics']['balanced_accuracy']:.3f}, "
            f"ROC-AUC {result['test_metrics']['roc_auc']:.3f}; "
            f"support {result['test_metrics']['class_support']}; confusion "
            f"`{result['test_metrics']['confusion_matrix_healthy_parkinson']}`."
        )
    report_lines.extend(["", "The final test partition was not used in model or representation selection. These evaluations remain candidate-group holdouts, not verified participant holdouts.", ""])
    (results_dir / "drawing_comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"split_summary": split_summary, "selected": selected, "results_dir": str(results_dir), "model_dir": str(model_dir)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "drawing_representation_comparison")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "drawing_representation_comparison")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparisons(args.image_root, args.results_dir, args.model_dir, seed=args.seed)


if __name__ == "__main__":
    main()
