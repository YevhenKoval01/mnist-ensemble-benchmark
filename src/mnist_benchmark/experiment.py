"""Model selection, evaluation, persistence, and reporting."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from mnist_benchmark.data import DatasetSplit, augment_with_translations

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = ("decision_tree", "random_forest", "extra_trees", "xgboost")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]


@dataclass
class ModelResult:
    name: str
    best_parameters: dict[str, Any]
    cross_validation_accuracy: float
    test_accuracy: float
    training_seconds: float
    augmented_test_accuracy: float | None = None
    augmented_training_seconds: float | None = None


@dataclass
class ExperimentArtifacts:
    results: list[ModelResult]
    selected_model_name: str
    selected_model: BaseEstimator
    selected_predictions: np.ndarray
    classification_report: dict[str, Any]


def build_model_specs(
    selected_models: tuple[str, ...],
    *,
    random_state: int,
) -> list[ModelSpec]:
    """Build estimators lazily so optional models fail with a helpful message."""
    unknown_models = set(selected_models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    specs: dict[str, ModelSpec] = {
        "decision_tree": ModelSpec(
            name="decision_tree",
            estimator=DecisionTreeClassifier(random_state=random_state),
            parameter_grid={
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [10, 20, 30, None],
            },
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_jobs=1,
                random_state=random_state,
            ),
            parameter_grid={
                "n_estimators": [100, 200],
                "max_depth": [20, None],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "extra_trees": ModelSpec(
            name="extra_trees",
            estimator=ExtraTreesClassifier(
                n_jobs=1,
                random_state=random_state,
            ),
            parameter_grid={
                "n_estimators": [100, 200],
                "max_depth": [20, None],
            },
        ),
    }

    if "xgboost" in selected_models:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError(
                "XGBoost is selected but not installed. Run: pip install -e ."
            ) from error

        specs["xgboost"] = ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=10,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=1,
                random_state=random_state,
            ),
            parameter_grid={
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
            },
        )

    return [specs[name] for name in selected_models]


def run_model_selection(
    split: DatasetSplit,
    *,
    selected_models: tuple[str, ...],
    cv_folds: int,
    n_jobs: int,
    random_state: int,
    augment: bool,
) -> ExperimentArtifacts:
    """Tune models with cross-validation and evaluate them once on the test set."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    _, class_counts = np.unique(split.y_train, return_counts=True)
    if int(class_counts.min()) < cv_folds:
        raise ValueError(
            "cv_folds exceeds the smallest class count in the training data."
        )

    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    results: list[ModelResult] = []
    fitted_models: dict[str, BaseEstimator] = {}
    predictions: dict[str, np.ndarray] = {}

    for spec in build_model_specs(selected_models, random_state=random_state):
        LOGGER.info("Tuning %s", spec.name)
        search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.parameter_grid,
            scoring="accuracy",
            cv=cross_validation,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )

        started_at = time.perf_counter()
        search.fit(split.X_train, split.y_train)
        elapsed = time.perf_counter() - started_at

        model_predictions = search.best_estimator_.predict(split.X_test)
        result = ModelResult(
            name=spec.name,
            best_parameters=_json_safe(search.best_params_),
            cross_validation_accuracy=float(search.best_score_),
            test_accuracy=float(accuracy_score(split.y_test, model_predictions)),
            training_seconds=elapsed,
        )

        if augment:
            LOGGER.info("Refitting %s on augmented training data", spec.name)
            augmented_X, augmented_y = augment_with_translations(
                split.X_train,
                split.y_train,
            )
            augmented_model = clone(search.best_estimator_)
            augmentation_started_at = time.perf_counter()
            augmented_model.fit(augmented_X, augmented_y)
            result.augmented_training_seconds = (
                time.perf_counter() - augmentation_started_at
            )
            augmented_predictions = augmented_model.predict(split.X_test)
            result.augmented_test_accuracy = float(
                accuracy_score(split.y_test, augmented_predictions)
            )

            fitted_models[spec.name] = augmented_model
            predictions[spec.name] = augmented_predictions
        else:
            fitted_models[spec.name] = search.best_estimator_
            predictions[spec.name] = model_predictions

        results.append(result)
        LOGGER.info(
            "%s complete: CV accuracy %.4f, test accuracy %.4f",
            spec.name,
            result.cross_validation_accuracy,
            result.test_accuracy,
        )

    selected_result = max(results, key=lambda item: item.cross_validation_accuracy)
    selected_predictions = predictions[selected_result.name]
    report = classification_report(
        split.y_test,
        selected_predictions,
        output_dict=True,
        zero_division=0,
    )
    return ExperimentArtifacts(
        results=results,
        selected_model_name=selected_result.name,
        selected_model=fitted_models[selected_result.name],
        selected_predictions=selected_predictions,
        classification_report=report,
    )


def save_artifacts(
    artifacts: ExperimentArtifacts,
    split: DatasetSplit,
    output_dir: Path,
) -> None:
    """Persist metrics, the selected model, and diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "selection_rule": "highest cross-validation accuracy",
        "selected_model": artifacts.selected_model_name,
        "models": [asdict(result) for result in artifacts.results],
        "classification_report": artifacts.classification_report,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    _save_metrics_csv(artifacts.results, output_dir / "metrics.csv")
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            split.y_test,
            artifacts.selected_predictions,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    joblib.dump(artifacts.selected_model, output_dir / "best_model.joblib")

    _save_confusion_matrix(
        split.y_test,
        artifacts.selected_predictions,
        output_dir / "confusion_matrix.png",
    )
    _save_feature_importance(
        artifacts.selected_model,
        split.feature_names,
        output_dir / "feature_importance.png",
    )


def save_dataset_plots(
    split: DatasetSplit,
    output_dir: Path,
    *,
    show_plots: bool,
) -> None:
    """Save a class histogram and sample digit grid."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, counts = np.unique(
        np.concatenate((split.y_train, split.y_test)),
        return_counts=True,
    )
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, counts, color="#2563eb")
    axis.set_xticks(labels)
    axis.set(title="MNIST class distribution", xlabel="Digit", ylabel="Samples")
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=160)
    _close_or_show(figure, show_plots)

    sample_count = min(8, len(split.X_train))
    figure, axes = plt.subplots(2, 4, figsize=(8, 4.5))
    for axis, image, label in zip(
        axes.flat,
        split.X_train[:sample_count],
        split.y_train[:sample_count],
        strict=False,
    ):
        axis.imshow(image.reshape(28, 28), cmap="gray")
        axis.set_title(f"Label: {label}")
        axis.axis("off")
    for axis in axes.flat[sample_count:]:
        axis.axis("off")
    figure.suptitle("Training samples")
    figure.tight_layout()
    figure.savefig(output_dir / "training_samples.png", dpi=160)
    _close_or_show(figure, show_plots)


def _save_metrics_csv(results: list[ModelResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "cross_validation_accuracy",
                "test_accuracy",
                "training_seconds",
                "augmented_test_accuracy",
                "augmented_training_seconds",
                "best_parameters",
            ],
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["best_parameters"] = json.dumps(row["best_parameters"], sort_keys=True)
            writer.writerow(row)


def _save_confusion_matrix(
    expected: np.ndarray,
    predicted: np.ndarray,
    path: Path,
) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        expected,
        predicted,
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Selected model confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=160)
    plt.close(display.figure_)


def _save_feature_importance(
    model: BaseEstimator,
    feature_names: tuple[str, ...],
    path: Path,
) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    top_indices = np.argsort(importances)[-20:][::-1]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        np.array(feature_names)[top_indices],
        np.asarray(importances)[top_indices],
        color="#0f766e",
    )
    axis.set(title="Top 20 feature importances", xlabel="Pixel", ylabel="Importance")
    axis.tick_params(axis="x", rotation=65)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _close_or_show(figure: plt.Figure, show_plots: bool) -> None:
    if show_plots:
        plt.show()
    plt.close(figure)


def _json_safe(values: dict[str, Any]) -> dict[str, Any]:
    safe_values: dict[str, Any] = {}
    for key, value in values.items():
        safe_values[key] = value.item() if isinstance(value, np.generic) else value
    return safe_values
