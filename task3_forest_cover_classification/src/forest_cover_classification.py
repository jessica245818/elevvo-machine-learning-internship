"""Compare tuned Random Forest and XGBoost models on UCI Covertype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET = "Cover_Type"
CLASS_NAMES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare forest cover type classifiers."
    )
    parser.add_argument(
        "--data-home",
        type=Path,
        default=Path("task3_forest_cover_classification/data"),
        help="Cache directory used by scikit-learn for the UCI dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3_forest_cover_classification/outputs"),
        help="Directory for Task 3 outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120_000,
        help="Stratified working sample size; use 0 for the full dataset.",
    )
    parser.add_argument(
        "--tuning-size",
        type=int,
        default=40_000,
        help="Maximum stratified subset used for hyperparameter tuning.",
    )
    return parser.parse_args()


def load_and_prepare(
    data_home: Path,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    dataset = fetch_covtype(
        data_home=data_home,
        download_if_missing=True,
        as_frame=True,
    )
    features = dataset.data.copy()
    target = dataset.target.astype(int).copy()

    original_rows = len(features)
    missing_values = int(features.isna().sum().sum() + target.isna().sum())
    combined = features.copy()
    combined[TARGET] = target
    duplicate_rows = int(combined.duplicated().sum())
    if duplicate_rows:
        combined = combined.drop_duplicates().reset_index(drop=True)
        features = combined.drop(columns=TARGET)
        target = combined[TARGET]

    wilderness_columns = [
        column for column in features if column.startswith("Wilderness_Area")
    ]
    soil_columns = [
        column for column in features if column.startswith("Soil_Type")
    ]
    if len(wilderness_columns) != 4 or len(soil_columns) != 40:
        raise ValueError(
            "Expected four wilderness and 40 soil one-hot columns; found "
            f"{len(wilderness_columns)} and {len(soil_columns)}."
        )

    categorical_values = set(
        np.unique(features[wilderness_columns + soil_columns].to_numpy())
    )
    if not categorical_values.issubset({0, 1}):
        raise ValueError("Categorical indicator columns contain non-binary values.")

    wilderness_encoding_violations = int(
        (features[wilderness_columns].sum(axis=1) != 1).sum()
    )
    soil_encoding_violations = int(
        (features[soil_columns].sum(axis=1) != 1).sum()
    )

    if sample_size > 0 and sample_size < len(features):
        features, _, target, _ = train_test_split(
            features,
            target,
            train_size=sample_size,
            stratify=target,
            random_state=RANDOM_STATE,
        )
        features = features.reset_index(drop=True)
        target = target.reset_index(drop=True)

    # XGBoost requires zero-based class labels; reports convert them back to 1-7.
    target = target - 1

    summary = {
        "source": "UCI Machine Learning Repository - Covertype",
        "uci_doi": "10.24432/C50K5N",
        "original_rows": original_rows,
        "original_features": int(dataset.data.shape[1]),
        "duplicate_rows_removed": duplicate_rows,
        "missing_values": missing_values,
        "working_rows": len(features),
        "numeric_features": 10,
        "wilderness_one_hot_features": len(wilderness_columns),
        "soil_one_hot_features": len(soil_columns),
        "wilderness_encoding_violations": wilderness_encoding_violations,
        "soil_encoding_violations": soil_encoding_violations,
        "classes": 7,
    }
    return features, target, summary


def make_model(model_name: str, parameters: dict[str, object]):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            **parameters,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **parameters,
        )
    raise ValueError(f"Unknown model: {model_name}")


def tune_model(
    model_name: str,
    parameter_grid: dict[str, list[object]],
    x_tune: pd.DataFrame,
    y_tune: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame]:
    x_fit, x_validate, y_fit, y_validate = train_test_split(
        x_tune,
        y_tune,
        test_size=0.25,
        stratify=y_tune,
        random_state=RANDOM_STATE,
    )
    rows: list[dict[str, object]] = []

    for parameters in ParameterGrid(parameter_grid):
        model = make_model(model_name, parameters)
        start = time.perf_counter()
        model.fit(x_fit, y_fit)
        predictions = model.predict(x_validate)
        fit_seconds = time.perf_counter() - start
        rows.append(
            {
                "model": model_name,
                "parameters": json.dumps(parameters, sort_keys=True),
                "validation_accuracy": accuracy_score(y_validate, predictions),
                "validation_macro_f1": f1_score(
                    y_validate,
                    predictions,
                    average="macro",
                ),
                "fit_seconds": fit_seconds,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=False,
    )
    best_parameters = json.loads(results.iloc[0]["parameters"])
    return best_parameters, results


def stratified_subset(
    features: pd.DataFrame,
    target: pd.Series,
    size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if size <= 0 or size >= len(features):
        return features, target
    subset_features, _, subset_target, _ = train_test_split(
        features,
        target,
        train_size=size,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    return subset_features, subset_target


def evaluate_model(
    model_name: str,
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame, pd.DataFrame]:
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test)
    predict_seconds = time.perf_counter() - start

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(7),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    importance = (
        pd.DataFrame(
            {
                "feature": x_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return metrics, predictions, report, importance


def save_class_distribution(target: pd.Series, figure_dir: Path) -> None:
    counts = target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASS_NAMES, counts.values, color="#2563EB")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set(
        title="Forest Cover Type Distribution in Working Sample",
        xlabel="Cover type",
        ylabel="Rows",
    )
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_class_distribution.png", dpi=180)
    plt.close(fig)


def save_confusion_matrices(
    y_test: pd.Series,
    predictions_by_model: dict[str, np.ndarray],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, (model_name, predictions) in zip(
        axes,
        predictions_by_model.items(),
    ):
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(7),
            normalize="true",
        )
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for row in range(7):
            for column in range(7):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=8,
                )
        ax.set(
            title=f"{model_name}\nNormalized Confusion Matrix",
            xlabel="Predicted class",
            ylabel="True class",
            xticks=np.arange(7),
            yticks=np.arange(7),
        )
        ax.set_xticklabels(range(1, 8))
        ax.set_yticklabels(range(1, 8))
    fig.tight_layout()
    fig.savefig(
        figure_dir / "02_confusion_matrices.png",
        dpi=180,
    )
    plt.close(fig)


def save_feature_importance(
    importance_by_model: dict[str, pd.DataFrame],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (model_name, importance) in zip(
        axes,
        importance_by_model.items(),
    ):
        top = importance.head(15).sort_values("importance")
        ax.barh(top["feature"], top["importance"], color="#F97316")
        ax.set(
            title=f"{model_name}\nTop 15 Feature Importances",
            xlabel="Importance",
        )
    fig.tight_layout()
    fig.savefig(figure_dir / "03_feature_importance.png", dpi=180)
    plt.close(fig)


def save_model_comparison(
    model_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    positions = np.arange(len(model_metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        positions - width,
        model_metrics["accuracy"],
        width,
        label="Accuracy",
        color="#2563EB",
    )
    ax.bar(
        positions,
        model_metrics["balanced_accuracy"],
        width,
        label="Balanced accuracy",
        color="#F97316",
    )
    ax.bar(
        positions + width,
        model_metrics["macro_f1"],
        width,
        label="Macro F1",
        color="#7C3AED",
    )
    ax.set(
        title="Forest Cover Model Comparison",
        ylabel="Score",
        ylim=(0, 1),
        xticks=positions,
    )
    ax.set_xticklabels(model_metrics["model"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "04_model_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    features, target, cleaning_summary = load_and_prepare(
        args.data_home,
        args.sample_size,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    x_tune, y_tune = stratified_subset(
        x_train,
        y_train,
        args.tuning_size,
    )

    grids = {
        "Random Forest": {
            "n_estimators": [200],
            "max_depth": [None, 24],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        },
        "XGBoost": {
            "n_estimators": [250],
            "max_depth": [6, 10],
            "learning_rate": [0.08, 0.15],
            "subsample": [0.9],
            "colsample_bytree": [0.8],
        },
    }

    tuning_frames: list[pd.DataFrame] = []
    best_parameters_by_model: dict[str, dict[str, object]] = {}
    for model_name, grid in grids.items():
        best_parameters, tuning_results = tune_model(
            model_name,
            grid,
            x_tune,
            y_tune,
        )
        best_parameters_by_model[model_name] = best_parameters
        tuning_frames.append(tuning_results)

    metrics_rows: list[dict[str, object]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    importance_by_model: dict[str, pd.DataFrame] = {}

    for model_name, parameters in best_parameters_by_model.items():
        model = make_model(model_name, parameters)
        metrics, predictions, report, importance = evaluate_model(
            model_name,
            model,
            x_train,
            y_train,
            x_test,
            y_test,
        )
        metrics["best_parameters"] = json.dumps(parameters, sort_keys=True)
        metrics_rows.append(metrics)
        predictions_by_model[model_name] = predictions
        importance_by_model[model_name] = importance
        report.to_csv(
            output_dir
            / f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
        )
        importance.to_csv(
            output_dir
            / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv",
            index=False,
        )

    model_metrics = pd.DataFrame(metrics_rows)
    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    tuning_results.to_csv(output_dir / "tuning_results.csv", index=False)
    with (output_dir / "cleaning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(cleaning_summary, file, indent=2)
    with (output_dir / "best_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(best_parameters_by_model, file, indent=2)

    plt.style.use("seaborn-v0_8-whitegrid")
    save_class_distribution(target, figure_dir)
    save_confusion_matrices(y_test, predictions_by_model, figure_dir)
    save_feature_importance(importance_by_model, figure_dir)
    save_model_comparison(model_metrics, figure_dir)

    print("\nCleaning and preprocessing summary")
    print(json.dumps(cleaning_summary, indent=2))
    print("\nBest hyperparameters")
    print(json.dumps(best_parameters_by_model, indent=2))
    print("\nTest metrics")
    print(
        model_metrics.drop(columns="best_parameters").to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nTop five features by model")
    for model_name, importance in importance_by_model.items():
        print(f"\n{model_name}")
        print(
            importance.head(5).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
