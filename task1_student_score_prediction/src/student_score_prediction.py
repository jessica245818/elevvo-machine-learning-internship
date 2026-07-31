"""Train and compare regression models for student exam-score prediction."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


RANDOM_STATE = 42
TARGET = "Exam_Score"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "lainguyn123/student-performance-factors"
)
BASELINE_FEATURES = ["Hours_Studied"]
SELECTED_NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict student exam scores and compare regression models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "task1_student_score_prediction/data/raw/"
            "StudentPerformanceFactors.csv"
        ),
        help="Path to the Student Performance Factors CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task1_student_score_prediction/outputs"),
        help="Directory for metrics, predictions, and figures.",
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    """Download the public CC0 Kaggle dataset when it is not available locally."""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found. Downloading from {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(
                f"Expected one CSV in the dataset archive, found {len(csv_names)}."
            )
        with archive.open(csv_names[0]) as source, path.open("wb") as destination:
            destination.write(source.read())


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dataset(path)
    data = pd.read_csv(path)
    original_rows = len(data)
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    invalid_target_mask = data[TARGET].isna() | ~data[TARGET].between(0, 100)
    invalid_target_rows = int(invalid_target_mask.sum())
    data = data.loc[~invalid_target_mask].reset_index(drop=True)

    summary = {
        "source_file": str(path),
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicate_rows,
        "invalid_target_rows_removed": invalid_target_rows,
        "clean_rows": len(data),
        "columns": len(data.columns),
        "missing_values_by_column": {
            column: int(count)
            for column, count in data.isna().sum().items()
            if count > 0
        },
    }
    return data, summary


def make_model_definitions(
    data: pd.DataFrame,
) -> dict[str, tuple[list[str], Pipeline | LinearRegression]]:
    feature_columns = [column for column in data.columns if column != TARGET]
    numeric_columns = data[feature_columns].select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]

    all_feature_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    return {
        "Study hours - linear": (
            BASELINE_FEATURES,
            LinearRegression(),
        ),
        "Study hours - polynomial (degree 2)": (
            BASELINE_FEATURES,
            Pipeline(
                [
                    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                    ("linear_regression", LinearRegression()),
                ]
            ),
        ),
        "Selected numeric features - linear": (
            SELECTED_NUMERIC_FEATURES,
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("linear_regression", LinearRegression()),
                ]
            ),
        ),
        "All features - linear": (
            feature_columns,
            Pipeline(
                [
                    ("preprocessor", all_feature_preprocessor),
                    ("linear_regression", LinearRegression()),
                ]
            ),
        ),
    }


def evaluate_models(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    y_train = data.iloc[train_indices][TARGET]
    y_test = data.iloc[test_indices][TARGET]

    model_definitions = make_model_definitions(data)
    metric_rows: list[dict[str, float | str | int]] = []
    predictions: dict[str, np.ndarray] = {}

    for model_name, (feature_columns, model) in model_definitions.items():
        x_train = data.iloc[train_indices][feature_columns]
        x_test = data.iloc[test_indices][feature_columns]
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        predictions[model_name] = predicted

        metric_rows.append(
            {
                "Model": model_name,
                "Features": len(feature_columns),
                "MAE": mean_absolute_error(y_test, predicted),
                "RMSE": mean_squared_error(y_test, predicted) ** 0.5,
                "R2": r2_score(y_test, predicted),
            }
        )

    test_results = data.iloc[test_indices][BASELINE_FEATURES + [TARGET]].copy()
    test_results["Predicted_Exam_Score"] = predictions["Study hours - linear"]
    test_results["Residual"] = (
        test_results[TARGET] - test_results["Predicted_Exam_Score"]
    )
    return pd.DataFrame(metric_rows), predictions, test_results


def save_eda_figures(data: pd.DataFrame, figure_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(data["Hours_Studied"], bins=20, color="#2563EB", edgecolor="white")
    axes[0].set(
        title="Distribution of Weekly Study Hours",
        xlabel="Hours studied",
        ylabel="Number of students",
    )
    axes[1].hist(data[TARGET], bins=20, color="#F97316", edgecolor="white")
    axes[1].set(
        title="Distribution of Exam Scores",
        xlabel="Exam score",
        ylabel="Number of students",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "01_distributions.png", dpi=180)
    plt.close(fig)

    numeric_data = data.select_dtypes(include=np.number)
    correlations = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(correlations.index)), correlations.index)
    ax.set_title("Correlation Matrix for Numeric Variables")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_correlation_matrix.png", dpi=180)
    plt.close(fig)


def save_model_figures(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    test_results: pd.DataFrame,
    figure_dir: Path,
) -> None:
    baseline = LinearRegression()
    baseline.fit(data[BASELINE_FEATURES], data[TARGET])
    x_line = np.linspace(
        data["Hours_Studied"].min(),
        data["Hours_Studied"].max(),
        200,
    ).reshape(-1, 1)
    y_line = baseline.predict(
        pd.DataFrame(x_line, columns=BASELINE_FEATURES)
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        data["Hours_Studied"],
        data[TARGET],
        alpha=0.25,
        s=16,
        color="#2563EB",
        label="Students",
    )
    ax.plot(x_line[:, 0], y_line, color="#DC2626", linewidth=2.5, label="Linear fit")
    ax.set(
        title="Exam Score vs Weekly Study Hours",
        xlabel="Hours studied",
        ylabel="Exam score",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "03_study_hours_regression.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    actual = test_results[TARGET]
    predicted = test_results["Predicted_Exam_Score"]
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    axes[0].scatter(actual, predicted, alpha=0.5, color="#2563EB")
    axes[0].plot([lower, upper], [lower, upper], "--", color="#DC2626")
    axes[0].set(
        title="Actual vs Predicted Scores",
        xlabel="Actual exam score",
        ylabel="Predicted exam score",
    )
    axes[1].scatter(predicted, test_results["Residual"], alpha=0.5, color="#7C3AED")
    axes[1].axhline(0, linestyle="--", color="#DC2626")
    axes[1].set(
        title="Baseline Residual Plot",
        xlabel="Predicted exam score",
        ylabel="Residual (actual - predicted)",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "04_baseline_diagnostics.png", dpi=180)
    plt.close(fig)

    ordered = metrics.sort_values("R2")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(ordered["Model"], ordered["R2"], color="#2563EB")
    axes[0].set(title="Model Comparison: R²", xlabel="R²")
    axes[1].barh(ordered["Model"], ordered["MAE"], color="#F97316")
    axes[1].set(title="Model Comparison: MAE", xlabel="Mean absolute error")
    fig.tight_layout()
    fig.savefig(figure_dir / "05_model_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, cleaning_summary = load_and_clean_data(args.data)
    metrics, _, test_results = evaluate_models(data)

    save_eda_figures(data, figure_dir)
    save_model_figures(data, metrics, test_results, figure_dir)

    metrics.to_csv(output_dir / "model_metrics.csv", index=False, float_format="%.4f")
    test_results.to_csv(output_dir / "baseline_test_predictions.csv", index=False)
    with (output_dir / "cleaning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(cleaning_summary, file, indent=2)

    print("\nCleaning summary")
    print(json.dumps(cleaning_summary, indent=2))
    print("\nModel metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
