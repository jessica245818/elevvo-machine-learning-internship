"""Predict loan approval with imbalance-aware classification pipelines."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET = "Loan_Status"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/maramsa/loan-database"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare imbalance-aware loan approval classifiers."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "task4_loan_approval_prediction/data/raw/loan_approval.xlsx"
        ),
        help="Path to the loan approval workbook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task4_loan_approval_prediction/outputs"),
        help="Directory for Task 4 outputs.",
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    """Download the public Kaggle workbook when it is missing."""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found. Downloading from {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        workbook_names = [
            name for name in archive.namelist() if name.lower().endswith(".xlsx")
        ]
        if len(workbook_names) != 1:
            raise RuntimeError(
                f"Expected one workbook in the archive, found {len(workbook_names)}."
            )
        with archive.open(workbook_names[0]) as source, path.open("wb") as destination:
            destination.write(source.read())


def load_and_clean(path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    ensure_dataset(path)
    data = pd.read_excel(path)
    original_rows = len(data)
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    required_columns = {
        "Loan_ID",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        TARGET,
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    missing_by_column = {
        column: int(count)
        for column, count in data.isna().sum().items()
        if count > 0
    }
    invalid_target_rows = int((~data[TARGET].isin(["Y", "N"])).sum())
    data = data.loc[data[TARGET].isin(["Y", "N"])].reset_index(drop=True)

    features = data.drop(columns=[TARGET, "Loan_ID"])
    target = data[TARGET].map({"N": 0, "Y": 1}).astype(int)
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    for column in categorical_columns:
        features[column] = features[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    class_counts = target.value_counts().sort_index()
    summary = {
        "source_file": str(path),
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicate_rows,
        "invalid_target_rows_removed": invalid_target_rows,
        "clean_rows": len(features),
        "identifier_removed": "Loan_ID",
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "missing_values_by_column": missing_by_column,
        "rejected_applications": int(class_counts.get(0, 0)),
        "approved_applications": int(class_counts.get(1, 0)),
        "approval_rate": float(target.mean()),
    }
    return features, target, summary


def make_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    classifier_name: str,
    classifier_parameters: dict[str, object],
    use_smote: bool,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0,
    )

    numeric_indices = list(range(len(numeric_columns)))
    categorical_indices = list(
        range(
            len(numeric_columns),
            len(numeric_columns) + len(categorical_columns),
        )
    )
    encoder = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric_indices),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_indices,
            ),
        ],
        sparse_threshold=0,
    )

    if classifier_name == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=2_000,
            random_state=RANDOM_STATE,
            solver="liblinear",
            **classifier_parameters,
        )
    elif classifier_name == "Decision Tree":
        classifier = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            **classifier_parameters,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTENC(
                    categorical_features=categorical_indices,
                    random_state=RANDOM_STATE,
                    k_neighbors=5,
                ),
            )
        )
    steps.extend([("encoder", encoder), ("classifier", classifier)])
    return Pipeline(steps)


def tune_variant(
    variant_name: str,
    classifier_name: str,
    use_smote: bool,
    base_class_weight: str | None,
    parameter_grid: dict[str, list[object]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    x_fit, x_validate, y_fit, y_validate = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    rows: list[dict[str, object]] = []

    for parameters in ParameterGrid(parameter_grid):
        classifier_parameters = dict(parameters)
        if base_class_weight is not None:
            classifier_parameters["class_weight"] = base_class_weight
        pipeline = make_pipeline(
            categorical_columns,
            numeric_columns,
            classifier_name,
            classifier_parameters,
            use_smote,
        )
        start = time.perf_counter()
        pipeline.fit(x_fit, y_fit)
        predictions = pipeline.predict(x_validate)
        fit_seconds = time.perf_counter() - start
        rejected_precision, rejected_recall, rejected_f1, _ = (
            precision_recall_fscore_support(
                y_validate,
                predictions,
                labels=[0],
                average=None,
                zero_division=0,
            )
        )
        rows.append(
            {
                "variant": variant_name,
                "classifier": classifier_name,
                "parameters": json.dumps(classifier_parameters, sort_keys=True),
                "validation_macro_f1": f1_score(
                    y_validate,
                    predictions,
                    average="macro",
                ),
                "validation_rejected_precision": rejected_precision[0],
                "validation_rejected_recall": rejected_recall[0],
                "validation_rejected_f1": rejected_f1[0],
                "fit_seconds": fit_seconds,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_rejected_f1"],
        ascending=False,
    )
    best_parameters = json.loads(results.iloc[0]["parameters"])
    return best_parameters, results


def evaluate_variant(
    variant_name: str,
    classifier_name: str,
    use_smote: bool,
    parameters: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame]:
    pipeline = make_pipeline(
        categorical_columns,
        numeric_columns,
        classifier_name,
        parameters,
        use_smote,
    )
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    predictions = pipeline.predict(x_test)

    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Rejected", "Approved"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    rejected = report.loc["Rejected"]
    approved = report.loc["Approved"]
    metrics = {
        "variant": variant_name,
        "classifier": classifier_name,
        "balancing": "SMOTE"
        if use_smote
        else ("Class Weight" if parameters.get("class_weight") else "None"),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "rejected_precision": rejected["precision"],
        "rejected_recall": rejected["recall"],
        "rejected_f1": rejected["f1-score"],
        "approved_precision": approved["precision"],
        "approved_recall": approved["recall"],
        "approved_f1": approved["f1-score"],
        "fit_seconds": fit_seconds,
        "best_parameters": json.dumps(parameters, sort_keys=True),
    }
    return metrics, predictions, report


def save_data_overview(
    target: pd.Series,
    cleaning_summary: dict[str, object],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    counts = target.value_counts().sort_index()
    bars = axes[0].bar(
        ["Rejected", "Approved"],
        counts.values,
        color=["#DC2626", "#16A34A"],
    )
    axes[0].bar_label(bars, padding=3)
    axes[0].set(
        title="Loan Approval Class Distribution",
        ylabel="Applications",
    )

    missing = pd.Series(cleaning_summary["missing_values_by_column"]).sort_values()
    axes[1].barh(missing.index, missing.values, color="#F97316")
    axes[1].set(
        title="Missing Values Before Pipeline Imputation",
        xlabel="Missing rows",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "01_data_overview.png", dpi=180)
    plt.close(fig)


def save_metric_comparison(
    model_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    ordered = model_metrics.sort_values("macro_f1")
    positions = np.arange(len(ordered))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(
        positions - width,
        ordered["rejected_precision"],
        width,
        label="Rejected precision",
        color="#2563EB",
    )
    ax.barh(
        positions,
        ordered["rejected_recall"],
        width,
        label="Rejected recall",
        color="#F97316",
    )
    ax.barh(
        positions + width,
        ordered["macro_f1"],
        width,
        label="Macro F1",
        color="#7C3AED",
    )
    ax.set(
        title="Imbalance-Aware Model Comparison",
        xlabel="Score",
        xlim=(0, 1),
        yticks=positions,
    )
    ax.set_yticklabels(ordered["variant"])
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_metric_comparison.png", dpi=180)
    plt.close(fig)


def save_confusion_matrices(
    y_test: pd.Series,
    selected_predictions: dict[str, np.ndarray],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (variant, predictions) in zip(
        axes,
        selected_predictions.items(),
    ):
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() / 2 else "black",
                    fontsize=13,
                )
        ax.set(
            title=variant,
            xlabel="Predicted",
            ylabel="Actual",
            xticks=[0, 1],
            yticks=[0, 1],
        )
        ax.set_xticklabels(["Rejected", "Approved"])
        ax.set_yticklabels(["Rejected", "Approved"])
    fig.suptitle("Best Logistic Regression vs. Best Decision Tree")
    fig.tight_layout()
    fig.savefig(figure_dir / "03_confusion_matrices.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    features, target, cleaning_summary = load_and_clean(args.data)
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    variants = [
        {
            "variant": "Logistic Regression",
            "classifier": "Logistic Regression",
            "smote": False,
            "class_weight": None,
            "grid": {"C": [0.1, 1.0, 10.0]},
        },
        {
            "variant": "Logistic Regression + Class Weight",
            "classifier": "Logistic Regression",
            "smote": False,
            "class_weight": "balanced",
            "grid": {"C": [0.1, 1.0, 10.0]},
        },
        {
            "variant": "Logistic Regression + SMOTE",
            "classifier": "Logistic Regression",
            "smote": True,
            "class_weight": None,
            "grid": {"C": [0.1, 1.0, 10.0]},
        },
        {
            "variant": "Decision Tree",
            "classifier": "Decision Tree",
            "smote": False,
            "class_weight": None,
            "grid": {
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [5, 15],
            },
        },
        {
            "variant": "Decision Tree + Class Weight",
            "classifier": "Decision Tree",
            "smote": False,
            "class_weight": "balanced",
            "grid": {
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [5, 15],
            },
        },
        {
            "variant": "Decision Tree + SMOTE",
            "classifier": "Decision Tree",
            "smote": True,
            "class_weight": None,
            "grid": {
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [5, 15],
            },
        },
    ]

    tuning_frames: list[pd.DataFrame] = []
    best_parameters: dict[str, dict[str, object]] = {}
    metrics_rows: list[dict[str, object]] = []
    predictions_by_variant: dict[str, np.ndarray] = {}

    for specification in variants:
        parameters, tuning_results = tune_variant(
            specification["variant"],
            specification["classifier"],
            specification["smote"],
            specification["class_weight"],
            specification["grid"],
            x_train,
            y_train,
            categorical_columns,
            numeric_columns,
        )
        best_parameters[specification["variant"]] = parameters
        tuning_frames.append(tuning_results)

        metrics, predictions, report = evaluate_variant(
            specification["variant"],
            specification["classifier"],
            specification["smote"],
            parameters,
            x_train,
            y_train,
            x_test,
            y_test,
            categorical_columns,
            numeric_columns,
        )
        metrics_rows.append(metrics)
        predictions_by_variant[specification["variant"]] = predictions
        safe_name = specification["variant"].lower().replace(" ", "_").replace("+", "plus")
        report.to_csv(output_dir / f"{safe_name}_classification_report.csv")

    model_metrics = pd.DataFrame(metrics_rows).sort_values(
        "macro_f1",
        ascending=False,
    )
    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    tuning_results.to_csv(output_dir / "tuning_results.csv", index=False)
    with (output_dir / "cleaning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(cleaning_summary, file, indent=2)
    with (output_dir / "best_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(best_parameters, file, indent=2)

    best_logistic = model_metrics.loc[
        model_metrics["classifier"] == "Logistic Regression"
    ].iloc[0]["variant"]
    best_tree = model_metrics.loc[
        model_metrics["classifier"] == "Decision Tree"
    ].iloc[0]["variant"]

    plt.style.use("seaborn-v0_8-whitegrid")
    save_data_overview(target, cleaning_summary, figure_dir)
    save_metric_comparison(model_metrics, figure_dir)
    save_confusion_matrices(
        y_test,
        {
            best_logistic: predictions_by_variant[best_logistic],
            best_tree: predictions_by_variant[best_tree],
        },
        figure_dir,
    )

    print("\nCleaning summary")
    print(json.dumps(cleaning_summary, indent=2))
    print("\nTest metrics")
    display_columns = [
        "variant",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "rejected_precision",
        "rejected_recall",
        "rejected_f1",
        "approved_precision",
        "approved_recall",
        "approved_f1",
    ]
    print(
        model_metrics[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nBest logistic variant: {best_logistic}")
    print(f"Best decision-tree variant: {best_tree}")
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
