"""Low-false-alarm predictive maintenance on the AI4I 2020 dataset."""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
TARGET = "Machine failure"
FAILURE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_NAMES = {
    "TWF": "Tool wear",
    "HDF": "Heat dissipation",
    "PWF": "Power",
    "OSF": "Overstrain",
    "RNF": "Random",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train low-false-discovery predictive-maintenance models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("task9_predictive_maintenance/data/raw/ai4i2020.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task9_predictive_maintenance/outputs"),
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.50,
        help="Minimum validation recall allowed during alarm threshold tuning.",
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AI4I data from {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected one CSV, found {csv_names}")
        with archive.open(csv_names[0]) as source, path.open("wb") as destination:
            destination.write(source.read())


def load_and_engineer(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    ensure_dataset(path)
    data = pd.read_csv(path)
    required = {
        "UDI", "Product ID", "Type", "Air temperature [K]",
        "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
        "Tool wear [min]", TARGET, *FAILURE_COLUMNS,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    duplicates = int(data.duplicated().sum())
    missing_values = int(data.isna().sum().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    sensors = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    features = data[["Type", *sensors]].copy()
    features["Temperature difference [K]"] = (
        features["Process temperature [K]"] - features["Air temperature [K]"]
    )
    angular_speed = features["Rotational speed [rpm]"] * 2 * np.pi / 60
    features["Mechanical power [W]"] = angular_speed * features["Torque [Nm]"]
    features["Tool stress"] = (
        features["Torque [Nm]"] * features["Tool wear [min]"]
    )
    features["Torque-speed ratio"] = features["Torque [Nm]"] / features[
        "Rotational speed [rpm]"
    ]
    target = data[TARGET].astype(int)
    failure_types = data[FAILURE_COLUMNS].astype(int)
    counts = failure_types.sum()
    summary = {
        "source": str(path),
        "rows": len(data),
        "duplicate_rows_removed": duplicates,
        "missing_values": missing_values,
        "failures": int(target.sum()),
        "normal_operations": int((1 - target).sum()),
        "failure_rate": float(target.mean()),
        "failure_type_counts": {
            FAILURE_NAMES[column]: int(counts[column])
            for column in FAILURE_COLUMNS
        },
        "untyped_failure_rows": int(
            ((target == 1) & (failure_types.sum(axis=1) == 0)).sum()
        ),
        "multi_failure_rows": int((failure_types.sum(axis=1) > 1).sum()),
        "engineered_features": [
            "Temperature difference [K]",
            "Mechanical power [W]",
            "Tool stress",
            "Torque-speed ratio",
        ],
    }
    return features, target, failure_types, summary


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Type"],
            ),
        ]
    )


def make_models(features: pd.DataFrame, positive_weight: float) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor(features)
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight={0: 1, 1: positive_weight},
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight={0: 1, 1: positive_weight},
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        reg_lambda=2,
                        scale_pos_weight=positive_weight,
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def tune_threshold(
    truth: pd.Series,
    probabilities: np.ndarray,
    minimum_recall: float,
) -> tuple[float, dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.99, 189):
        predicted = (probabilities >= threshold).astype(int)
        precision = precision_score(truth, predicted, zero_division=0)
        recall = recall_score(truth, predicted, zero_division=0)
        if recall >= minimum_recall and predicted.sum() > 0:
            candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "fdr": float(1 - precision),
                    "f1": float(f1_score(truth, predicted, zero_division=0)),
                }
            )
    if not candidates:
        raise RuntimeError("No threshold satisfies the minimum-recall constraint.")
    best = min(candidates, key=lambda row: (row["fdr"], -row["recall"], -row["f1"]))
    return best["threshold"], best


def binary_metrics(
    truth: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    precision = precision_score(truth, predicted, zero_division=0)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "false_discovery_rate": float(1 - precision),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "average_precision": float(average_precision_score(truth, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(predicted.sum()),
    }


def train_failure_type_models(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_types: pd.DataFrame,
    test_types: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, Pipeline]]:
    results: dict[str, dict] = {}
    models: dict[str, Pipeline] = {}
    for column in FAILURE_COLUMNS:
        positives = int(train_types[column].sum())
        weight = (len(train_types) - positives) / max(positives, 1)
        model = Pipeline(
            [
                ("preprocessor", make_preprocessor(train_x)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight={0: 1, 1: weight},
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        model.fit(train_x, train_types[column])
        probability = model.predict_proba(test_x)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_types[column], prediction, average="binary", zero_division=0
        )
        results[FAILURE_NAMES[column]] = {
            "test_support": int(test_types[column].sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "average_precision": float(
                average_precision_score(test_types[column], probability)
            ),
        }
        models[column] = model
    return results, models


def sensor_correlations(
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    sensor_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Temperature difference [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Mechanical power [W]",
        "Tool stress",
        "Torque-speed ratio",
    ]
    return {
        column: float(features[column].corr(target))
        for column in sensor_columns
    }


def save_figures(
    output_dir: Path,
    target: pd.Series,
    failure_types: pd.DataFrame,
    model_results: dict[str, dict],
    best_truth: pd.Series,
    best_probabilities: np.ndarray,
    correlations: dict[str, float],
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    class_counts = [int((target == 0).sum()), int((target == 1).sum())]
    bars = axes[0].bar(["Normal", "Failure"], class_counts, color=["#64748B", "#DC2626"])
    axes[0].bar_label(bars)
    axes[0].set(title="Severely Imbalanced Alarm Target", ylabel="Operations")
    type_counts = failure_types.sum().rename(index=FAILURE_NAMES)
    bars = axes[1].bar(type_counts.index, type_counts.values, color="#F97316")
    axes[1].bar_label(bars)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(title="Failure Types (Multi-label)", ylabel="Occurrences")
    fig.tight_layout()
    fig.savefig(figure_dir / "01_failure_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(model_results)
    fdr = [100 * model_results[name]["test"]["false_discovery_rate"] for name in names]
    recall = [100 * model_results[name]["test"]["recall"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, fdr, width, label="False discovery rate", color="#DC2626")
    ax.bar(x + width / 2, recall, width, label="Failure recall", color="#2563EB")
    ax.set_xticks(x, names)
    ax.set_ylabel("Percent")
    ax.set_title("Low-False-Alarm Operating Points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "02_fdr_model_comparison.png", dpi=180)
    plt.close(fig)

    precision, recall_curve, _ = precision_recall_curve(
        best_truth, best_probabilities
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_curve, precision, color="#2563EB")
    ax.set(
        title="Precision–Recall Curve for Selected Alarm Model",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "03_precision_recall_curve.png", dpi=180)
    plt.close(fig)

    ordered = sorted(correlations.items(), key=lambda item: abs(item[1]))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        color=["#DC2626" if item[1] < 0 else "#16A34A" for item in ordered],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(
        title="Point-Biserial Correlation with Machine Failure",
        xlabel="Correlation (association, not temporal causality)",
        xlim=(-0.16, 0.23),
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "04_sensor_correlations.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features, target, failure_types, summary = load_and_engineer(args.data)

    all_indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    train_indices, validation_indices = train_test_split(
        train_indices,
        test_size=0.25,
        stratify=target.iloc[train_indices],
        random_state=RANDOM_STATE,
    )
    train_x, validation_x, test_x = (
        features.iloc[train_indices],
        features.iloc[validation_indices],
        features.iloc[test_indices],
    )
    train_y, validation_y, test_y = (
        target.iloc[train_indices],
        target.iloc[validation_indices],
        target.iloc[test_indices],
    )
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum())
    models = make_models(train_x, positive_weight)

    model_results: dict[str, dict] = {}
    fitted_models: dict[str, Pipeline] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        validation_probability = model.predict_proba(validation_x)[:, 1]
        threshold, validation_metrics = tune_threshold(
            validation_y, validation_probability, args.minimum_recall
        )
        test_probability = model.predict_proba(test_x)[:, 1]
        model_results[name] = {
            "validation": validation_metrics,
            "test": binary_metrics(test_y, test_probability, threshold),
        }
        fitted_models[name] = model
        test_probabilities[name] = test_probability

    best_name = min(
        model_results,
        key=lambda name: (
            model_results[name]["validation"]["fdr"],
            -model_results[name]["validation"]["recall"],
            -model_results[name]["validation"]["f1"],
        ),
    )
    best_result = model_results[best_name]["test"]
    best_prediction = (
        test_probabilities[best_name] >= best_result["threshold"]
    ).astype(int)
    report = classification_report(
        test_y,
        best_prediction,
        target_names=["Normal", "Failure"],
        output_dict=True,
        zero_division=0,
    )

    diagnostic_results, diagnostic_models = train_failure_type_models(
        train_x,
        test_x,
        failure_types.iloc[train_indices],
        failure_types.iloc[test_indices],
    )
    correlations = sensor_correlations(features, target)
    strongest_sensor = max(correlations, key=lambda name: abs(correlations[name]))
    summary.update(
        {
            "split_rows": {
                "train": len(train_indices),
                "validation": len(validation_indices),
                "test": len(test_indices),
            },
            "minimum_validation_recall_constraint": args.minimum_recall,
            "positive_class_weight": positive_weight,
            "strongest_absolute_sensor_association": strongest_sensor,
            "strongest_sensor_correlation": correlations[strongest_sensor],
            "time_to_failure_note": (
                "Not estimated: AI4I rows are independent synthetic products, "
                "not longitudinal histories for individual machines. A remaining-"
                "useful-life target cannot be derived without inventing chronology."
            ),
        }
    )

    save_figures(
        output_dir,
        target,
        failure_types,
        model_results,
        test_y,
        test_probabilities[best_name],
        correlations,
    )
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "selected_alarm_model.pkl").open("wb") as destination:
        pickle.dump(
            {
                "model": fitted_models[best_name],
                "threshold": best_result["threshold"],
                "model_name": best_name,
                "features": list(features.columns),
            },
            destination,
        )
    with (model_dir / "failure_type_models.pkl").open("wb") as destination:
        pickle.dump(diagnostic_models, destination)

    artifacts = {
        "dataset_summary.json": summary,
        "model_comparison.json": {
            "selection": (
                "Thresholds chosen on validation data for minimum FDR subject "
                f"to recall >= {args.minimum_recall:.2f}; test set used once."
            ),
            "selected_model": best_name,
            "models": model_results,
        },
        "selected_model_report.json": report,
        "failure_type_metrics.json": diagnostic_results,
        "sensor_correlations.json": correlations,
    }
    for filename, content in artifacts.items():
        (output_dir / filename).write_text(
            json.dumps(content, indent=2) + "\n", encoding="utf-8"
        )
    predictions = pd.DataFrame(
        {
            "row_index": test_indices,
            "actual_failure": test_y.to_numpy(),
            "failure_probability": test_probabilities[best_name],
            "predicted_alarm": best_prediction,
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print(json.dumps(artifacts["model_comparison.json"], indent=2))
    print(f"\nSelected: {best_name}")
    print(f"Artifacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
