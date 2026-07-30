"""Classify GTZAN music genres from extracted audio features."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
TARGET = "label"
DATASET_API = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "andradaolteanu/gtzan-dataset-music-genre-classification"
)
DATASET_FILE = "Data/features_30_sec.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiclass genre classifiers on GTZAN features."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "task6_music_genre_classification/data/raw/features_30_sec.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task6_music_genre_classification/outputs"),
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    """Download only the pre-extracted 30-second feature table from Kaggle."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DATASET_API}?{urllib.parse.urlencode({'filename': DATASET_FILE})}"
    print(f"Dataset not found. Downloading {DATASET_FILE} from Kaggle...")
    with urllib.request.urlopen(url) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith("features_30_sec.csv")
        ]
        if len(csv_names) != 1:
            raise RuntimeError(
                f"Expected one feature CSV in archive, found {len(csv_names)}."
            )
        with archive.open(csv_names[0]) as source, path.open("wb") as destination:
            destination.write(source.read())


def load_and_clean(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    ensure_dataset(path)
    data = pd.read_csv(path)
    original_rows = len(data)
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    if TARGET not in data or "filename" not in data:
        raise ValueError("Dataset must include filename and label columns.")
    missing_values = int(data.isna().sum().sum())
    invalid_numeric = int(
        np.isinf(
            data.drop(columns=["filename", TARGET]).to_numpy(dtype=float)
        ).sum()
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    filenames = data.pop("filename")
    target = data.pop(TARGET).astype(str)
    if "length" in data:
        data = data.drop(columns=["length"])

    non_numeric = data.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(f"Unexpected non-numeric features: {non_numeric}")

    class_counts = target.value_counts().sort_index()
    summary = {
        "source_file": str(path),
        "original_rows": original_rows,
        "clean_rows": len(data),
        "duplicate_rows_removed": duplicate_rows,
        "rows_removed_for_invalid_values": original_rows
        - duplicate_rows
        - len(data),
        "missing_values_found": missing_values,
        "infinite_values_found": invalid_numeric,
        "features": data.shape[1],
        "genres": len(class_counts),
        "class_counts": class_counts.to_dict(),
        "filenames_unique": int(filenames.nunique()),
    }
    return data, target, summary


def model_definitions() -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=5_000,
                random_state=RANDOM_STATE,
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "RBF SVM": (
            SVC(random_state=RANDOM_STATE),
            {
                "classifier__C": [1.0, 10.0, 100.0],
                "classifier__gamma": ["scale", 0.01],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=400,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            {
                "classifier__max_depth": [None, 15],
                "classifier__min_samples_leaf": [1, 2],
            },
        ),
    }


def train_and_evaluate(
    variant: str,
    estimator: object,
    parameter_grid: dict[str, list[object]],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    labels: list[str],
) -> tuple[
    dict[str, object],
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    GridSearchCV,
]:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )
    start = time.perf_counter()
    search.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    predictions = search.predict(x_test)

    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    metrics = {
        "variant": variant,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "cv_macro_f1": search.best_score_,
        "fit_seconds": fit_seconds,
        "best_parameters": json.dumps(
            {
                key.replace("classifier__", ""): value
                for key, value in search.best_params_.items()
            },
            sort_keys=True,
        ),
    }
    tuning = pd.DataFrame(search.cv_results_)[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "rank_test_score",
        ]
    ].copy()
    tuning.insert(0, "variant", variant)
    tuning["params"] = tuning["params"].map(
        lambda params: json.dumps(
            {
                key.replace("classifier__", ""): value
                for key, value in params.items()
            },
            sort_keys=True,
        )
    )
    return metrics, predictions, report, tuning, search


def save_dataset_overview(
    features: pd.DataFrame,
    target: pd.Series,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = target.value_counts().sort_index()
    bars = axes[0].bar(counts.index, counts.values, color="#2563EB")
    axes[0].bar_label(bars, padding=3)
    axes[0].set(
        title="GTZAN Genre Distribution",
        xlabel="Genre",
        ylabel="Tracks",
        ylim=(0, 115),
    )
    axes[0].tick_params(axis="x", rotation=45)

    scaled = StandardScaler().fit_transform(features)
    components = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        scaled
    )
    for genre in counts.index:
        mask = target.to_numpy() == genre
        axes[1].scatter(
            components[mask, 0],
            components[mask, 1],
            s=15,
            alpha=0.65,
            label=genre,
        )
    axes[1].set(
        title="PCA Projection of Audio Features",
        xlabel="Principal component 1",
        ylabel="Principal component 2",
    )
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_dataset_overview.png", dpi=180)
    plt.close(fig)


def save_model_comparison(metrics: pd.DataFrame, figure_dir: Path) -> None:
    ordered = metrics.sort_values("macro_f1")
    positions = np.arange(len(ordered))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    accuracy_bars = ax.barh(
        positions - width / 2,
        ordered["accuracy"],
        width,
        label="Accuracy",
        color="#2563EB",
    )
    f1_bars = ax.barh(
        positions + width / 2,
        ordered["macro_f1"],
        width,
        label="Macro F1",
        color="#F97316",
    )
    ax.bar_label(accuracy_bars, fmt="%.3f", padding=3)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3)
    ax.set(
        title="GTZAN Multiclass Model Comparison",
        xlabel="Held-out test score",
        xlim=(0, 1.05),
        yticks=positions,
    )
    ax.set_yticklabels(ordered["variant"])
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_model_comparison.png", dpi=180)
    plt.close(fig)


def save_confusion_matrices(
    y_test: np.ndarray,
    selected_predictions: dict[str, np.ndarray],
    labels: list[str],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, (variant, predictions) in zip(
        axes,
        selected_predictions.items(),
    ):
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(len(labels)),
            normalize="true",
        )
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix[row, column]
                if value >= 0.10:
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > 0.55 else "black",
                    )
        ax.set(
            title=variant,
            xlabel="Predicted genre",
            ylabel="Actual genre",
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
        )
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    fig.suptitle("Normalized Confusion Matrices for the Top Two Models")
    fig.tight_layout()
    fig.savefig(figure_dir / "03_confusion_matrices.png", dpi=180)
    plt.close(fig)


def save_feature_importance(
    importance: pd.DataFrame,
    figure_dir: Path,
) -> None:
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["feature"], top["importance"], color="#7C3AED")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(
        title="Top Random Forest Features (Permutation Importance)",
        xlabel="Mean decrease in test accuracy",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "04_feature_importance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    features, target, cleaning_summary = load_and_clean(args.data)
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(target)
    labels = encoder.classes_.tolist()
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        encoded_target,
        test_size=0.20,
        stratify=encoded_target,
        random_state=RANDOM_STATE,
    )

    feature_sets = {
        "All Audio Features": features.columns.tolist(),
        "MFCC Features Only": [
            column for column in features if column.startswith("mfcc")
        ],
    }
    variants = [
        ("Logistic Regression - All Features", "Logistic Regression", "All Audio Features"),
        ("RBF SVM - All Features", "RBF SVM", "All Audio Features"),
        ("Random Forest - All Features", "Random Forest", "All Audio Features"),
        ("RBF SVM - MFCC Only", "RBF SVM", "MFCC Features Only"),
    ]
    definitions = model_definitions()
    metric_rows: list[dict[str, object]] = []
    tuning_frames: list[pd.DataFrame] = []
    predictions_by_variant: dict[str, np.ndarray] = {}
    searches: dict[str, GridSearchCV] = {}

    for variant, model_name, feature_set in variants:
        print(f"Tuning {variant}...")
        columns = feature_sets[feature_set]
        estimator, grid = definitions[model_name]
        metrics, predictions, report, tuning, search = train_and_evaluate(
            variant,
            estimator,
            grid,
            x_train[columns],
            y_train,
            x_test[columns],
            y_test,
            labels,
        )
        metrics["feature_set"] = feature_set
        metrics["feature_count"] = len(columns)
        metric_rows.append(metrics)
        tuning_frames.append(tuning)
        predictions_by_variant[variant] = predictions
        searches[variant] = search
        safe_name = variant.lower().replace(" ", "_").replace("-", "")
        report.to_csv(output_dir / f"{safe_name}_classification_report.csv")

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.concat(tuning_frames, ignore_index=True).to_csv(
        output_dir / "tuning_results.csv",
        index=False,
    )
    cleaning_summary.update(
        {
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "feature_sets": {
                name: len(columns) for name, columns in feature_sets.items()
            },
        }
    )
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(cleaning_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    best_parameters = {
        variant: {
            key.replace("classifier__", ""): value
            for key, value in search.best_params_.items()
        }
        for variant, search in searches.items()
    }
    (output_dir / "best_parameters.json").write_text(
        json.dumps(best_parameters, indent=2) + "\n",
        encoding="utf-8",
    )

    forest_variant = "Random Forest - All Features"
    forest_search = searches[forest_variant]
    forest_importance = permutation_importance(
        forest_search.best_estimator_,
        x_test,
        y_test,
        scoring="accuracy",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": forest_importance.importances_mean,
            "importance_std": forest_importance.importances_std,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    save_dataset_overview(features, target, figure_dir)
    save_model_comparison(metrics, figure_dir)
    top_variants = metrics.head(2)["variant"].tolist()
    save_confusion_matrices(
        y_test,
        {
            variant: predictions_by_variant[variant]
            for variant in top_variants
        },
        labels,
        figure_dir,
    )
    save_feature_importance(importance, figure_dir)

    display_columns = [
        "variant",
        "feature_count",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "cv_macro_f1",
    ]
    print("\nCleaning summary")
    print(json.dumps(cleaning_summary, indent=2))
    print("\nTest metrics")
    print(
        metrics[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
