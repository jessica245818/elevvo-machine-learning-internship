"""Segment mall customers with K-Means and compare the result with DBSCAN."""

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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"
FEATURES = [INCOME, SPENDING]
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "hosammhmdali/mall-customers-dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster mall customers by annual income and spending score."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "task2_customer_segmentation/data/raw/Mall_Customers.csv"
        ),
        help="Path to Mall_Customers.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task2_customer_segmentation/outputs"),
        help="Directory for Task 2 outputs.",
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    """Download the public CC0 Kaggle dataset when it is missing."""
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
                f"Expected one CSV in the archive, found {len(csv_names)}."
            )
        with archive.open(csv_names[0]) as source, path.open("wb") as destination:
            destination.write(source.read())


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dataset(path)
    data = pd.read_csv(path)
    original_rows = len(data)
    duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    required = {"CustomerID", "Age", INCOME, SPENDING}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    invalid_mask = (
        data[INCOME].isna()
        | data[SPENDING].isna()
        | (data[INCOME] < 0)
        | ~data[SPENDING].between(1, 100)
    )
    invalid_rows = int(invalid_mask.sum())
    data = data.loc[~invalid_mask].reset_index(drop=True)

    summary = {
        "source_file": str(path),
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicates,
        "invalid_rows_removed": invalid_rows,
        "clean_rows": len(data),
        "missing_values_after_cleaning": int(data.isna().sum().sum()),
        "features_used_for_clustering": FEATURES,
    }
    return data, summary


def evaluate_kmeans(
    scaled_features: np.ndarray,
    k_values: range = range(2, 11),
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(scaled_features, labels),
            }
        )

    scores = pd.DataFrame(rows)
    best_k = int(
        scores.loc[scores["silhouette_score"].idxmax(), "k"]
    )
    return scores, best_k


def tune_dbscan(
    scaled_features: np.ndarray,
) -> tuple[DBSCAN, np.ndarray, pd.DataFrame, dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    best: tuple[float, float, int, DBSCAN, np.ndarray] | None = None

    for eps in np.arange(0.15, 1.01, 0.05):
        for min_samples in range(3, 11):
            model = DBSCAN(eps=float(eps), min_samples=min_samples)
            labels = model.fit_predict(scaled_features)
            cluster_labels = set(labels) - {-1}
            cluster_count = len(cluster_labels)
            noise_count = int((labels == -1).sum())
            noise_ratio = noise_count / len(labels)

            score = np.nan
            non_noise_mask = labels != -1
            if (
                2 <= cluster_count <= 10
                and non_noise_mask.sum() > cluster_count
                and noise_ratio <= 0.30
            ):
                score = silhouette_score(
                    scaled_features[non_noise_mask],
                    labels[non_noise_mask],
                )
                candidate = (score, -noise_ratio, -cluster_count, model, labels)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

            rows.append(
                {
                    "eps": float(eps),
                    "min_samples": min_samples,
                    "clusters": cluster_count,
                    "noise_points": noise_count,
                    "noise_ratio": noise_ratio,
                    "silhouette_non_noise": score,
                }
            )

    search_results = pd.DataFrame(rows)
    if best is None:
        raise RuntimeError("DBSCAN search found no valid multi-cluster solution.")

    best_score, neg_noise_ratio, _, best_model, best_labels = best
    details = {
        "eps": float(best_model.eps),
        "min_samples": int(best_model.min_samples),
        "clusters": len(set(best_labels) - {-1}),
        "noise_points": int((best_labels == -1).sum()),
        "noise_ratio": float(-neg_noise_ratio),
        "silhouette_non_noise": float(best_score),
    }
    return best_model, best_labels, search_results, details


def assign_segment_names(
    profiles: pd.DataFrame,
    data: pd.DataFrame,
) -> dict[int, str]:
    income_low, income_high = data[INCOME].quantile([0.33, 0.67])
    spending_low, spending_high = data[SPENDING].quantile([0.33, 0.67])

    def level(value: float, low: float, high: float) -> str:
        if value < low:
            return "Low"
        if value > high:
            return "High"
        return "Moderate"

    names: dict[int, str] = {}
    for cluster_id, row in profiles.iterrows():
        income_level = level(row[INCOME], income_low, income_high)
        spending_level = level(row[SPENDING], spending_low, spending_high)
        names[int(cluster_id)] = (
            f"{income_level} Income / {spending_level} Spending"
        )
    return names


def build_cluster_profiles(data: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        data.groupby("KMeans_Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Average_Age=("Age", "mean"),
            **{
                "Average_Income_k": (INCOME, "mean"),
                "Average_Spending_Score": (SPENDING, "mean"),
            },
        )
        .sort_index()
    )

    naming_frame = profiles.rename(
        columns={
            "Average_Income_k": INCOME,
            "Average_Spending_Score": SPENDING,
        }
    )
    names = assign_segment_names(naming_frame, data)
    profiles.insert(
        0,
        "Segment",
        [names[int(cluster_id)] for cluster_id in profiles.index],
    )
    return profiles.reset_index()


def save_eda_plots(data: pd.DataFrame, figure_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(data[INCOME], bins=18, color="#2563EB", edgecolor="white")
    axes[0].set(
        title="Annual Income Distribution",
        xlabel="Annual income (k$)",
        ylabel="Customers",
    )
    axes[1].hist(data[SPENDING], bins=18, color="#F97316", edgecolor="white")
    axes[1].set(
        title="Spending Score Distribution",
        xlabel="Spending score",
        ylabel="Customers",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "01_feature_distributions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        data[INCOME],
        data[SPENDING],
        alpha=0.75,
        s=38,
        color="#2563EB",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set(
        title="Customer Income and Spending Before Clustering",
        xlabel="Annual income (k$)",
        ylabel="Spending score (1-100)",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "02_income_spending_exploration.png", dpi=180)
    plt.close(fig)


def save_model_plots(
    data: pd.DataFrame,
    k_scores: pd.DataFrame,
    centroids_original: np.ndarray,
    profiles: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(k_scores["k"], k_scores["inertia"], marker="o", color="#2563EB")
    axes[0].set(
        title="Elbow Method",
        xlabel="Number of clusters (K)",
        ylabel="Inertia",
        xticks=k_scores["k"],
    )
    axes[1].plot(
        k_scores["k"],
        k_scores["silhouette_score"],
        marker="o",
        color="#F97316",
    )
    axes[1].set(
        title="Silhouette Analysis",
        xlabel="Number of clusters (K)",
        ylabel="Silhouette score",
        xticks=k_scores["k"],
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "03_k_selection.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        data[INCOME],
        data[SPENDING],
        c=data["KMeans_Cluster"],
        cmap="tab10",
        s=48,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    centroid_points = ax.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        marker="X",
        s=230,
        c="black",
        edgecolor="white",
        linewidth=1.2,
        label="Centroids",
    )
    ax.set(
        title="K-Means Customer Segments",
        xlabel="Annual income (k$)",
        ylabel="Spending score (1-100)",
    )
    cluster_legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="upper left",
    )
    ax.add_artist(cluster_legend)
    ax.legend(
        handles=[centroid_points],
        labels=["Centroids"],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "04_kmeans_clusters.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        data[INCOME],
        data[SPENDING],
        c=data["DBSCAN_Cluster"],
        cmap="tab10",
        s=48,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set(
        title="DBSCAN Customer Segments (-1 = Noise)",
        xlabel="Annual income (k$)",
        ylabel="Spending score (1-100)",
    )
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "05_dbscan_clusters.png", dpi=180)
    plt.close(fig)

    ordered = profiles.sort_values("Average_Spending_Score")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(
        ordered["Segment"],
        ordered["Average_Spending_Score"],
        color="#7C3AED",
    )
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set(
        title="Average Spending Score by K-Means Segment",
        xlabel="Average spending score",
        xlim=(0, 105),
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "06_average_spending_by_segment.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, cleaning_summary = load_and_clean(args.data)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[FEATURES])

    k_scores, best_k = evaluate_kmeans(scaled_features)
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    data["KMeans_Cluster"] = kmeans.fit_predict(scaled_features)
    kmeans_silhouette = silhouette_score(
        scaled_features,
        data["KMeans_Cluster"],
    )
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

    _, dbscan_labels, dbscan_search, dbscan_details = tune_dbscan(
        scaled_features
    )
    data["DBSCAN_Cluster"] = dbscan_labels

    profiles = build_cluster_profiles(data)
    model_summary = {
        "selected_k": best_k,
        "kmeans_silhouette_score": float(kmeans_silhouette),
        "kmeans_inertia": float(kmeans.inertia_),
        "dbscan": dbscan_details,
    }

    save_eda_plots(data, figure_dir)
    save_model_plots(
        data,
        k_scores,
        centroids_original,
        profiles,
        figure_dir,
    )

    data.to_csv(output_dir / "clustered_customers.csv", index=False)
    k_scores.to_csv(output_dir / "k_selection_metrics.csv", index=False)
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    dbscan_search.to_csv(output_dir / "dbscan_parameter_search.csv", index=False)
    with (output_dir / "cleaning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(cleaning_summary, file, indent=2)
    with (output_dir / "model_summary.json").open("w", encoding="utf-8") as file:
        json.dump(model_summary, file, indent=2)

    print("\nCleaning summary")
    print(json.dumps(cleaning_summary, indent=2))
    print("\nK selection")
    print(k_scores.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSelected K-Means model")
    print(json.dumps(model_summary, indent=2))
    print("\nCluster profiles")
    print(profiles.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
