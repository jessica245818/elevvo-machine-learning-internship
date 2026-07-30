"""Build and evaluate similarity-based recommenders on MovieLens 100K."""

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
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


RANDOM_STATE = 42
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate collaborative filtering on MovieLens 100K."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("task5_movie_recommendation_system/data/raw/ml-100k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task5_movie_recommendation_system/outputs"),
    )
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--svd-components", type=int, default=50)
    return parser.parse_args()


def ensure_dataset(data_dir: Path) -> None:
    """Download and extract the official GroupLens archive if needed."""
    required = [data_dir / "u1.base", data_dir / "u1.test", data_dir / "u.item"]
    if all(path.exists() for path in required):
        return

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found. Downloading from {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        safe_members = [
            member
            for member in archive.namelist()
            if member.startswith("ml-100k/") and ".." not in Path(member).parts
        ]
        archive.extractall(data_dir.parent, safe_members)


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dataset(data_dir)
    train = pd.read_csv(
        data_dir / "u1.base",
        sep="\t",
        names=RATING_COLUMNS,
        dtype={"user_id": int, "movie_id": int, "rating": float},
    )
    test = pd.read_csv(
        data_dir / "u1.test",
        sep="\t",
        names=RATING_COLUMNS,
        dtype={"user_id": int, "movie_id": int, "rating": float},
    )
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=MOVIE_COLUMNS,
        encoding="latin-1",
        usecols=["movie_id", "title"],
    )
    if train.duplicated(["user_id", "movie_id"]).any():
        train = train.drop_duplicates(["user_id", "movie_id"], keep="last")
    return train, test, movies


def create_user_item_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_ids = np.sort(
        np.union1d(train["user_id"].unique(), test["user_id"].unique())
    )
    movie_ids = np.sort(
        np.union1d(
            movies["movie_id"].unique(),
            np.union1d(train["movie_id"].unique(), test["movie_id"].unique()),
        )
    )
    matrix = (
        train.pivot(index="user_id", columns="movie_id", values="rating")
        .reindex(index=user_ids, columns=movie_ids)
        .to_numpy(dtype=float)
    )
    return matrix, user_ids, movie_ids


def fallback_scores(matrix: np.ndarray) -> np.ndarray:
    """Return Bayesian-shrunk movie means to avoid favoring tiny samples."""
    observed = ~np.isnan(matrix)
    totals = np.nansum(matrix, axis=0)
    counts = observed.sum(axis=0)
    global_mean = float(np.nanmean(matrix))
    prior_ratings = 10
    return (totals + prior_ratings * global_mean) / (counts + prior_ratings)


def user_based_scores(matrix: np.ndarray, neighbors: int) -> np.ndarray:
    """Predict ratings from the most similar users."""
    observed = ~np.isnan(matrix)
    user_means = np.nanmean(matrix, axis=1)
    centered = np.where(observed, matrix - user_means[:, None], 0.0)
    similarities = cosine_similarity(centered)
    overlap_counts = observed.astype(float) @ observed.astype(float).T
    similarities *= overlap_counts / (overlap_counts + 10.0)
    np.fill_diagonal(similarities, 0.0)
    scores = np.empty_like(matrix)
    fallback = fallback_scores(matrix)

    for user_index in range(matrix.shape[0]):
        neighbor_indices = np.argsort(similarities[user_index])[-neighbors:]
        weights = similarities[user_index, neighbor_indices]
        positive = weights > 0
        neighbor_indices = neighbor_indices[positive]
        weights = weights[positive]
        if len(neighbor_indices) == 0:
            scores[user_index] = fallback
            continue
        neighbor_observed = observed[neighbor_indices]
        numerator = weights @ centered[neighbor_indices]
        denominator = np.abs(weights) @ neighbor_observed
        neighbor_prediction = user_means[user_index] + np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        support = neighbor_observed.sum(axis=0)
        confidence = support / (support + 5.0)
        scores[user_index] = (
            confidence * neighbor_prediction + (1 - confidence) * fallback
        )
    return np.clip(scores, 1, 5)


def item_based_scores(matrix: np.ndarray, neighbors: int) -> np.ndarray:
    """Predict ratings from similarity among movies."""
    observed = ~np.isnan(matrix)
    item_means = fallback_scores(matrix)
    centered = np.where(observed, matrix - item_means[None, :], 0.0)
    similarities = cosine_similarity(centered.T)
    np.fill_diagonal(similarities, 0.0)

    keep = min(neighbors, matrix.shape[1] - 1)
    top_indices = np.argpartition(
        similarities,
        kth=matrix.shape[1] - keep,
        axis=1,
    )[:, -keep:]
    mask = np.zeros_like(similarities, dtype=bool)
    mask[np.arange(matrix.shape[1])[:, None], top_indices] = True
    similarities = np.where(mask & (similarities > 0), similarities, 0.0)

    numerator = centered @ similarities.T
    denominator = observed.astype(float) @ np.abs(similarities).T
    adjustments = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.clip(item_means[None, :] + adjustments, 1, 5)


def svd_scores(matrix: np.ndarray, components: int) -> np.ndarray:
    """Apply truncated SVD to the mean-centered user-item matrix."""
    observed = ~np.isnan(matrix)
    user_means = np.nanmean(matrix, axis=1)
    centered = np.where(observed, matrix - user_means[:, None], 0.0)
    component_count = min(components, min(centered.shape) - 1)
    model = TruncatedSVD(
        n_components=component_count,
        random_state=RANDOM_STATE,
    )
    user_factors = model.fit_transform(centered)
    reconstructed = model.inverse_transform(user_factors)
    return np.clip(reconstructed + user_means[:, None], 1, 5)


def popularity_scores(matrix: np.ndarray) -> np.ndarray:
    """Create a non-personalized benchmark from training-set movie means."""
    return np.tile(fallback_scores(matrix), (matrix.shape[0], 1))


def evaluate_precision_at_k(
    scores: np.ndarray,
    matrix: np.ndarray,
    test: pd.DataFrame,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    k: int,
    relevance_threshold: float = 4.0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    user_lookup = {value: index for index, value in enumerate(user_ids)}
    movie_lookup = {value: index for index, value in enumerate(movie_ids)}
    test_relevant = (
        test.loc[test["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .agg(set)
    )
    rows: list[dict[str, float | int]] = []

    for user_id, relevant_movies in test_relevant.items():
        if user_id not in user_lookup:
            continue
        user_index = user_lookup[user_id]
        candidate_scores = scores[user_index].copy()
        candidate_scores[~np.isnan(matrix[user_index])] = -np.inf
        recommendation_indices = np.argsort(candidate_scores)[::-1][:k]
        recommended_movies = set(movie_ids[recommendation_indices])
        hits = len(recommended_movies.intersection(relevant_movies))
        rows.append(
            {
                "user_id": int(user_id),
                "relevant_test_movies": len(relevant_movies),
                "hits": hits,
                f"precision_at_{k}": hits / k,
            }
        )

    per_user = pd.DataFrame(rows)
    metric_name = f"precision_at_{k}"
    summary: dict[str, float | int] = {
        metric_name: float(per_user[metric_name].mean()),
        "evaluated_users": len(per_user),
        "total_hits": int(per_user["hits"].sum()),
    }
    return summary, per_user


def recommend_for_user(
    scores: np.ndarray,
    matrix: np.ndarray,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    movies: pd.DataFrame,
    user_id: int,
    k: int,
) -> pd.DataFrame:
    matches = np.flatnonzero(user_ids == user_id)
    if len(matches) == 0:
        raise ValueError(f"User {user_id} does not exist in the dataset.")
    user_index = int(matches[0])
    candidate_scores = scores[user_index].copy()
    candidate_scores[~np.isnan(matrix[user_index])] = -np.inf
    recommended_indices = np.argsort(candidate_scores)[::-1][:k]
    recommendations = pd.DataFrame(
        {
            "rank": np.arange(1, k + 1),
            "movie_id": movie_ids[recommended_indices],
            "predicted_rating": candidate_scores[recommended_indices],
        }
    )
    return recommendations.merge(movies, on="movie_id", how="left")[
        ["rank", "movie_id", "title", "predicted_rating"]
    ]


def save_visualizations(
    train: pd.DataFrame,
    metrics: pd.DataFrame,
    recommendations: pd.DataFrame,
    figure_dir: Path,
    k: int,
    user_id: int,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    rating_counts = train["rating"].value_counts().sort_index()
    bars = axes[0].bar(
        rating_counts.index.astype(int),
        rating_counts.values,
        color="#2563EB",
    )
    axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(
        title="Training Rating Distribution",
        xlabel="Rating",
        ylabel="Count",
        xticks=rating_counts.index,
    )
    ratings_per_user = train.groupby("user_id").size()
    axes[1].hist(ratings_per_user, bins=25, color="#7C3AED", edgecolor="white")
    axes[1].set(
        title="Ratings per User",
        xlabel="Training ratings",
        ylabel="Users",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "01_dataset_overview.png", dpi=180)
    plt.close(fig)

    ordered = metrics.sort_values(f"precision_at_{k}", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        ordered["model"],
        ordered[f"precision_at_{k}"],
        color=["#94A3B8", "#16A34A", "#F97316", "#2563EB"],
    )
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(
        title=f"Recommendation Performance: Precision@{k}",
        xlabel=f"Mean precision@{k}",
        xlim=(0, max(ordered[f"precision_at_{k}"]) * 1.2),
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "02_precision_at_k.png", dpi=180)
    plt.close(fig)

    shown = recommendations.sort_values("rank", ascending=False)
    labels = shown["title"].str.slice(0, 38)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, shown["predicted_rating"], color="#0EA5E9")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(
        title=f"Top User-Based Recommendations for User {user_id}",
        xlabel="Predicted rating",
        xlim=(0, 5.35),
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "03_sample_recommendations.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, movies = load_data(args.data_dir)
    matrix, user_ids, movie_ids = create_user_item_matrix(train, test, movies)

    score_builders = {
        "Popularity Baseline": lambda: popularity_scores(matrix),
        "User-Based CF": lambda: user_based_scores(matrix, args.neighbors),
        "Item-Based CF": lambda: item_based_scores(matrix, args.neighbors),
        "SVD Matrix Factorization": lambda: svd_scores(
            matrix, args.svd_components
        ),
    }
    metric_rows: list[dict[str, float | int | str]] = []
    per_user_frames: list[pd.DataFrame] = []
    all_scores: dict[str, np.ndarray] = {}

    for model_name, build_scores in score_builders.items():
        print(f"Evaluating {model_name}...")
        scores = build_scores()
        all_scores[model_name] = scores
        summary, per_user = evaluate_precision_at_k(
            scores,
            matrix,
            test,
            user_ids,
            movie_ids,
            args.top_k,
        )
        metric_rows.append({"model": model_name, **summary})
        per_user.insert(0, "model", model_name)
        per_user_frames.append(per_user)

    metrics = pd.DataFrame(metric_rows).sort_values(
        f"precision_at_{args.top_k}",
        ascending=False,
    )
    recommendations = recommend_for_user(
        all_scores["User-Based CF"],
        matrix,
        user_ids,
        movie_ids,
        movies,
        args.user_id,
        args.top_k,
    )

    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.concat(per_user_frames, ignore_index=True).to_csv(
        output_dir / "per_user_precision.csv",
        index=False,
    )
    recommendations.to_csv(
        output_dir / f"user_{args.user_id}_recommendations.csv",
        index=False,
    )
    summary = {
        "dataset": "MovieLens 100K",
        "train_ratings": len(train),
        "test_ratings": len(test),
        "users": len(user_ids),
        "movies": len(movie_ids),
        "matrix_density": float(np.count_nonzero(~np.isnan(matrix)) / matrix.size),
        "rating_scale": [1, 5],
        "relevant_rating_threshold": 4,
        "top_k": args.top_k,
        "neighbors": args.neighbors,
        "svd_components": args.svd_components,
        "sample_user": args.user_id,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    save_visualizations(
        train,
        metrics,
        recommendations,
        output_dir / "figures",
        args.top_k,
        args.user_id,
    )

    print("\nDataset summary")
    print(json.dumps(summary, indent=2))
    print("\nModel metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nTop recommendations for user {args.user_id}")
    print(
        recommendations.to_string(
            index=False,
            formatters={"predicted_rating": lambda value: f"{value:.3f}"},
        )
    )
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
