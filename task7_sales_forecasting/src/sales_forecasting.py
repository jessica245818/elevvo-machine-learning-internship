"""Forecast Walmart store sales with leakage-safe time-series features."""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET = "Weekly_Sales"
DATASET_API = (
    "https://www.kaggle.com/api/v1/datasets/download/yasserh/walmart-dataset"
)
DATASET_FILE = "Walmart.csv"
TEST_WEEKS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast weekly Walmart sales with time-aware validation."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("task7_sales_forecasting/data/raw/Walmart.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task7_sales_forecasting/outputs"),
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    """Download the public Kaggle Walmart CSV when it is missing."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DATASET_API}?{urllib.parse.urlencode({'filename': DATASET_FILE})}"
    print(f"Dataset not found. Downloading {DATASET_FILE} from Kaggle...")
    with urllib.request.urlopen(url) as response:
        content = response.read()

    if content.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith("walmart.csv")
            ]
            if len(csv_names) != 1:
                raise RuntimeError(
                    f"Expected one Walmart CSV, found {len(csv_names)}."
                )
            with archive.open(csv_names[0]) as source, path.open("wb") as target:
                target.write(source.read())
    else:
        path.write_bytes(content)


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dataset(path)
    data = pd.read_csv(path)
    required = {
        "Store",
        "Date",
        TARGET,
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
    }
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    original_rows = len(data)
    duplicates = int(data.duplicated(["Store", "Date"]).sum())
    missing_values = int(data.isna().sum().sum())
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
    data = (
        data.drop_duplicates(["Store", "Date"], keep="last")
        .dropna()
        .sort_values(["Date", "Store"])
        .reset_index(drop=True)
    )
    if (data[TARGET] < 0).any():
        raise ValueError("Weekly sales must be non-negative.")

    summary = {
        "source_file": str(path),
        "original_rows": original_rows,
        "clean_rows": len(data),
        "duplicates_removed": duplicates,
        "missing_values_found": missing_values,
        "stores": int(data["Store"].nunique()),
        "weeks": int(data["Date"].nunique()),
        "start_date": data["Date"].min().date().isoformat(),
        "end_date": data["Date"].max().date().isoformat(),
        "sales_min": float(data[TARGET].min()),
        "sales_max": float(data[TARGET].max()),
        "sales_mean": float(data[TARGET].mean()),
    }
    return data, summary


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and strictly past-looking store sales features."""
    featured = data.copy()
    iso = featured["Date"].dt.isocalendar()
    featured["Year"] = featured["Date"].dt.year
    featured["Month"] = featured["Date"].dt.month
    featured["Quarter"] = featured["Date"].dt.quarter
    featured["WeekOfYear"] = iso.week.astype(int)
    featured["WeekSin"] = np.sin(2 * np.pi * featured["WeekOfYear"] / 52.0)
    featured["WeekCos"] = np.cos(2 * np.pi * featured["WeekOfYear"] / 52.0)

    grouped_sales = featured.groupby("Store", sort=False)[TARGET]
    for lag in [1, 2, 4, 13, 52]:
        featured[f"SalesLag{lag}"] = grouped_sales.shift(lag)
    for window in [4, 13]:
        featured[f"RollingMean{window}"] = grouped_sales.transform(
            lambda values: values.shift(1).rolling(window).mean()
        )
        featured[f"RollingStd{window}"] = grouped_sales.transform(
            lambda values: values.shift(1).rolling(window).std()
        )

    return featured.dropna().sort_values(["Date", "Store"]).reset_index(drop=True)


def make_date_splits(
    dates: pd.Series,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window CV folds whose validation dates are later."""
    unique_dates = np.sort(dates.unique())
    splitter = TimeSeriesSplit(n_splits=n_splits)
    date_values = dates.to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_date_indices, validate_date_indices in splitter.split(unique_dates):
        train_dates = unique_dates[train_date_indices]
        validate_dates = unique_dates[validate_date_indices]
        train_indices = np.flatnonzero(np.isin(date_values, train_dates))
        validate_indices = np.flatnonzero(np.isin(date_values, validate_dates))
        splits.append((train_indices, validate_indices))
    return splits


def make_pipeline(model: object, scale: bool) -> Pipeline:
    numeric_transformer: object = StandardScaler() if scale else "passthrough"
    preprocessor = ColumnTransformer(
        [
            (
                "store",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Store"],
            ),
            ("numeric", numeric_transformer, lambda frame: [
                column for column in frame.columns if column != "Store"
            ]),
        ],
        sparse_threshold=0,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def model_definitions() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "Ridge Regression": (
            make_pipeline(Ridge(solver="lsqr"), scale=True),
            {"model__alpha": [1.0, 10.0, 100.0]},
        ),
        "Random Forest": (
            make_pipeline(
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                scale=False,
            ),
            {
                "model__max_depth": [12, None],
                "model__min_samples_leaf": [2],
            },
        ),
        "XGBoost": (
            make_pipeline(
                XGBRegressor(
                    objective="reg:squarederror",
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                scale=False,
            ),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [3, 6],
            },
        ),
    }


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    absolute_error = np.abs(actual - predicted)
    denominator = np.maximum(np.abs(actual), 1.0)
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
        "mape_percent": float(np.mean(absolute_error / denominator) * 100),
        "wmape_percent": float(absolute_error.sum() / np.abs(actual).sum() * 100),
    }


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, GridSearchCV],
]:
    cv_splits = make_date_splits(train_dates)
    metric_rows: list[dict[str, object]] = []
    tuning_frames: list[pd.DataFrame] = []
    predictions: dict[str, np.ndarray] = {}
    searches: dict[str, GridSearchCV] = {}

    baseline = x_test["SalesLag52"].to_numpy()
    metric_rows.append(
        {
            "model": "Seasonal Naive (52-week lag)",
            **regression_metrics(y_test.to_numpy(), baseline),
            "cv_rmse": np.nan,
            "fit_seconds": 0.0,
            "best_parameters": "{}",
        }
    )
    predictions["Seasonal Naive (52-week lag)"] = baseline

    for model_name, (pipeline, parameter_grid) in model_definitions().items():
        print(f"Tuning {model_name}...")
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            n_jobs=1,
            return_train_score=True,
        )
        start = time.perf_counter()
        search.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        predicted = np.maximum(search.predict(x_test), 0)
        predictions[model_name] = predicted
        searches[model_name] = search
        metric_rows.append(
            {
                "model": model_name,
                **regression_metrics(y_test.to_numpy(), predicted),
                "cv_rmse": -search.best_score_,
                "fit_seconds": fit_seconds,
                "best_parameters": json.dumps(
                    {
                        key.replace("model__", ""): value
                        for key, value in search.best_params_.items()
                    },
                    sort_keys=True,
                ),
            }
        )
        tuning = pd.DataFrame(search.cv_results_)[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "mean_train_score",
                "rank_test_score",
            ]
        ].copy()
        tuning.insert(0, "model", model_name)
        tuning["cv_rmse"] = -tuning.pop("mean_test_score")
        tuning["params"] = tuning["params"].map(
            lambda params: json.dumps(
                {
                    key.replace("model__", ""): value
                    for key, value in params.items()
                },
                sort_keys=True,
            )
        )
        tuning_frames.append(tuning)

    metrics = pd.DataFrame(metric_rows).sort_values("rmse")
    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    return metrics, tuning_results, predictions, searches


def save_sales_overview(data: pd.DataFrame, figure_dir: Path) -> None:
    aggregate = data.groupby("Date")[TARGET].sum().sort_index()
    rolling = aggregate.rolling(13, center=True).mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(aggregate.index, aggregate.values / 1e6, label="Weekly sales", alpha=0.65)
    ax.plot(
        rolling.index,
        rolling.values / 1e6,
        label="13-week centered average",
        linewidth=2.5,
        color="#DC2626",
    )
    ax.set(
        title="Total Walmart Weekly Sales and Rolling Trend",
        xlabel="Date",
        ylabel="Sales (millions)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "01_sales_trend.png", dpi=180)
    plt.close(fig)


def save_seasonal_decomposition(data: pd.DataFrame, figure_dir: Path) -> None:
    aggregate = data.groupby("Date")[TARGET].sum().sort_index()
    decomposition = seasonal_decompose(
        aggregate,
        model="additive",
        period=52,
        extrapolate_trend="freq",
    )
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(aggregate.index, aggregate.values / 1e6, color="#2563EB")
    axes[0].set_ylabel("Observed\n(millions)")
    axes[1].plot(
        decomposition.trend.index,
        decomposition.trend.values / 1e6,
        color="#16A34A",
    )
    axes[1].set_ylabel("Trend\n(millions)")
    axes[2].plot(
        decomposition.seasonal.index,
        decomposition.seasonal.values / 1e6,
        color="#F97316",
    )
    axes[2].set_ylabel("Seasonal\n(millions)")
    axes[2].set_xlabel("Date")
    fig.suptitle("Additive Seasonal Decomposition (52-Week Period)")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_seasonal_decomposition.png", dpi=180)
    plt.close(fig)


def save_forecast_plot(
    test_frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    best_model: str,
    figure_dir: Path,
) -> None:
    comparison = test_frame[["Date", TARGET]].copy()
    comparison["Predicted"] = predictions[best_model]
    aggregate = comparison.groupby("Date").sum()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        aggregate.index,
        aggregate[TARGET] / 1e6,
        marker="o",
        label="Actual",
        color="#111827",
    )
    ax.plot(
        aggregate.index,
        aggregate["Predicted"] / 1e6,
        marker="o",
        label=f"Predicted - {best_model}",
        color="#2563EB",
    )
    ax.set(
        title="Actual vs. Predicted Total Weekly Sales",
        xlabel="Test week",
        ylabel="Sales (millions)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "03_actual_vs_predicted.png", dpi=180)
    plt.close(fig)


def save_model_comparison(metrics: pd.DataFrame, figure_dir: Path) -> None:
    ordered = metrics.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(
        ordered["model"],
        ordered["rmse"] / 1_000,
        color=["#94A3B8", "#F97316", "#16A34A", "#2563EB"],
    )
    ax.bar_label(bars, fmt="$%.1fk", padding=4)
    ax.set(
        title="One-Week-Ahead Forecast Error",
        xlabel="Test RMSE (thousands of dollars; lower is better)",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "04_model_comparison.png", dpi=180)
    plt.close(fig)


def save_feature_importance(
    importance: pd.DataFrame,
    figure_dir: Path,
) -> None:
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["feature"], top["importance"] / 1_000, color="#7C3AED")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(
        title="XGBoost Permutation Importance",
        xlabel="Increase in RMSE when shuffled (thousands of dollars)",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "05_feature_importance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw_data, cleaning_summary = load_and_clean(args.data)
    featured = add_time_features(raw_data)
    unique_dates = np.sort(featured["Date"].unique())
    test_dates = unique_dates[-TEST_WEEKS:]
    train_mask = ~featured["Date"].isin(test_dates)
    train_frame = featured.loc[train_mask].reset_index(drop=True)
    test_frame = featured.loc[~train_mask].reset_index(drop=True)

    excluded = [TARGET, "Date"]
    feature_columns = [
        column for column in featured.columns if column not in excluded
    ]
    x_train = train_frame[feature_columns]
    y_train = train_frame[TARGET]
    x_test = test_frame[feature_columns]
    y_test = test_frame[TARGET]

    metrics, tuning, predictions, searches = train_models(
        x_train,
        y_train,
        train_frame["Date"],
        x_test,
        y_test,
    )
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    tuning.to_csv(output_dir / "tuning_results.csv", index=False)

    best_model = metrics.loc[
        metrics["model"] != "Seasonal Naive (52-week lag)", "model"
    ].iloc[0]
    prediction_output = test_frame[["Store", "Date", TARGET]].copy()
    for model_name, values in predictions.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
        )
        prediction_output[f"predicted_{safe_name}"] = values
    prediction_output.to_csv(output_dir / "test_predictions.csv", index=False)

    best_parameters = {
        name: {
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        }
        for name, search in searches.items()
    }
    (output_dir / "best_parameters.json").write_text(
        json.dumps(best_parameters, indent=2) + "\n",
        encoding="utf-8",
    )
    cleaning_summary.update(
        {
            "featured_rows": len(featured),
            "features": len(feature_columns),
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
            "train_weeks": int(train_frame["Date"].nunique()),
            "test_weeks": int(test_frame["Date"].nunique()),
            "forecast_type": "rolling one-week-ahead",
            "first_test_date": test_frame["Date"].min().date().isoformat(),
            "last_test_date": test_frame["Date"].max().date().isoformat(),
        }
    )
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(cleaning_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    xgboost_search = searches["XGBoost"]
    importance_result = permutation_importance(
        xgboost_search.best_estimator_,
        x_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    save_sales_overview(raw_data, figure_dir)
    save_seasonal_decomposition(raw_data, figure_dir)
    save_forecast_plot(test_frame, predictions, best_model, figure_dir)
    save_model_comparison(metrics, figure_dir)
    save_feature_importance(importance, figure_dir)

    display_columns = [
        "model",
        "mae",
        "rmse",
        "r2",
        "mape_percent",
        "wmape_percent",
        "cv_rmse",
    ]
    print("\nData summary")
    print(json.dumps(cleaning_summary, indent=2))
    print("\nForecast metrics")
    print(
        metrics[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:,.2f}",
        )
    )
    print(f"\nBest trained model: {best_model}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
