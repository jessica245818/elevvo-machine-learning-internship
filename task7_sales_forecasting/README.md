# Task 7: Sales Forecasting

This project forecasts the next week's sales for 45 Walmart stores from their
historical weekly sales and economic context.

## Dataset

The project uses the
[Walmart Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
from Kaggle. It contains 6,435 weekly store observations with:

- Store identifier and date
- Weekly sales
- Holiday flag
- Temperature
- Fuel price
- Consumer Price Index (CPI)
- Unemployment rate

The script downloads the CSV automatically when it is missing.

## Time-series features

Features are created independently within each store:

- Year, month, quarter, and ISO week
- Sine/cosine annual seasonality
- Sales lags of 1, 2, 4, 13, and 52 weeks
- Shifted 4-week and 13-week rolling means
- Shifted 4-week and 13-week rolling standard deviations

Every sales-derived feature is shifted by at least one week, so the model never
uses the target week while creating its predictors.

## Forecast design

The project performs a rolling one-week-ahead evaluation:

- The last 20 common weeks are reserved as the untouched test period.
- Earlier observations are used for training.
- Hyperparameters are selected with expanding-window, date-based validation.
- All 45 stores from a week remain in the same fold.

This design is more realistic than a random train/test split and prevents
future observations from leaking into model training.

## Models

1. Seasonal naive baseline using the same store's sales 52 weeks earlier
2. Ridge Regression
3. Random Forest
4. XGBoost

Models are compared using MAE, RMSE, R², MAPE, and WMAPE.

## Bonus analysis

- Shifted rolling-average and rolling-volatility features
- Additive seasonal decomposition with a 52-week period
- XGBoost with time-aware validation
- Permutation importance for the XGBoost forecast

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task7_sales_forecasting/src/sales_forecasting.py
```

## Results

Creating the 52-week lag leaves 4,095 complete modeling rows. The chronological
split uses 3,195 rows from 71 weeks for training and 900 rows from the final 20
weeks (15 June–26 October 2012) for testing.

| Model | MAE | RMSE | R² | MAPE | WMAPE |
|---|---:|---:|---:|---:|---:|
| Random Forest | **$39,965** | **$65,174** | **0.9849** | **3.90%** | **3.86%** |
| XGBoost | $42,176 | $65,350 | 0.9848 | 4.19% | 4.07% |
| Ridge Regression | $52,105 | $71,898 | 0.9816 | 5.65% | 5.03% |
| Seasonal naive | $53,465 | $84,106 | 0.9748 | 5.46% | 5.16% |

The tuned Random Forest (`max_depth=None`, `min_samples_leaf=2`) has the lowest
test error, narrowly outperforming XGBoost. It reduces RMSE by $18,932, or
22.5%, compared with simply using sales from the same week one year earlier.

The aggregate forecast follows the held-out weekly pattern well but smooths the
largest July peak. XGBoost permutation importance identifies the 52-week lag as
the dominant predictor, followed by the shifted 4-week rolling mean, one-week
lag, and shifted 13-week rolling mean. This agrees with the decomposition,
which shows strong recurring holiday-season spikes.

## Generated artifacts

- `outputs/model_metrics.csv`
- `outputs/tuning_results.csv`
- `outputs/best_parameters.json`
- `outputs/cleaning_summary.json`
- `outputs/test_predictions.csv`
- `outputs/feature_importance.csv`
- Sales trend and rolling-average plot
- Seasonal decomposition
- Actual versus predicted test-period plot
- Model error comparison
- XGBoost feature importance
