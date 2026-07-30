# Elevvo Machine Learning Internship

This repository contains completed projects for the Elevvo Machine Learning
Internship.

## Projects

1. **Task 1 - Student Score Prediction:** Regression analysis using study hours
   and other academic factors. The best all-feature linear model achieved
   R² 0.8250.
2. **[Task 2 - Customer Segmentation](task2_customer_segmentation/):**
   K-Means clustering of mall customers using annual income and spending
   score, with DBSCAN as a bonus comparison.
3. **[Task 3 - Forest Cover Type Classification](task3_forest_cover_classification/):**
   Tuned Random Forest and XGBoost models for seven-class forest cover
   prediction using the official UCI Covertype dataset.
4. **[Task 4 - Loan Approval Prediction](task4_loan_approval_prediction/):**
   Leakage-safe binary classification with Logistic Regression, Decision
   Trees, class weighting, and SMOTE for imbalanced loan decisions.
5. **[Task 5 - Movie Recommendation System](task5_movie_recommendation_system/):**
   User-based collaborative filtering with item-based and SVD bonus models,
   evaluated using precision at K on held-out ratings.
6. **[Task 6 - Music Genre Classification](task6_music_genre_classification/):**
   Ten-class GTZAN genre prediction from MFCC and spectral audio features,
   comparing Logistic Regression, SVM, and Random Forest models.
7. **[Task 7 - Sales Forecasting](task7_sales_forecasting/):**
   Time-aware Walmart weekly sales forecasting with lag and rolling features,
   chronological validation, seasonal analysis, and XGBoost.
8. **[Task 8 - Traffic Sign Recognition](task8_traffic_sign_recognition/):**
   Full-scene GTSDB object detection with YOLOv8, mAP/FPS evaluation,
   confidence filtering, ONNX export, and webcam inference.
9. **[Task 9 - Industrial Predictive Maintenance](task9_predictive_maintenance/):**
   Cost-sensitive machine-failure alarms optimized for low false-discovery
   rate, multi-label failure diagnosis, and sensor indicator analysis.
10. **[Task 10 - End-to-End MLOps Pipeline](task10_mlops_pipeline/):**
    FastAPI model serving with strict Pydantic validation, Docker packaging,
    Streamlit demo, automated tests, and GitHub Actions CI.

---

# Task 1: Student Score Prediction

Task 1 of the Elevvo Machine Learning Internship.

This project predicts students' final exam scores from study habits and other
academic factors. It covers data cleaning, exploratory visualization,
train/test splitting, linear regression, evaluation metrics, polynomial
regression, and feature-combination experiments.

## Dataset

The project uses the
[Student Performance Factors dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
from Kaggle. It contains 6,607 student records and 20 columns. The dataset is
published under the CC0 public-domain license.

The CSV is stored at:

```text
data/raw/StudentPerformanceFactors.csv
```

If that file is missing, the analysis script downloads the public dataset
archive automatically from Kaggle before training.

## Models compared

1. Linear regression using only `Hours_Studied`
2. Polynomial regression (degree 2) using only `Hours_Studied`
3. Linear regression using selected numeric features:
   `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`,
   `Tutoring_Sessions`, and `Physical_Activity`
4. Linear regression using all numeric and categorical features

The full-feature pipeline imputes missing values and one-hot encodes categorical
variables without leaking test-set information into training.

## Evaluation

All models use the same reproducible 80/20 train/test split
(`random_state=42`). Performance is compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Study hours - linear | 2.4190 | 3.1559 | 0.2469 |
| Study hours - polynomial (degree 2) | 2.4151 | 3.1538 | 0.2479 |
| Selected numeric features - linear | 1.2442 | 2.0432 | 0.6843 |
| All features - linear | **0.4160** | **1.5213** | **0.8250** |

Study hours have a clear positive relationship with exam score, but they explain
only about 25% of the variation by themselves. Adding a quadratic study-hours
term produces almost no improvement, so the relationship is adequately
represented by a straight line within this dataset. Attendance, previous
scores, sleep, tutoring, and other contextual factors contain substantial
additional predictive information. The all-feature model performs best,
explaining 82.5% of test-set variation and missing the true score by about
0.42 points on average.

The raw data contains no duplicate rows. The cleaning step removes one exam
score above the valid 0-100 range. Missing categorical values in
`Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home` are
imputed using the most frequent training-set category.

Run the project to regenerate the exact results:

```bash
python -m pip install -r requirements.txt
python src/student_score_prediction.py
```

Generated files are written to `outputs/`:

- `model_metrics.csv`
- `baseline_test_predictions.csv`
- `cleaning_summary.json`
- `figures/01_distributions.png`
- `figures/02_correlation_matrix.png`
- `figures/03_study_hours_regression.png`
- `figures/04_baseline_diagnostics.png`
- `figures/05_model_comparison.png`

## Project structure

```text
.
├── data/
│   └── raw/
│       └── StudentPerformanceFactors.csv
├── outputs/
│   └── figures/
├── src/
│   └── student_score_prediction.py
├── README.md
└── requirements.txt
```
