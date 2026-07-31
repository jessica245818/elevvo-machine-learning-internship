# Task 1: Student Score Prediction

This project predicts students' final exam scores from study habits and other
academic factors. It covers data cleaning, exploratory visualization,
train/test splitting, linear regression, evaluation metrics, polynomial
regression, and feature-combination experiments.

## Dataset

The project uses the
[Student Performance Factors dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
from Kaggle. It contains 6,607 student records and 20 columns and is published
under the CC0 public-domain license.

The script downloads the public dataset automatically when
`data/raw/StudentPerformanceFactors.csv` is missing.

## Models compared

1. Linear regression using only `Hours_Studied`
2. Polynomial regression (degree 2) using only `Hours_Studied`
3. Linear regression using selected numeric features
4. Linear regression using all numeric and categorical features

The full-feature pipeline imputes missing values and one-hot encodes categorical
variables without leaking test-set information into training.

## Evaluation

All models use the same reproducible 80/20 train/test split
(`random_state=42`). Performance is compared using MAE, RMSE, and R².

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Study hours - linear | 2.4190 | 3.1559 | 0.2469 |
| Study hours - polynomial (degree 2) | 2.4151 | 3.1538 | 0.2479 |
| Selected numeric features - linear | 1.2442 | 2.0432 | 0.6843 |
| All features - linear | **0.4160** | **1.5213** | **0.8250** |

Study hours explain about 25% of score variation by themselves. The quadratic
term adds almost no improvement. Attendance, previous scores, sleep, tutoring,
and other contextual factors substantially improve performance. The
all-feature model explains 82.5% of test-set variation.

The cleaning step removes one exam score outside the valid 0–100 range.
Missing categorical values are imputed using the most frequent training-set
category.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task1_student_score_prediction/src/student_score_prediction.py
```

Generated metrics and figures are written to
`task1_student_score_prediction/outputs/`.

## Project structure

```text
task1_student_score_prediction/
├── data/
│   └── raw/
├── outputs/
│   └── figures/
├── src/
│   └── student_score_prediction.py
└── README.md
```
