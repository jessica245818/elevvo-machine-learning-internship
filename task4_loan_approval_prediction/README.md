# Task 4: Loan Approval Prediction

This project predicts whether a loan application will be approved while
explicitly addressing missing data, categorical encoding, and class imbalance.

## Dataset

The project uses the CC0
[Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/maramsa/loan-database)
from Kaggle. It contains 614 applications and 13 columns, including applicant
demographics, income, loan details, credit history, and the approval target.

The script downloads the workbook automatically when it is missing.

## Data preparation

- Remove the identifier `Loan_ID`.
- Drop duplicate or invalid-target rows.
- Impute numerical features with training-set medians.
- Impute categorical features with training-set modes.
- Ordinally encode categories before imbalance handling.
- One-hot encode categories after optional SMOTE.
- Standardize numerical features.
- Preserve the approval/rejection ratio with stratified train/test splits.

Every preprocessing and resampling step is fitted only on training data through
an imbalanced-learn pipeline, preventing data leakage.

## Models and imbalance strategies

The analysis compares six tuned variants:

1. Logistic Regression
2. Logistic Regression with balanced class weights
3. Logistic Regression with SMOTE
4. Decision Tree
5. Decision Tree with balanced class weights
6. Decision Tree with SMOTE

Hyperparameters are selected using validation macro F1 so performance on the
minority rejected class contributes equally to model selection.

## Evaluation

The held-out test evaluation emphasizes:

- Precision, recall, and F1 for rejected applications
- Precision, recall, and F1 for approved applications
- Macro F1
- Balanced accuracy
- Confusion matrices

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task4_loan_approval_prediction/src/loan_approval_prediction.py
```

## Results

The dataset contains 422 approved and 192 rejected applications, so accuracy
alone would overstate performance. On the stratified held-out test set:

| Model | Accuracy | Balanced accuracy | Macro F1 | Rejected precision | Rejected recall | Rejected F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8636** | **0.7869** | **0.8182** | **0.9655** | 0.5833 | **0.7273** |
| Decision Tree | 0.8052 | 0.7730 | 0.7730 | 0.6875 | 0.6875 | 0.6875 |
| Logistic + class weight | 0.7987 | 0.7797 | 0.7717 | 0.6604 | 0.7292 | 0.6931 |
| Decision Tree + SMOTE | 0.7727 | 0.7665 | 0.7494 | 0.6102 | **0.7500** | 0.6729 |
| Logistic + SMOTE | 0.7727 | 0.7437 | 0.7395 | 0.6275 | 0.6667 | 0.6465 |
| Decision Tree + class weight | 0.6948 | 0.7099 | 0.6782 | 0.5070 | **0.7500** | 0.6050 |

Plain logistic regression is the strongest overall model: it has the highest
macro F1, balanced accuracy, and rejected-class F1. It identifies 28 of 48
rejected applications while misclassifying only one approved application as
rejected. If missing fewer rejected applications is more important than
overall performance, class-weighted logistic regression raises rejected recall
from 0.5833 to 0.7292, with lower precision and macro F1.

SMOTE was applied only to training data and did not improve the best overall
score on this dataset. This is useful evidence that imbalance techniques should
be validated rather than assumed to help.

Generated artifacts include:

- Model comparison metrics and per-model classification reports
- Hyperparameter tuning results and selected parameters
- Cleaning and missing-value summary
- Class distribution and missing-value visualization
- Precision, recall, and macro-F1 comparison
- Confusion matrices for the best logistic and decision-tree models
