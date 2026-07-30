# Task 2: Customer Segmentation

This project segments mall customers using annual income and spending score. It
implements data cleaning, visual exploration, feature scaling, K-Means cluster
selection, two-dimensional cluster visualization, cluster profiling, and a
DBSCAN bonus comparison.

## Dataset

The project uses the
[Mall Customers Dataset](https://www.kaggle.com/datasets/hosammhmdali/mall-customers-dataset)
from Kaggle. It contains 200 customers with demographic, income, and spending
information and is published under the CC0 public-domain license.

The script automatically downloads the dataset from Kaggle if
`data/raw/Mall_Customers.csv` is not available.

## Method

1. Remove duplicates and invalid income or spending values.
2. Select `Annual Income (k$)` and `Spending Score (1-100)`.
3. Standardize both features with `StandardScaler`.
4. Evaluate K-Means for K=2 through K=10 using inertia and silhouette score.
5. Select the K with the highest silhouette score.
6. Profile each segment using customer count, average age, income, and spending.
7. Tune DBSCAN over multiple `eps` and `min_samples` combinations and compare
   its non-noise silhouette score.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task2_customer_segmentation/src/customer_segmentation.py
```

The generated outputs include:

- `clustered_customers.csv`
- `k_selection_metrics.csv`
- `cluster_profiles.csv`
- `dbscan_parameter_search.csv`
- `model_summary.json`
- `cleaning_summary.json`
- Six figures covering exploration, cluster selection, K-Means, DBSCAN, and
  average spending by segment

## Results

K=5 gives the highest K-Means silhouette score (**0.5547**) and also matches
the strongest bend in the inertia curve. It produces five clear, useful
customer segments:

| Segment | Customers | Avg. age | Avg. income (k$) | Avg. spending |
|---|---:|---:|---:|---:|
| Moderate Income / Moderate Spending | 81 | 42.72 | 55.30 | 49.52 |
| High Income / High Spending | 39 | 32.69 | 86.54 | 82.13 |
| Low Income / High Spending | 22 | 25.27 | 25.73 | 79.36 |
| High Income / Low Spending | 35 | 41.11 | 88.20 | 17.11 |
| Low Income / Low Spending | 23 | 45.22 | 26.30 | 20.91 |

The high-income/high-spending segment has the greatest average spending score
(82.13) and is a strong target for premium retention campaigns. The
low-income/high-spending segment also has high engagement (79.36), suggesting
value-focused offers. High-income/low-spending customers are an opportunity
for activation campaigns.

The best DBSCAN configuration uses `eps=0.35` and `min_samples=8`. Its
non-noise silhouette score is higher at **0.5994**, but it forms only four
clusters and labels 55 customers (27.5%) as noise. K-Means is therefore the
better primary result because it gives every customer a segment and produces
five straightforward business profiles.
