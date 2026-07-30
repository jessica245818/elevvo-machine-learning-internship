# Task 5: Movie Recommendation System

This project builds a collaborative-filtering system that recommends movies
from user preferences in the MovieLens 100K dataset.

## Dataset

The project uses the official
[MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/)
from GroupLens Research. It contains 100,000 ratings from 943 users for 1,682
movies. Ratings range from 1 to 5.

The script downloads the dataset automatically when it is missing and uses the
official `u1.base` and `u1.test` split:

- 80,000 training ratings
- 20,000 held-out test ratings

## Models

1. **Popularity baseline:** recommends unseen movies with the highest average
   training rating.
2. **User-based collaborative filtering:** creates a user-item matrix, centers
   ratings by each user's mean, computes cosine similarity between users, and
   predicts ratings from the 30 nearest positive-similarity neighbors.
   Co-rated-item overlap shrinkage and a Bayesian movie-mean fallback prevent
   tiny samples from producing overconfident recommendations.
3. **Item-based collaborative filtering (bonus):** computes cosine similarity
   among mean-centered movie rating vectors and predicts from similar movies
   already rated by the user.
4. **SVD matrix factorization (bonus):** uses truncated SVD with 50 latent
   components to reconstruct missing preferences.

Movies already rated in training are excluded from every recommendation list.

## Evaluation

A held-out movie is relevant when the user rated it at least 4 stars. For each
eligible user, the model recommends 10 unseen movies:

```text
Precision@10 = relevant recommended movies / 10
```

Mean precision@10 is calculated across all users who have at least one relevant
movie in the official test split. The popularity model provides a
non-personalized benchmark.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task5_movie_recommendation_system/src/movie_recommendation.py
```

Use a different sample user or list size:

```bash
python task5_movie_recommendation_system/src/movie_recommendation.py \
  --user-id 50 --top-k 5
```

## Results

The user-item matrix is sparse: only 5.04% of its cells contain training
ratings. Evaluation covers 456 users with at least one relevant held-out movie.

| Model | Precision@10 | Relevant recommendations |
|---|---:|---:|
| SVD matrix factorization | **0.2171** | **990** |
| User-based collaborative filtering | 0.1654 | 754 |
| Popularity baseline | 0.1404 | 640 |
| Item-based collaborative filtering | 0.0779 | 355 |

The required user-similarity model beats the non-personalized baseline by
0.0250 precision points (17.8% relative improvement). The SVD bonus performs
best, showing that latent factors capture useful preference structure in this
sparse matrix. The item-based implementation underperforms here, illustrating
that the same neighborhood size and similarity design need not work equally
well for users and items.

For sample user 1, the user-based model's highest-rated unseen recommendations
include *The Shawshank Redemption*, *Schindler's List*, *Lawrence of Arabia*,
and *A Close Shave*. The exact ranked list and predicted ratings are saved in
`outputs/user_1_recommendations.csv`.

## Generated artifacts

- `outputs/model_metrics.csv`
- `outputs/per_user_precision.csv`
- `outputs/user_1_recommendations.csv`
- `outputs/dataset_summary.json`
- `outputs/figures/01_dataset_overview.png`
- `outputs/figures/02_precision_at_k.png`
- `outputs/figures/03_sample_recommendations.png`
