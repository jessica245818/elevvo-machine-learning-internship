# Task 3: Forest Cover Type Classification

This project predicts one of seven forest cover types from cartographic and
environmental features using tree-based multi-class classification.

## Dataset

The project uses the official
[UCI Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype)
(DOI: `10.24432/C50K5N`). It contains 581,012 rows and 54 predictors:

- 10 quantitative cartographic features
- 4 one-hot wilderness-area indicators
- 40 one-hot soil-type indicators
- 7 target cover types

Scikit-learn downloads and caches the UCI data automatically. The script uses
a reproducible, stratified 120,000-row working sample by default so model
comparison and tuning remain practical on a laptop while preserving every
class.

## Preprocessing

- Check and remove duplicate rows.
- Check missing values.
- Validate that the 44 categorical indicators contain only 0/1 values.
- Validate that every row has one wilderness area and one soil type.
- Convert cover labels from 1-7 to the zero-based representation required by
  XGBoost.
- Use a stratified 80/20 train/test split.

Scaling is unnecessary because both selected models are tree ensembles.

## Models and tuning

The project compares:

1. Random Forest with class-balanced sampling
2. XGBoost with histogram-based tree construction

Each model is tuned over four parameter combinations on a stratified
40,000-row subset. The best configuration is selected using validation macro
F1, then retrained on the full training split and evaluated once on the held-out
test split.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task3_forest_cover_classification/src/forest_cover_classification.py
```

To use the complete UCI dataset instead of the working sample:

```bash
python task3_forest_cover_classification/src/forest_cover_classification.py \
  --sample-size 0
```

## Outputs

- Model comparison metrics
- Hyperparameter tuning results and selected parameters
- Per-class classification reports
- Feature-importance tables
- Normalized confusion matrices
- Class-distribution, feature-importance, and model-comparison figures

## Results

The tuned XGBoost model performs best on the held-out 24,000-row test set:

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.9060 | 0.8581 | 0.8531 | 0.9057 |
| XGBoost | **0.9290** | **0.8701** | **0.8866** | **0.9284** |

Selected hyperparameters:

- Random Forest: 200 trees, unlimited depth, `min_samples_leaf=2`, and square
  root feature sampling.
- XGBoost: 250 trees, depth 10, learning rate 0.15, 90% row subsampling, and
  80% column subsampling.

XGBoost improves overall accuracy by 2.3 percentage points and macro F1 by
3.35 percentage points. It achieves F1 above 0.92 for Spruce/Fir, Lodgepole
Pine, Ponderosa Pine, and Krummholz. Aspen is the hardest class, with 0.7471
F1, and is most often confused with Lodgepole Pine.

Random Forest identifies elevation as the strongest feature, followed by
distance to roadways, fire points, and hydrology. XGBoost relies heavily on
Wilderness Area 3 and several soil-type indicators, demonstrating the value of
properly handling the categorical one-hot features.
