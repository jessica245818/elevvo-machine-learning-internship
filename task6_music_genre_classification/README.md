# Task 6: Music Genre Classification

This project classifies music into ten genres using acoustic features extracted
from the GTZAN audio collection.

## Dataset

The project uses the `features_30_sec.csv` table from the
[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
on Kaggle. It represents 1,000 thirty-second tracks across ten genres:

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

The classes are balanced with 100 tracks per genre. The script automatically
downloads only the 1.1 MB feature table, rather than the full 1.14 GiB audio
archive.

## Audio features

The 57 predictors were extracted from audio using Librosa-compatible signal
processing and include:

- 20 MFCC means and 20 MFCC variances
- Chroma STFT
- RMS energy
- Spectral centroid and bandwidth
- Spectral roll-off
- Zero-crossing rate
- Harmonic and perceptual components
- Tempo

The filename and constant-length field are excluded from modeling.

## Models

The project compares four leakage-safe, tuned pipelines:

1. Logistic Regression using all audio features
2. RBF Support Vector Machine using all audio features
3. Random Forest using all audio features
4. RBF Support Vector Machine using only MFCC features

The first, second, and fourth variants standardize features inside their
training pipelines. Hyperparameters are selected with five-fold
cross-validation using macro F1. Final accuracy, macro F1, weighted F1, and
per-genre metrics are measured once on a stratified 20% test set.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task6_music_genre_classification/src/music_genre_classification.py
```

## Results

All 1,000 rows are complete and unique. The stratified split contains 800
training tracks and 200 test tracks, with exactly 20 test examples per genre.

| Model | Features | Accuracy | Macro F1 | CV macro F1 |
|---|---:|---:|---:|---:|
| RBF SVM - all features | 57 | **0.7650** | **0.7646** | **0.7360** |
| Logistic Regression - all features | 57 | 0.7100 | 0.7107 | 0.7050 |
| RBF SVM - MFCC only | 40 | 0.7050 | 0.7023 | 0.6927 |
| Random Forest - all features | 57 | 0.7000 | 0.6947 | 0.7062 |

The tuned all-feature RBF SVM (`C=100`, `gamma=scale`) performs best. It
correctly classifies 153 of 200 held-out tracks. Its strongest genre recall is
0.85 for blues, classical, country, and jazz; disco and rock are the most
difficult, with recalls of 0.60 and 0.65.

Using the same RBF approach with only MFCC features reduces macro F1 by 0.0623.
This indicates that chroma, energy, spectral, rhythm, harmonic, and perceptual
features provide useful information beyond MFCCs. Permutation analysis of the
Random Forest also identifies `chroma_stft_mean` and `mfcc9_mean` as its two
most influential test-set features.

## Generated artifacts

- `outputs/model_metrics.csv`
- `outputs/tuning_results.csv`
- `outputs/best_parameters.json`
- `outputs/cleaning_summary.json`
- Per-model classification reports
- `outputs/feature_importance.csv`
- Dataset overview and PCA projection
- Model performance comparison
- Normalized confusion matrices
- Random Forest permutation importance

## Image-based bonus

This repository uses the compact tabular feature release so the project can be
cloned and rerun quickly. A CNN or transfer-learning experiment would require
the separate spectrogram images or full audio archive and substantially larger
dependencies; it is therefore documented as future work rather than presented
as an unverified comparison.
