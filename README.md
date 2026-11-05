# City Hotel (H2) Booking — ML Project

CSE382 Introduction to Machine Learning — Fall 2026

Machine learning system for City Hotel (H2) bookings with three tasks:

- **Regression:** predict `ADR` (average daily rate)
- **Classification:** predict `IsCanceled`
- **Clustering:** segment bookings without using labels

**Prediction point:** all supervised predictions are made at the moment the
booking is created. See [`reports/leakage_audit.md`](reports/leakage_audit.md).

## Team

| Name            | GitHub account                                                     | Milestone 1 contribution                                                       |
| --------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Mohamed Hussein | [@mohamed-hussein21](https://github.com/mohamed-hussein21)         | Repository structure, CLI (`main.py`), config, README, code review and merging |
| Mark Ezzat      | [@Markezzat](https://github.com/Markezzat)                         | Preprocessing pipeline, cleaning rules, stratified split, feature selection    |
| Reem            | [@reemelgahawy088-cyber](https://github.com/reemelgahawy088-cyber) | Data loading, data-cleaning audit, EDA and figures, K-Means baseline (K = 6)   |
| Youssef         | [@mdlvkopwefvmwkop](https://github.com/mdlvkopwefvmwkop)           | XGBoost regression and classification baselines, feature-set experiments       |
| Fatma Osama     | [@fatma-osama11](https://github.com/fatma-osama11)                 | Shared clustering features and evaluation, leakage audit                       |
| Omar Elhosiny   | [@OmarElhosiny](https://github.com/OmarElhosiny)                   | <contribution>                                                                 |

## Setup

Tested with Python 3.14 on Windows (PowerShell).

1. Clone the repository and enter the folder:
   ```
   git clone https://github.com/OmarElhosiny/ML-Project.git
   cd ML-Project
   ```
2. Place the dataset at `data/raw/rawdata.csv` (it is not committed to the repository).
3. Create a virtual environment and install the requirements:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

All experiments use `random_state = 42` (set in `config.yaml`).

## How to run (Milestone 1)

| Step                                                     | Command                                        | Output                                                             |
| -------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------ |
| Check the dataset exists                                 | `python main.py check-data`                    | Console message                                                    |
| Data audit (missing values, duplicates, types, outliers) | `python -m src.data.clean`                     | Console report                                                     |
| EDA figures                                              | `python -m src.data.eda`                       | Console summary, figures in `outputs/figures/`                     |
| Regression baseline (XGBoost, Set A)                     | `python main.py train --task regression`       | `outputs/tables/xgb_regression_baseline.csv`                       |
| Classification baseline (XGBoost, Set A)                 | `python main.py train --task classification`   | `outputs/tables/xgb_classification_baseline.csv`, confusion matrix |
| Both supervised baselines                                | `python main.py train --task all`              | Both tables above                                                  |
| K-Means clustering (K = 6)                               | `python -m src.models.clustering.kmeans_model` | Cluster sizes and profiles in the console                          |

### Planned for Milestone 2

These commands exist in the CLI but are not implemented yet:
`analyze`, `preprocess`, `cluster`, `evaluate`, `predict`, and the Streamlit
app (`streamlit run app/app.py`).

## Milestone 1 baseline results (validation set)

| Task           | Model                        | Features              | Result                                                                 |
| -------------- | ---------------------------- | --------------------- | ---------------------------------------------------------------------- |
| Regression     | XGBoost (default parameters) | Set A (21)            | MAE 9.77, RMSE 14.98, R² 0.857                                         |
| Classification | XGBoost (default parameters) | Set A (21)            | Accuracy 0.774, Precision 0.667, Recall 0.502, F1 0.573, ROC-AUC 0.844 |
| Clustering     | K-Means (K = 6)              | 5 clustering features | Silhouette 0.301, Davies-Bouldin 1.139                                 |

The test set (20%) has not been used. It is reserved for one final evaluation
in Milestone 2.

Feature-set experiments (Sets B–D) and the reasons for using Set A only are
documented in [`reports/leakage_audit.md`](reports/leakage_audit.md).

## Project structure

- `src/data/` — loading, data audit, EDA
- `src/preprocessing/` — cleaning rules, preprocessing transformer, train/test split
- `src/features/` — supervised feature set (Set A)
- `src/models/` — regression, classification, clustering models
- `src/evaluation/` — clustering metrics (Milestone 2: comparison tables)
- `src/dimred/` — PCA and LDA (Milestone 2)
- `src/inference/` — prediction functions used by the CLI and app (Milestone 2)
- `app/` — Streamlit application (Milestone 2)
- `outputs/` — generated figures and tables
- `reports/` — milestone report, leakage audit, final report

## Submission

- GitHub: https://github.com/OmarElhosiny/ML-Project
- Final commit SHA: `<sha>`
