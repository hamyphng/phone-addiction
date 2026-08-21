<div align="center">

# Predicting Smartphone Addiction

### Kaggle Playground Series — Season 6, Episode 8

<p>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-0969da?style=for-the-badge" alt="Task"/>
  <img src="https://img.shields.io/badge/Metric-ROC%20AUC-6f42c1?style=for-the-badge" alt="Metric"/>
  <img src="https://img.shields.io/badge/Baseline-LightGBM-2ea44f?style=for-the-badge" alt="Baseline"/>
  <img src="https://img.shields.io/badge/OOF%20AUC-0.96372-orange?style=for-the-badge" alt="OOF AUC"/>
</p>

<p>
A reproducible competition pipeline for predicting smartphone addiction,
with fixed stratified cross-validation, fold-safe categorical preprocessing,
out-of-fold experiment tracking, and competition-ready probability submissions.
</p>

**Current baseline: 5-fold LightGBM OOF ROC AUC = 0.96372 ± 0.00052.**

[Competition Page](https://www.kaggle.com/competitions/playground-series-s6e8)

</div>

---

## Project at a Glance

| | |
|---|---|
| **Competition** | Kaggle Playground Series S6E8 |
| **Goal** | Predict smartphone addiction |
| **Task** | Binary classification |
| **Target** | `addicted_label` |
| **Evaluation** | ROC AUC |
| **Validation** | Fixed 5-fold StratifiedKFold |
| **Current Model** | LightGBM |
| **Current OOF AUC** | **0.963722** |
| **Fold-to-fold Std.** | **0.000519** |
| **Submission** | Probability for `addicted_label` |
| **Status** | Baseline established; model diversity and ensembling in progress |

> **Core principle:** optimize against a reliable OOF protocol first, then use the public leaderboard as an external check rather than as the primary validation signal.

---

## Competition Objective

The task is to estimate the probability that each test observation belongs to the positive smartphone-addiction class.

Kaggle evaluates submissions using **area under the ROC curve (ROC AUC)**, so ranking quality matters rather than a fixed classification threshold.

The required submission schema is:

```text
id,addicted_label
691369,0.2
691370,0.3
691371,0.1
...
```

Predictions therefore remain continuous probabilities rather than hard `0/1` labels.

---

## Validation Strategy

A single fixed validation protocol is used across the main experiments:

```text
Training Data
     │
     ▼
5-Fold StratifiedKFold
     │
     ├── Fold 0
     ├── Fold 1
     ├── Fold 2
     ├── Fold 3
     └── Fold 4
     │
     ▼
Out-of-Fold Predictions
     │
     ▼
Global ROC AUC
```

Configuration:

```text
n_splits     = 5
shuffle      = True
random_state = 42
```

Using the same folds across experiments makes model and feature comparisons more reliable.

### Leakage-Safe Categorical Processing

The current data contract treats the following variables as categorical:

- `gender`
- `stress_level`
- `academic_work_impact`

For the LightGBM baseline, the categorical encoder is fitted **inside each fold using training rows only** and then applied to that fold's validation rows and the competition test set.

This prevents category information from the validation fold from leaking into preprocessing.

---

## Data Validation

Before training, the pipeline verifies several competition-critical invariants:

- `addicted_label` exists in the training data;
- the target is absent from the test data;
- training and test IDs are unique;
- test IDs exactly match `sample_submission.csv` in both values and order;
- train/test feature schemas match;
- expected categorical columns are available.

These checks reduce the risk of producing a structurally invalid Kaggle submission.

---

## Current Baseline

The first reproducible baseline uses LightGBM on the raw competition features.

### Model Configuration

| Parameter | Value |
|---|---:|
| Estimators | 3000 |
| Learning rate | 0.04 |
| Number of leaves | 63 |
| Minimum child samples | 100 |
| Row subsampling | 0.85 |
| Column subsampling | 0.85 |
| L2 regularization | 1.0 |
| Early stopping | 150 rounds |

### 5-Fold Results

| Fold | ROC AUC |
|---:|---:|
| 0 | 0.962942 |
| 1 | 0.963593 |
| 2 | 0.964038 |
| 3 | **0.964497** |
| 4 | 0.963574 |
| **OOF** | **0.963722** |

Fold variability is small:

\[
\sigma_{\mathrm{fold}} \approx 0.000519
\]

which suggests that the fixed split provides a reasonably stable baseline for controlled experiments.

---

## Training and Submission Pipeline

```text
train.csv / test.csv / sample_submission.csv
                     │
                     ▼
              Schema Validation
                     │
                     ▼
            Fixed Stratified Folds
                     │
                     ▼
          Fold-Safe Preprocessing
                     │
                     ▼
                LightGBM
                     │
             ┌───────┴────────┐
             ▼                ▼
      OOF Predictions    Test Predictions
             │                │
             ▼                ▼
          OOF AUC       Fold Averaging
                              │
                              ▼
                       sample_submission
                              │
                              ▼
                     Kaggle Submission
```

For each fold, the model is trained on four folds and validated on the remaining fold. Test probabilities are averaged across the five fitted models.

The pipeline also saves:

- fold assignments;
- OOF predictions;
- averaged test predictions;
- feature importance;
- per-run metrics;
- experiment metadata;
- ready-to-submit CSV files.

---

## Experiment Tracking

`experiments.csv` acts as the experiment registry.

Each run records:

| Field | Purpose |
|---|---|
| `run_id` | Unique experiment identifier |
| `model` | Model family |
| `feature_set` | Feature configuration |
| `fold_seed` | CV reproducibility |
| `model_seed` | Model reproducibility |
| `params` | Training configuration |
| `mean_auc` | Global OOF ROC AUC |
| `std_auc` | Fold stability |
| `fold_*_auc` | Individual fold results |
| `oof_path` | OOF artifact |
| `test_pred_path` | Test-prediction artifact |
| `public_lb` | Optional leaderboard checkpoint |
| `notes` | Experiment rationale / observations |

This makes local CV the primary experiment record rather than relying on leaderboard memory.

---

## Competition-Specific Modeling Strategy

The dataset is synthetically generated, so the project emphasizes controlled experimentation over indiscriminate feature engineering.

### 01 — Strong Raw Baselines

Establish competitive raw-feature models before adding transformations.

Current:

- **LightGBM** — implemented

Planned complementary models:

- XGBoost
- CatBoost
- Logistic Regression sanity baseline

### 02 — Dataset Forensics

Investigate:

- target-rate surfaces;
- train/test distribution drift;
- missingness drift;
- categorical frequency differences;
- high-signal raw variables;
- synthetic-generator structure.

### 03 — Controlled Feature Ablations

Feature engineering is accepted only when it improves the fixed OOF protocol consistently.

Explicit missingness indicators are not treated as default features because train/test missing-rate drift can make them encode split identity rather than transferable target signal.

### 04 — Model Diversity

Once strong individual models are available, compare their OOF prediction correlations.

The goal is not simply to collect more models, but to find models with complementary errors.

### 05 — OOF Ensemble

Blend weights should be selected using OOF predictions rather than public-leaderboard trial and error.

### 06 — Finalist Tuning

Hyperparameter optimization is reserved for model and feature families that have already demonstrated robust value.

---

## Why Raw Features Matter

Public community analysis for this competition indicates that a small set of behavioral variables carries much of the useful signal, particularly variables related to:

- daily screen time;
- social-media usage;
- weekend screen time;
- notifications;
- app-opening frequency.

This supports the project's decision to establish strong raw gradient-boosting baselines before introducing large numbers of engineered ratios or interaction features.

---

## Repository Structure

```text
phone-addiction/
├── .gitignore
├── experiments.csv
├── requirements.txt
├── requirements.lock.txt
├── plan_s6e8_v2.md
├── code_explainer.html
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── cv.py
│   ├── data.py
│   └── train_lgbm.py
│
├── data/                 # local competition files; not committed
├── artifacts/            # generated OOF / metrics / importance
├── submissions/          # generated Kaggle submissions
└── models/               # optional persisted models
```

Generated competition data, model artifacts and submissions are intentionally excluded from version control.

---

## Reproduce the Baseline

### 1. Create the environment

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download the Kaggle competition data

Place the following files inside `data/`:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Competition:

https://www.kaggle.com/competitions/playground-series-s6e8

### 3. Run the baseline

```bash
python -m src.train_lgbm --run-name lgbm_raw_v01
```

The pipeline will generate OOF predictions, averaged test predictions, feature importance, experiment metrics and a submission CSV.

---

## Roadmap

| Stage | Status |
|---|---|
| Reproducible environment | **Completed** |
| Fixed stratified folds | **Completed** |
| Schema / submission validation | **Completed** |
| Fold-safe preprocessing | **Completed** |
| Raw LightGBM baseline | **Completed** |
| OOF experiment registry | **Completed** |
| Competition-specific EDA | In progress |
| Dataset forensics | Planned |
| XGBoost baseline | Planned |
| CatBoost baseline | Planned |
| Feature ablations | Planned |
| OOF diversity analysis | Planned |
| OOF blending | Planned |
| Finalist tuning | Planned |
| Final submission selection | Planned |

---

## Competition Notes

- Metric: **ROC AUC**
- Target: **`addicted_label`**
- Submission type: **continuous probability**
- Maximum team size: **3**
- Submission limit: **10 per day**
- Up to **2 final submissions**
- Final deadline: **31 August 2026, 23:59 UTC**

The competition data is licensed under **CC BY 4.0**. External data is permitted subject to the competition rules.

---

## References

- [Kaggle — Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)
- [Competition Evaluation](https://www.kaggle.com/competitions/playground-series-s6e8/overview/evaluation)
- [Competition Rules](https://www.kaggle.com/competitions/playground-series-s6e8/rules)

---

<div align="center">

### Kaggle Playground Series S6E8

**Reliable CV • Tabular Gradient Boosting • OOF Experimentation • Ensembling**

</div>
