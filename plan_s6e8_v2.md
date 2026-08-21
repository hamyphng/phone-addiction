# Kaggle Playground Series S6E8 — Competition Plan v2

## 0. Competition objective

**Competition:** Predicting Smartphone Addiction — Playground Series S6E8  
**Task:** Binary classification  
**Target:** `addicted_label`  
**Metric:** ROC AUC  
**Submission:** `id,addicted_label` where `addicted_label` is a probability/ranking score.  
**Deadline:** 31 Aug 2026, 23:59 UTC = 01 Sep 2026, 06:59 ICT (Hanoi/Bangkok).  
**Submission limit:** 10/day; up to 2 final submissions.

### Main strategy

Do **not** start by engineering many features.

```text
Reliable CV
   ↓
Strong raw GBM baselines
   ↓
Dataset forensics / synthetic-generator analysis
   ↓
Small controlled feature ablations
   ↓
Model diversity
   ↓
OOF ensemble
   ↓
Tune finalists only
   ↓
Robust final submissions
```

The competition data is synthetic. Community experiments also show that raw variables carry very strong signal, while explicit missingness features and many arithmetic ratios can fail to transfer from local CV to the leaderboard. Therefore feature engineering is treated as an **experiment**, not as the center of the solution.

---

# 1. Tool stack — what each tool is for

| Tool | Used for | Why we use it |
|---|---|---|
| **Python 3.11** | Main language | Stable support for the tabular ML ecosystem |
| **VS Code** | Project/code management | Easier to maintain reusable scripts than notebook-only work |
| **Jupyter Lab** | EDA and visual experiments | Fast iteration and visualization |
| **pandas** | Load, inspect, transform CSV data | Core tabular data manipulation |
| **NumPy** | Arrays/math/vectorized operations | Fast numerical operations |
| **scikit-learn** | CV, pipelines, preprocessing, AUC | Provides `StratifiedKFold`, `roc_auc_score`, encoders, Logistic Regression |
| **matplotlib** | EDA plots | Distribution, target-rate and 2D surface plots |
| **LightGBM** | Main gradient-boosted tree model | Fast on ~700k tabular rows; strong raw-feature baseline |
| **XGBoost** | Second strong tree family | Adds prediction diversity for blending |
| **CatBoost** | Third strong tree family | Handles categoricals/missing data naturally; useful complementary errors |
| **Optuna** | Hyperparameter optimization | Used only after choosing strong model/feature families |
| **SHAP** | Model interpretation | Used selectively to understand nonlinear effects/interactions |
| **joblib** | Save fitted artifacts/models if needed | Reproducibility without retraining every time |
| **Kaggle CLI/API** | Download/submit competition files | Faster submission workflow from terminal |
| **Git** | Track code/experiment changes | Prevent losing a working pipeline while experimenting |

### Tools that are intentionally *not* core

- Deep learning frameworks: unnecessary unless later evidence shows a clear benefit.
- AutoML: allowed by the rules, but not needed for the main learning/competition workflow.
- Heavy dashboarding libraries: not useful for score improvement.
- SHAP on every run: too slow and unnecessary; use only on finalists or forensics.

---

# 2. Project structure

```text
phone_addiction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── external/                  # optional public original-like dataset
│
├── notebooks/
│   ├── 01_audit_eda.ipynb
│   ├── 02_forensics.ipynb
│   └── 03_ensemble_analysis.ipynb
│
├── src/
│   ├── config.py                  # seeds, folds, paths
│   ├── data.py                    # load/validate data
│   ├── features.py                # optional feature ablations
│   ├── cv.py                      # fixed fold creation
│   ├── train_lgbm.py
│   ├── train_xgb.py
│   ├── train_cat.py
│   ├── ensemble.py
│   └── submit.py
│
├── artifacts/
│   ├── folds.csv
│   ├── oof/
│   ├── test_preds/
│   ├── importance/
│   └── eda/
│
├── submissions/
├── models/
├── experiments.csv
├── requirements.txt
└── plan.md
```

### Why this structure?

- **Notebook = exploration.**
- **`src/` = reusable pipeline.**
- **OOF predictions are first-class artifacts**, because ensemble decisions must be made from OOF rather than public leaderboard guessing.
- `experiments.csv` becomes the single source of truth for experiments.

---

# 3. Phase 0 — Environment and reproducibility

## Goal

Make every experiment repeatable before trying to improve score.

## Tools

- Python / `.venv`
- `pip`
- Git
- pandas
- scikit-learn

## Tasks

- [ ] Keep direct dependencies and version bounds in `requirements.txt`.
- [ ] Create `requirements.lock.txt` for a reproducible environment using `pip freeze > requirements.lock.txt` after the first successful run.
- [ ] Create project folders.
- [ ] Fix global seed initially at `42`.
- [ ] Create one fixed 5-fold stratified split and save fold assignment to `artifacts/folds.csv`.
- [ ] Create `experiments.csv`.

### CV

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Every main experiment must use the same folds.

### Categorical-data contract

- `gender`, `stress_level`, and `academic_work_impact` must be encoded inside each CV fold for LightGBM/XGBoost; fit the encoder only on that fold's training rows.
- For CatBoost, fill missing categorical values with a sentinel such as `__MISSING__` and pass the columns as categorical features.
- Numeric missing values stay missing for tree models unless an imputation experiment explicitly tests another treatment.

### Why `StratifiedKFold`?

The target is binary. Stratification keeps approximately the same positive/negative proportion in each fold, reducing variance between folds.

---

# 4. Phase 1 — Data audit and competition-specific EDA

## Goal

Understand the dataset before modelling, especially train/test drift and the structure of the synthetic target.

## Tools

### `pandas`
Use for:
- `shape`, `dtypes`, `nunique`
- missing rates
- quantiles
- train/test comparison
- groupby target rates

### `NumPy`
Use for:
- vectorized binning/calculations
- numerical checks

### `matplotlib`
Use for:
- distributions
- target-rate curves
- 2D heatmaps

### `sklearn.metrics.roc_auc_score`
Use for:
- univariate feature signal
- sanity-check predictions

## Checklist

- [ ] Confirm target, IDs and submission alignment.
- [ ] Check target class balance.
- [ ] Compare train/test dtypes and ranges.
- [ ] Compute missing rate train vs test for every column.
- [ ] Check numeric train/test distribution drift.
- [ ] Check categorical value frequencies.
- [ ] Calculate univariate ROC AUC for numeric variables.
- [ ] Plot target rate against major raw features.
- [ ] Plot the following 2D target surfaces:
  - `daily_screen_time_hours × social_media_hours`
  - `daily_screen_time_hours × weekend_screen_time`
  - `notifications_per_day × app_opens_per_day`

## Important competition-specific rule

**Do not automatically create missing indicators.**

Community analysis found substantial missing-rate differences between train and test, while missing-count features can improve local CV slightly but hurt leaderboard score. Therefore:

```text
missing_count / col_is_missing
= diagnostic/ablation only
≠ default features
```

## Output

- `artifacts/eda/eda_summary.md`
- important plots
- a short list of modelling hypotheses

---

# 5. Phase 2 — Reliable raw baselines

## Goal

Measure how strong the raw data already is before feature engineering.

## 5.1 Logistic Regression — sanity baseline

### Tools

- `sklearn.pipeline.Pipeline`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `LogisticRegression`

### Purpose

Not expected to win. It answers:

> Is the target mostly linear/simple, or do nonlinear tree interactions matter strongly?

Save:

- `artifacts/oof/oof_logreg.csv`
- fold AUCs

Implementation note: ordinal-encode the three categorical columns inside each fold; do not use a single encoder fit on all training rows.

---

## 5.2 LightGBM — primary baseline

### Tool

`lightgbm.LGBMClassifier`

### Why

- fast on large tabular data
- native missing-value handling
- strong nonlinear interactions
- useful gain-based feature importance

Start with reasonable fixed parameters + early stopping. Do not Optuna yet.

Save:

- OOF prediction
- averaged test prediction
- fold AUC
- best iteration per fold
- gain importance

This should be the **first Kaggle submission** to validate the full pipeline.

---

## 5.3 XGBoost — diversity baseline

### Tool

`xgboost.XGBClassifier`

### Why

Even if its standalone AUC is close to LightGBM, different tree construction and regularization can create useful residual diversity for ensemble.

Use the same folds and raw features.

---

## 5.4 CatBoost — categorical/diversity baseline

### Tool

`catboost.CatBoostClassifier`

### Why

- categorical features can stay categorical
- native missing handling
- different boosting implementation from LGBM/XGB

Use early stopping and the same fold assignments.

---

# 6. Phase 2 decision gate

Create a comparison table:

| Model | Mean OOF AUC | Std | Worst fold | Correlation with LGBM | Runtime |
|---|---:|---:|---:|---:|---:|
| Logistic | | | | | |
| LGBM | | | | 1.000 | |
| XGB | | | | | |
| CatBoost | | | | | |

## Keep a model if

1. OOF is strong, **or**
2. its prediction errors differ enough to help an ensemble.

Do not remove a model solely because it is 0.0002–0.0005 below the best model.

---

# 7. Phase 3 — Synthetic dataset forensics

## Goal

Understand whether the synthetic generator has transformed a simpler original relationship into the competition distribution.

## Tools

### pandas + matplotlib
Compare distributions and conditional target probabilities.

### scikit-learn
Use CV to test small hypotheses rather than eyeballing plots.

### Optional SHAP
Use only if tree behaviour is difficult to understand from gain importance and target surfaces.

## Experiments

- [ ] Inspect the public original-like smartphone addiction dataset only if it is publicly/equally accessible and compliant with competition rules.
- [ ] Align shared columns carefully.
- [ ] Compare feature distributions between original-like, train and test.
- [ ] Check whether simple rules around `daily_screen_time_hours` / `social_media_hours` appear in the original-like data.
- [ ] Check whether those rules remain sharp or become smooth probabilistic boundaries in S6E8.
- [ ] Measure correlations that changed after synthesis.

## Critical caution

The proposed original dataset and its hard-rule generator are **community hypotheses**, not official Kaggle ground truth.

Use this phase to generate modelling hypotheses; do not hard-code a community rule into final predictions unless it validates robustly on S6E8 OOF.

---

# 8. Phase 4 — Feature ablation, not feature explosion

## Goal

Test only features with a clear hypothesis.

## Tool

`src/features.py` + the same LightGBM CV pipeline.

Every feature group is tested independently against the raw baseline.

## Priority A — transformations of dominant variables

Try a small number of candidates such as:

- log-like transforms if a variable is strongly skewed
- coarse bins for major screen-time variables
- carefully motivated interaction features
- rule-distance features around empirically observed transition zones

Example concept:

```text
distance_to_screen_boundary
interaction between daily_screen_time and social_media_hours
```

Only implement after EDA/forensics suggests a boundary.

## Priority B — domain ratios

Test separately:

- `weekend_delta`
- `weekend_ratio`
- `notifications_per_open`
- selected sleep/screen ratios

These are **low priority** because tree models can often infer equivalent relationships directly from raw features.

## Priority C — missingness

Only as a negative-control experiment:

- `missing_count`
- selected `_is_missing`

Expected default decision: **do not keep** unless repeated CV + LB evidence supports transfer.

## Acceptance criterion

A feature group stays only if:

- mean OOF AUC improves,
- improvement is not concentrated in one fold,
- result repeats with another seed or repeat CV,
- no obvious train/test drift risk is introduced.

---

# 9. Phase 5 — Model interpretation

## Goal

Understand what the best models are learning before tuning/blending aggressively.

## Tools

### LightGBM gain importance
Fast global ranking of useful features.

### SHAP
Use on a sample, not the full 700k rows.

Questions to answer:

- Which raw features dominate?
- Are engineered features actually used?
- Are top interactions consistent with EDA?
- Does a model depend on missingness artifacts?

SHAP is for **diagnosis**, not for increasing score directly.

---

# 10. Phase 6 — Ensemble before heavy tuning

## Goal

Exploit different model errors under ROC AUC.

## Tools

- pandas / NumPy
- `sklearn.metrics.roc_auc_score`
- `scipy.stats.rankdata` or `Series.rank(pct=True)`

## Step 1 — prediction correlation

Calculate Pearson/Spearman correlation between OOF predictions.

## Step 2 — probability blend

Examples:

```text
0.50 LGBM + 0.30 XGB + 0.20 CatBoost
0.40 LGBM + 0.30 XGB + 0.30 CatBoost
```

Search weights using **OOF only**.

## Step 3 — rank blend

Because ROC AUC depends on ranking, convert each model prediction to ranks before averaging.

```text
rank_lgbm
rank_xgb
rank_cat
    ↓
weighted average
```

Test both:

- probability blend
- rank blend

Keep whichever has stronger OOF.

Before writing a rank-blend submission, normalize its final ranks to the interval `[0, 1]` so `addicted_label` remains a valid probability-like score.

## Important

Never optimize blend weights on Public LB.

---

# 11. Phase 7 — Hyperparameter tuning

## Goal

Tune only the 1–2 strongest model families after the pipeline and feature set are stable.

## Tool

**Optuna**

### Why Optuna now?

Before this point, hyperparameter tuning can hide bigger questions:

- wrong feature set?
- wrong validation?
- insufficient model diversity?

Now those questions are already answered.

## Efficient workflow

```text
Optuna 3-fold / fixed holdout
        ↓
select top configurations
        ↓
confirm with canonical 5-fold CV
```

### LightGBM search

- `learning_rate`
- `num_leaves`
- `max_depth`
- `min_child_samples`
- `feature_fraction/colsample_bytree`
- `bagging_fraction/subsample`
- `reg_alpha`
- `reg_lambda`

### XGBoost search

- `max_depth`
- `min_child_weight`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

### CatBoost search

- `depth`
- `learning_rate`
- `l2_leaf_reg`
- `random_strength`
- `bagging_temperature`

Do not waste trials tuning `n_estimators/iterations` tightly; use a high ceiling + early stopping.

---

# 12. Phase 8 — Multi-seed robustness

## Goal

Ensure tiny OOF improvements are not seed noise.

## Tools

Same model libraries; change only random seeds.

Recommended finalist seeds:

```text
42
2026
3407
```

For each finalist:

- run same 5 folds under multiple model seeds
- average test predictions
- optionally average/rank-average OOF predictions

Do **not** multi-seed every early experiment; it is too expensive.

---

# 13. Phase 9 — Final submission selection

Kaggle permits 10 submissions/day and up to 2 final submissions.

## Submission A — conservative

Best robust OOF ensemble.

Example:

```text
multi-seed LGBM/XGB/CatBoost rank blend
```

## Submission B — diversified

A materially different but still strong candidate, e.g.:

- probability blend instead of rank blend
- raw-model ensemble vs forensic-feature ensemble
- different model-family weighting

The purpose is to reduce Private LB risk, not to submit two nearly identical files.

## Submission checks

- [ ] Exact `id` order from `sample_submission.csv`
- [ ] Columns exactly `id, addicted_label`
- [ ] No missing/infinite predictions
- [ ] Values valid as probability/ranking scores
- [ ] No accidental index column
- [ ] Save source experiment IDs in submission filename/log

---

# 14. Experiment log schema

`experiments.csv`

```text
run_id
created_at
model
feature_set
fold_seed
model_seed
params
mean_auc
std_auc
fold_0_auc
fold_1_auc
fold_2_auc
fold_3_auc
fold_4_auc
oof_path
test_pred_path
public_lb
notes
```

### Rule

Public LB is recorded **after** the experiment. It is never used to retroactively choose folds or fit blend weights.

---

# 15. Recommended order from now to deadline

## Stage A — establish the benchmark

1. `01_audit_eda.ipynb`
2. fixed folds
3. raw LightGBM
4. first valid Kaggle submission
5. raw XGBoost
6. raw CatBoost
7. compare OOF + prediction correlations

## Stage B — understand the data

8. synthetic/original-like forensics
9. 2D target surfaces
10. small feature ablations

## Stage C — climb score efficiently

11. raw-model ensemble
12. rank ensemble
13. tune only finalists
14. multi-seed finalists
15. final two submissions

---

# 16. Practical schedule — 22 Aug to deadline

| Date | Main work |
|---|---|
| **22 Aug** | Audit/EDA + fixed folds + raw LightGBM |
| **23 Aug** | XGBoost + CatBoost + OOF comparison |
| **24 Aug** | Forensics + 2D target surfaces |
| **25 Aug** | Feature ablations |
| **26 Aug** | Ensemble + rank blend |
| **27–28 Aug** | Optuna finalists |
| **29 Aug** | Multi-seed robustness |
| **30 Aug** | Final ensemble candidates + clean rerun |
| **31 Aug** | Final submission selection; no risky large rewrite |

Keep the final day for validation and submission safety, not for rebuilding the entire pipeline.

---

# 17. Minimal installation

```powershell
pip install pandas numpy scikit-learn matplotlib lightgbm xgboost catboost optuna shap scipy joblib jupyter kaggle
```

Create a lock file after packages are installed:

```powershell
pip freeze > requirements.lock.txt
```

---

# 18. Daily commands

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Verify main packages
python -c "import pandas, sklearn, lightgbm, xgboost, catboost, optuna; print('environment OK')"

# Start notebook environment
jupyter lab
```

Example Kaggle CLI submission after authentication:

```powershell
kaggle competitions submit `
  -c playground-series-s6e8 `
  -f submissions/lgbm_raw_v01.csv `
  -m "raw LGBM fixed 5-fold baseline"
```

---

# 19. What each phase teaches you as a Data Scientist

| Phase | Skill being trained |
|---|---|
| Audit / EDA | data quality, drift, hypothesis formation |
| CV | correct offline evaluation |
| Logistic baseline | linear modelling + preprocessing |
| LGBM/XGB/CatBoost | modern tabular ML |
| Forensics | synthetic-data reasoning and distribution shift |
| Feature ablation | scientific experimentation |
| SHAP/importance | model interpretation |
| Ensemble | error diversity and ranking optimization |
| Optuna | efficient hyperparameter search |
| Experiment tracking | reproducible ML engineering |
| Final selection | leaderboard risk management |

---

# 20. Definition of done

The project is complete when it has:

- [ ] repeatable raw CSV → OOF → test prediction → submission pipeline
- [ ] fixed reproducible folds
- [ ] Logistic, LightGBM, XGBoost and CatBoost baselines
- [ ] OOF AUC and fold scores for every serious model
- [ ] train/test drift audit
- [ ] synthetic/original-like dataset forensics
- [ ] controlled feature ablation results
- [ ] OOF correlation matrix
- [ ] probability and rank ensemble experiments
- [ ] tuned finalists only
- [ ] multi-seed robustness check
- [ ] two defensible final submissions
- [ ] clean experiment log

---

# 21. Evidence behind the v2 changes

Official Kaggle information:

- Evaluation is ROC AUC.
- S6E8 uses synthetic Playground data.
- Deadline: 31 Aug 2026 23:59 UTC.
- Up to 10 submissions/day and 2 final submissions.
- Public/reasonably accessible external data is allowed under the competition rules.

Competition-specific community evidence motivating the strategy:

- A strict 5-fold LightGBM analysis reports that raw `daily_screen_time_hours`, `social_media_hours`, `weekend_screen_time`, `notifications_per_day`, and `app_opens_per_day` dominate gain importance.
- The same analysis reports that arithmetic feature engineering and explicit missing indicators did not improve the raw model reliably.
- Missing rates differ materially between train/test; `missing_count` showed a tiny local CV gain but a Public LB decline in one documented experiment.
- A community investigation of an original-like public dataset reports strong threshold-like relationships involving `daily_screen_time_hours` and `social_media_hours`; this is useful for forensics but remains a hypothesis rather than official ground truth.
