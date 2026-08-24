"""Train a fold-safe raw XGBoost baseline for Kaggle Playground Series S6E8.

Run from the repository root:
    python -m src.train_xgb --run-name xgb_raw_v01
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLUMNS,
    EXPERIMENTS_PATH,
    FOLD_SEED,
    ID_COLUMN,
    OOF_DIR,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PRED_DIR,
)
from src.cv import make_folds
from src.data import load_competition_data


def encode_fold(train_x, valid_x, test_x, feature_columns):
    """Fit categorical encoding on fold-training rows only."""
    numeric_columns = [column for column in feature_columns if column not in CATEGORICAL_COLUMNS]
    train_encoded = train_x.loc[:, numeric_columns].copy()
    valid_encoded = valid_x.loc[:, numeric_columns].copy()
    test_encoded = test_x.loc[:, numeric_columns].copy()

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float32,
    )
    train_encoded[CATEGORICAL_COLUMNS] = encoder.fit_transform(
        train_x[CATEGORICAL_COLUMNS].fillna("__MISSING__")
    )
    valid_encoded[CATEGORICAL_COLUMNS] = encoder.transform(
        valid_x[CATEGORICAL_COLUMNS].fillna("__MISSING__")
    )
    test_encoded[CATEGORICAL_COLUMNS] = encoder.transform(
        test_x[CATEGORICAL_COLUMNS].fillna("__MISSING__")
    )
    return train_encoded, valid_encoded, test_encoded


def main(run_name: str, model_seed: int) -> None:
    start = time.perf_counter()
    for directory in (ARTIFACTS_DIR, OOF_DIR, TEST_PRED_DIR, SUBMISSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    train, test, sample_submission, feature_columns = load_competition_data()
    y = train[TARGET]
    folds = make_folds(y)
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], "fold": folds}).to_csv(
        ARTIFACTS_DIR / "folds.csv", index=False
    )

    oof_prediction = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    importance_frames: list[pd.DataFrame] = []

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "random_state": model_seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
    }

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        x_train, x_valid, x_test = encode_fold(
            train.loc[train_mask, feature_columns],
            train.loc[valid_mask, feature_columns],
            test.loc[:, feature_columns],
            feature_columns,
        )
        model = XGBClassifier(**params)
        model.fit(x_train, y.loc[train_mask], eval_set=[(x_valid, y.loc[valid_mask])], verbose=False)
        valid_prediction = model.predict_proba(x_valid)[:, 1]
        oof_prediction[valid_mask] = valid_prediction
        test_prediction += model.predict_proba(x_test)[:, 1] / len(np.unique(folds))
        fold_auc = float(roc_auc_score(y.loc[valid_mask], valid_prediction))
        fold_scores.append(fold_auc)
        importance_frames.append(
            pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_, "fold": fold})
        )
        print(f"fold={fold} auc={fold_auc:.6f} best_iteration={model.best_iteration}", flush=True)

    overall_auc = float(roc_auc_score(y, oof_prediction))
    print(f"OOF AUC={overall_auc:.6f} +/- {np.std(fold_scores):.6f}")

    oof_path = OOF_DIR / f"{run_name}.csv"
    test_path = TEST_PRED_DIR / f"{run_name}.csv"
    submission_path = SUBMISSIONS_DIR / f"{run_name}.csv"
    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: y, "fold": folds, "prediction": oof_prediction}
    ).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "prediction": test_prediction}).to_csv(test_path, index=False)
    submission = sample_submission.copy()
    submission[TARGET] = np.clip(test_prediction, 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    (
        pd.concat(importance_frames, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .to_csv(ARTIFACTS_DIR / f"importance_{run_name}.csv", index=False)
    )

    metrics = {
        "run_name": run_name,
        "model": "XGBoost",
        "feature_set": "raw",
        "fold_seed": FOLD_SEED,
        "model_seed": model_seed,
        "params": params,
        "mean_auc": overall_auc,
        "std_fold_auc": float(np.std(fold_scores)),
        "fold_auc": fold_scores,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    with open(ARTIFACTS_DIR / f"metrics_{run_name}.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    experiment_row = {
        "run_id": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "XGBoost",
        "feature_set": "raw",
        "fold_seed": FOLD_SEED,
        "model_seed": model_seed,
        "params": json.dumps(params, sort_keys=True),
        "mean_auc": overall_auc,
        "std_auc": float(np.std(fold_scores)),
        **{f"fold_{fold}_auc": score for fold, score in enumerate(fold_scores)},
        "oof_path": str(oof_path.relative_to(ARTIFACTS_DIR.parent)),
        "test_pred_path": str(test_path.relative_to(ARTIFACTS_DIR.parent)),
        "public_lb": np.nan,
        "notes": "raw XGBoost baseline; fold-safe ordinal encoding",
    }
    experiment = pd.DataFrame([experiment_row])
    if EXPERIMENTS_PATH.exists():
        experiment = pd.concat([pd.read_csv(EXPERIMENTS_PATH), experiment], ignore_index=True)
    experiment.to_csv(EXPERIMENTS_PATH, index=False)
    print(f"Saved submission: {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="xgb_raw_v01")
    parser.add_argument("--model-seed", type=int, default=42)
    args = parser.parse_args()
    main(args.run_name, args.model_seed)
