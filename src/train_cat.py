"""Train a fold-safe raw CatBoost baseline for Kaggle Playground Series S6E8.

Run from the repository root:
    python -m src.train_cat --run-name catboost_raw_v01
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

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


def prepare_catboost_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return raw features with categorical missing values made CatBoost-safe."""
    result = frame.loc[:, feature_columns].copy()
    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    return result


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

    x_train_all = prepare_catboost_features(train, feature_columns)
    x_test = prepare_catboost_features(test, feature_columns)
    oof_prediction = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    importance_frames: list[pd.DataFrame] = []

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 250,
        "learning_rate": 0.15,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "random_seed": model_seed,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
    }

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = CatBoostClassifier(**params)
        model.fit(
            x_train_all.loc[train_mask],
            y.loc[train_mask],
            cat_features=CATEGORICAL_COLUMNS,
            eval_set=(x_train_all.loc[valid_mask], y.loc[valid_mask]),
            early_stopping_rounds=50,
            verbose=False,
        )
        valid_prediction = model.predict_proba(x_train_all.loc[valid_mask])[:, 1]
        oof_prediction[valid_mask] = valid_prediction
        test_prediction += model.predict_proba(x_test)[:, 1] / len(np.unique(folds))
        fold_auc = float(roc_auc_score(y.loc[valid_mask], valid_prediction))
        fold_scores.append(fold_auc)
        importance_frames.append(
            pd.DataFrame(
                {"feature": feature_columns, "importance": model.get_feature_importance(), "fold": fold}
            )
        )
        print(f"fold={fold} auc={fold_auc:.6f} best_iteration={model.get_best_iteration()}")

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
        "model": "CatBoost",
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
        "model": "CatBoost",
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
        "notes": "raw CatBoost baseline; native numeric missing values",
    }
    experiment = pd.DataFrame([experiment_row])
    if EXPERIMENTS_PATH.exists():
        experiment = pd.concat([pd.read_csv(EXPERIMENTS_PATH), experiment], ignore_index=True)
    experiment.to_csv(EXPERIMENTS_PATH, index=False)
    print(f"Saved submission: {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="catboost_raw_v01")
    parser.add_argument("--model-seed", type=int, default=42)
    args = parser.parse_args()
    main(args.run_name, args.model_seed)
