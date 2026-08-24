"""Create an OOF-validated probability or rank ensemble.

Example:
    python -m src.ensemble --runs lgbm_raw_v01 xgb_raw_v01 \
        --weights 0.4 0.6 --method probability --name lgbm_xgb_prob_v01
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS_DIR, EXPERIMENTS_PATH, FOLD_SEED, ID_COLUMN, OOF_DIR, SUBMISSIONS_DIR, TARGET, TEST_PRED_DIR
from src.data import load_competition_data


def normalized_weights(weights: list[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if (values < 0).any() or values.sum() <= 0:
        raise ValueError("Weights must be non-negative and sum to a positive value.")
    return values / values.sum()


def blend(columns: list[pd.Series], weights: np.ndarray, method: str) -> np.ndarray:
    matrix = pd.concat(columns, axis=1)
    if method == "rank":
        matrix = matrix.rank(method="average", pct=True)
    return matrix.to_numpy() @ weights


def main(run_names: list[str], weights: list[float], method: str, name: str, write_log: bool = True) -> None:
    start = time.perf_counter()
    if len(run_names) < 2:
        raise ValueError("An ensemble needs at least two runs.")
    if len(run_names) != len(weights):
        raise ValueError("--runs and --weights must have the same length.")
    weights_array = normalized_weights(weights)
    train, test, sample_submission, _ = load_competition_data()

    oof_frames = [pd.read_csv(OOF_DIR / f"{run}.csv") for run in run_names]
    test_frames = [pd.read_csv(TEST_PRED_DIR / f"{run}.csv") for run in run_names]
    reference = oof_frames[0]
    for run, oof, test_pred in zip(run_names, oof_frames, test_frames):
        if not reference[ID_COLUMN].equals(oof[ID_COLUMN]):
            raise ValueError(f"OOF IDs are misaligned for {run}.")
        if not reference[TARGET].equals(oof[TARGET]):
            raise ValueError(f"OOF targets are misaligned for {run}.")
        if not test[ID_COLUMN].equals(test_pred[ID_COLUMN]):
            raise ValueError(f"Test IDs are misaligned for {run}.")

    oof_prediction = blend([frame["prediction"] for frame in oof_frames], weights_array, method)
    test_prediction = blend([frame["prediction"] for frame in test_frames], weights_array, method)
    oof_auc = float(roc_auc_score(reference[TARGET], oof_prediction))

    oof_path = OOF_DIR / f"{name}.csv"
    test_path = TEST_PRED_DIR / f"{name}.csv"
    submission_path = SUBMISSIONS_DIR / f"{name}.csv"
    pd.DataFrame(
        {
            ID_COLUMN: reference[ID_COLUMN],
            TARGET: reference[TARGET],
            "fold": reference["fold"],
            "prediction": oof_prediction,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "prediction": test_prediction}).to_csv(test_path, index=False)
    submission = sample_submission.copy()
    submission[TARGET] = np.clip(test_prediction, 0.0, 1.0)
    submission.to_csv(submission_path, index=False)

    fold_scores = {
        f"fold_{fold}_auc": float(
            roc_auc_score(reference.loc[reference.fold == fold, TARGET], oof_prediction[reference.fold == fold])
        )
        for fold in sorted(reference.fold.unique())
    }
    metrics = {
        "run_name": name,
        "model": "Ensemble",
        "feature_set": "raw-model blend",
        "fold_seed": FOLD_SEED,
        "runs": run_names,
        "weights": weights_array.tolist(),
        "method": method,
        "mean_auc": oof_auc,
        "fold_auc": list(fold_scores.values()),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    with open(ARTIFACTS_DIR / f"metrics_{name}.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    experiment_row = {
        "run_id": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "Ensemble",
        "feature_set": "raw-model blend",
        "fold_seed": FOLD_SEED,
        "model_seed": np.nan,
        "params": json.dumps({"runs": run_names, "weights": weights_array.tolist(), "method": method}),
        "mean_auc": oof_auc,
        "std_auc": np.nan,
        **fold_scores,
        "oof_path": str(oof_path.relative_to(ARTIFACTS_DIR.parent)),
        "test_pred_path": str(test_path.relative_to(ARTIFACTS_DIR.parent)),
        "public_lb": np.nan,
        "notes": "OOF-selected ensemble",
    }
    if write_log:
        experiment = pd.DataFrame([experiment_row])
        if EXPERIMENTS_PATH.exists():
            experiment = pd.concat([pd.read_csv(EXPERIMENTS_PATH), experiment], ignore_index=True)
        experiment.to_csv(EXPERIMENTS_PATH, index=False)

    print(f"OOF AUC={oof_auc:.6f}")
    print(f"Saved submission: {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--method", choices=["probability", "rank"], default="probability")
    parser.add_argument("--name", required=True)
    parser.add_argument("--skip-log", action="store_true", help="Regenerate output files without appending experiments.csv.")
    args = parser.parse_args()
    main(args.runs, args.weights, args.method, args.name, write_log=not args.skip_log)
