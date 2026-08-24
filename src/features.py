"""Feature groups for controlled OOF ablation experiments."""

import pandas as pd


AVAILABLE_FEATURE_SETS = ("raw", "time_residual", "weekend", "missingness")


def make_features(frame: pd.DataFrame, feature_set: str = "raw") -> pd.DataFrame:
    """Create one additive feature group without using the target column.

    Each group is intentionally isolated so its contribution can be measured
    against the same raw baseline and folds.
    """
    if feature_set not in AVAILABLE_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}. Choose from {AVAILABLE_FEATURE_SETS}.")

    result = frame.copy()
    if feature_set == "raw":
        return result

    if feature_set == "time_residual":
        result["screen_activity_total"] = (
            result["social_media_hours"]
            + result["gaming_hours"]
            + result["work_study_hours"]
        )
        result["screen_time_residual"] = (
            result["daily_screen_time_hours"] - result["screen_activity_total"]
        )

    elif feature_set == "weekend":
        result["weekend_delta"] = result["weekend_screen_time"] - result["daily_screen_time_hours"]
        result["weekend_ratio"] = result["weekend_screen_time"] / (result["daily_screen_time_hours"] + 0.1)

    elif feature_set == "missingness":
        source_columns = [column for column in result.columns if column not in {"id", "addicted_label"}]
        result["missing_count"] = result[source_columns].isna().sum(axis=1).astype("int8")
        for column in source_columns:
            result[f"{column}_is_missing"] = result[column].isna().astype("int8")

    return result
