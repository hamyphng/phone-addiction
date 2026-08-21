from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_COLUMNS, DATA_DIR, ID_COLUMN, TARGET


def load_competition_data(data_dir: Path = DATA_DIR):
    """Load CSVs and validate the invariants required for a safe submission."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    if TARGET not in train:
        raise ValueError(f"Training data is missing target column: {TARGET}")
    if TARGET in test:
        raise ValueError("Test data unexpectedly contains the target column.")
    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("IDs must be unique in both train and test.")
    if not test[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        raise ValueError("test IDs do not exactly match sample_submission IDs and order.")

    feature_columns = [column for column in test.columns if column != ID_COLUMN]
    expected = set(feature_columns) | {ID_COLUMN, TARGET}
    if set(train.columns) != expected:
        raise ValueError("Train/test feature schemas do not match.")

    missing_categoricals = set(CATEGORICAL_COLUMNS) - set(feature_columns)
    if missing_categoricals:
        raise ValueError(f"Missing expected categorical columns: {sorted(missing_categoricals)}")

    return train, test, sample_submission, feature_columns
