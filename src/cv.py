import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import FOLD_SEED, N_FOLDS


def make_folds(y, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Return deterministic, stratified fold assignments."""
    folds = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, validation_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[validation_index] = fold
    if (folds < 0).any():
        raise RuntimeError("Some rows were not assigned to a fold.")
    return folds
