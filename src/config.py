from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
OOF_DIR = ARTIFACTS_DIR / "oof"
TEST_PRED_DIR = ARTIFACTS_DIR / "test_preds"
SUBMISSIONS_DIR = ROOT / "submissions"
EXPERIMENTS_PATH = ROOT / "experiments.csv"

TARGET = "addicted_label"
ID_COLUMN = "id"
CATEGORICAL_COLUMNS = ["gender", "stress_level", "academic_work_impact"]
N_FOLDS = 5
FOLD_SEED = 42
