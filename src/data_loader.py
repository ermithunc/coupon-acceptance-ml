"""Load official coupon-acceptance CSVs without modifying raw files.

The test set is isolated: loaders never mix test rows into training frames,
and they never use test labels (none exist in the official test file).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR_CANDIDATES = (
    PROJECT_ROOT / "Dataset" / "Datasets",
    PROJECT_ROOT / "data" / "raw",
)

IDENTIFIER_COL = "customer_id"
TARGET_COL = "Y"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

# Official problem statement sizes. Actual files matched these counts.
EXPECTED_TRAIN_ROWS = 10147
EXPECTED_TEST_ROWS = 2537

TRAIN_COLUMNS = [
    "customer_id",
    "destination",
    "passanger",
    "weather",
    "temperature",
    "time",
    "coupon",
    "expiration",
    "gender",
    "age",
    "maritalStatus",
    "has_children",
    "education",
    "occupation",
    "income",
    "car",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
    "Y",
]

TEST_COLUMNS = [col for col in TRAIN_COLUMNS if col != TARGET_COL]
SAMPLE_SUBMISSION_COLUMNS = [IDENTIFIER_COL, TARGET_COL]

# Inspected encodings from the actual CSVs (empty string = missing).
CSV_NA_VALUES = ["", "NA", "NaN", "null"]


class DataContractError(ValueError):
    """Raised when a raw file does not match the documented data contract."""


def find_raw_dir() -> Path:
    """Return the first existing directory that contains the official CSVs."""
    for candidate in RAW_DIR_CANDIDATES:
        if (candidate / TRAIN_FILE).is_file() and (candidate / TEST_FILE).is_file():
            return candidate
    searched = ", ".join(str(p) for p in RAW_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Could not find train.csv and test.csv. Looked in: " + searched
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, na_values=CSV_NA_VALUES, keep_default_na=True)


def _require_columns(df: pd.DataFrame, expected: list[str], file_name: str) -> None:
    actual = list(df.columns)
    if actual != expected:
        raise DataContractError(
            f"{file_name} columns do not match the contract.\n"
            f"expected={expected}\nactual={actual}"
        )


def _require_row_count(df: pd.DataFrame, expected: int, file_name: str) -> None:
    n = len(df)
    if n != expected:
        raise DataContractError(
            f"{file_name} has {n} rows; official expected size is {expected}."
        )


def load_train(*, validate: bool = True) -> pd.DataFrame:
    """Load training data only. Does not read or join the test file."""
    path = find_raw_dir() / TRAIN_FILE
    df = _read_csv(path)
    if validate:
        _require_columns(df, TRAIN_COLUMNS, TRAIN_FILE)
        _require_row_count(df, EXPECTED_TRAIN_ROWS, TRAIN_FILE)
        if TARGET_COL not in df.columns:
            raise DataContractError("train.csv is missing target column Y.")
        if IDENTIFIER_COL not in df.columns:
            raise DataContractError("train.csv is missing identifier customer_id.")
    return df.copy()


def load_test(*, validate: bool = True) -> pd.DataFrame:
    """Load unlabeled test data. Isolated from training."""
    path = find_raw_dir() / TEST_FILE
    df = _read_csv(path)
    if validate:
        _require_columns(df, TEST_COLUMNS, TEST_FILE)
        _require_row_count(df, EXPECTED_TEST_ROWS, TEST_FILE)
        if TARGET_COL in df.columns:
            raise DataContractError("test.csv unexpectedly contains target column Y.")
        if IDENTIFIER_COL not in df.columns:
            raise DataContractError("test.csv is missing identifier customer_id.")
    return df.copy()


def load_sample_submission(*, validate: bool = True) -> pd.DataFrame:
    """Load the official submission template (not ground-truth test labels)."""
    path = find_raw_dir() / SAMPLE_SUBMISSION_FILE
    df = _read_csv(path)
    if validate:
        _require_columns(df, SAMPLE_SUBMISSION_COLUMNS, SAMPLE_SUBMISSION_FILE)
        _require_row_count(df, EXPECTED_TEST_ROWS, SAMPLE_SUBMISSION_FILE)
    return df.copy()


def split_xy(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate identifier+features from target. Training frame only."""
    if TARGET_COL not in train_df.columns:
        raise DataContractError("Cannot split X/y: target Y is missing.")
    y = train_df[TARGET_COL].copy()
    X = train_df.drop(columns=[TARGET_COL])
    return X, y


def feature_columns(include_identifier: bool = False) -> list[str]:
    """Feature names from the train contract, excluding Y."""
    cols = [col for col in TRAIN_COLUMNS if col != TARGET_COL]
    if include_identifier:
        return cols
    return [col for col in cols if col != IDENTIFIER_COL]
