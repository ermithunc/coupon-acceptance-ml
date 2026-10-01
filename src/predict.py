"""Fit the selected model on the training file and score the official test file.

The test file is used only to produce submission rows. It is not used to fit
the pipeline or to choose the model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import (
    EXPECTED_TEST_ROWS,
    IDENTIFIER_COL,
    PROJECT_ROOT,
    TARGET_COL,
    load_sample_submission,
    load_test,
    load_train,
)

MODEL_PATH = PROJECT_ROOT / "models" / "inference_pipeline.joblib"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
DECISION_THRESHOLD = 0.5


def save_pipeline(pipeline, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_submission(pipeline, test_df: pd.DataFrame, threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """Predict class labels. Row order follows the test frame."""
    probabilities = pipeline.predict_proba(test_df)[:, 1]
    labels = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        {
            IDENTIFIER_COL: test_df[IDENTIFIER_COL].to_numpy(),
            TARGET_COL: labels,
        }
    )


def validate_submission(
    submission: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> dict:
    """Check the submission contract. Raises ValueError when a check fails."""
    problems: list[str] = []
    if list(submission.columns) != [IDENTIFIER_COL, TARGET_COL]:
        problems.append(f"columns are {list(submission.columns)}")
    if len(submission) != EXPECTED_TEST_ROWS:
        problems.append(f"row count is {len(submission)}, expected {EXPECTED_TEST_ROWS}")
    if submission[IDENTIFIER_COL].tolist() != test_df[IDENTIFIER_COL].tolist():
        problems.append("customer_id order does not match test.csv")
    if submission[IDENTIFIER_COL].tolist() != sample_df[IDENTIFIER_COL].tolist():
        problems.append("customer_id order does not match sample_submission.csv")
    if submission[TARGET_COL].isna().any():
        problems.append("submission contains missing predictions")
    invalid = ~submission[TARGET_COL].isin([0, 1])
    if invalid.any():
        problems.append(f"{int(invalid.sum())} predictions are not 0 or 1")
    if problems:
        raise ValueError("; ".join(problems))
    counts = submission[TARGET_COL].value_counts().to_dict()
    return {
        "rows": int(len(submission)),
        "predicted_0": int(counts.get(0, 0)),
        "predicted_1": int(counts.get(1, 0)),
    }


def build_submission(
    model_path: Path = MODEL_PATH,
    submission_path: Path = SUBMISSION_PATH,
) -> dict:
    from src.explain import fit_selected_model

    pipeline, _, _ = fit_selected_model(load_train())
    save_pipeline(pipeline, model_path)
    test_df = load_test()
    sample_df = load_sample_submission()
    submission = predict_submission(pipeline, test_df)
    summary = validate_submission(submission, test_df, sample_df)
    submission.to_csv(submission_path, index=False)
    summary["model_path"] = str(model_path)
    summary["submission_path"] = str(submission_path)
    return summary
