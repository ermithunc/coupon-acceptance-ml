"""Submission contract checks. The full refit is run outside the unit tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import EXPECTED_TEST_ROWS, load_sample_submission, load_test
from src.predict import SUBMISSION_PATH, validate_submission


def test_validate_submission_rejects_a_bad_label():
    test_df = pd.DataFrame({"customer_id": [1, 2], "destination": ["Home", "Work"]})
    sample_df = pd.DataFrame({"customer_id": [1, 2], "Y": [0, 0]})
    submission = pd.DataFrame({"customer_id": [1, 2], "Y": [0, 2]})
    with pytest.raises(ValueError, match="not 0 or 1"):
        validate_submission(submission, test_df, sample_df)


def test_validate_submission_rejects_reordered_ids():
    test_df = pd.DataFrame({"customer_id": [1, 2]})
    sample_df = pd.DataFrame({"customer_id": [1, 2], "Y": [0, 1]})
    submission = pd.DataFrame({"customer_id": [2, 1], "Y": [1, 0]})
    with pytest.raises(ValueError, match="test.csv"):
        validate_submission(submission, test_df, sample_df)


def test_saved_submission_matches_the_official_test_ids():
    submission = pd.read_csv(SUBMISSION_PATH)
    summary = validate_submission(submission, load_test(), load_sample_submission())
    assert summary["rows"] == EXPECTED_TEST_ROWS
    assert summary["predicted_0"] + summary["predicted_1"] == EXPECTED_TEST_ROWS
