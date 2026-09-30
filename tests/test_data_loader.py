"""Contract tests for official CSV ingestion. Uses real files only."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    IDENTIFIER_COL,
    TARGET_COL,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
    DataContractError,
    feature_columns,
    load_sample_submission,
    load_test,
    load_train,
    split_xy,
)


def test_train_contract_matches_official_files():
    train = load_train()
    assert list(train.columns) == TRAIN_COLUMNS
    assert len(train) == EXPECTED_TRAIN_ROWS
    assert train[IDENTIFIER_COL].is_unique
    assert set(train[TARGET_COL].dropna().unique()) <= {0, 1}


def test_test_file_has_no_target_and_expected_shape():
    test = load_test()
    assert list(test.columns) == TEST_COLUMNS
    assert len(test) == EXPECTED_TEST_ROWS
    assert TARGET_COL not in test.columns
    assert test[IDENTIFIER_COL].is_unique


def test_train_and_test_identifiers_do_not_overlap():
    train = load_train()
    test = load_test()
    overlap = set(train[IDENTIFIER_COL]).intersection(set(test[IDENTIFIER_COL]))
    assert overlap == set()


def test_sample_submission_aligns_with_test_ids():
    test = load_test()
    submission = load_sample_submission()
    assert list(submission.columns) == [IDENTIFIER_COL, TARGET_COL]
    assert len(submission) == EXPECTED_TEST_ROWS
    pd.testing.assert_series_equal(
        submission[IDENTIFIER_COL].reset_index(drop=True),
        test[IDENTIFIER_COL].reset_index(drop=True),
        check_names=False,
    )


def test_split_xy_does_not_include_target_in_features():
    train = load_train()
    X, y = split_xy(train)
    assert TARGET_COL not in X.columns
    assert len(X) == len(y) == len(train)
    assert IDENTIFIER_COL in X.columns


def test_feature_columns_exclude_identifier_by_default():
    cols = feature_columns(include_identifier=False)
    assert IDENTIFIER_COL not in cols
    assert TARGET_COL not in cols
    assert "passanger" in cols


def test_passanger_spelling_is_preserved():
    train = load_train()
    test = load_test()
    assert "passanger" in train.columns
    assert "passenger" not in train.columns
    assert "passanger" in test.columns


def test_contract_rejects_wrong_columns(tmp_path, monkeypatch):
    from src import data_loader as dl

    fake_dir = tmp_path / "raw"
    fake_dir.mkdir()
    pd.DataFrame({"wrong": [1]}).to_csv(fake_dir / "train.csv", index=False)
    pd.DataFrame({"wrong": [1]}).to_csv(fake_dir / "test.csv", index=False)
    monkeypatch.setattr(dl, "RAW_DIR_CANDIDATES", (fake_dir,))
    with pytest.raises(DataContractError):
        dl.load_train()
