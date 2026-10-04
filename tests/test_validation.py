"""Data-quality checks: real official files plus small synthetic cases."""

from __future__ import annotations

import pandas as pd

from src.data_loader import IDENTIFIER_COL, TARGET_COL, load_test, load_train
from tests.csv_requirement import requires_official_csv
from src.validation import (
    assess_quality,
    case_collisions,
    constant_columns,
    feature_duplicate_summary,
    near_constant_columns,
    unexpected_categories,
    whitespace_issues,
)


@requires_official_csv
def test_official_files_have_no_row_or_id_duplicates():
    train = load_train()
    test = load_test()
    report = assess_quality(train, test)
    assert report.duplicate_rows_train == 0
    assert report.duplicate_rows_test == 0
    assert report.duplicate_ids_train == 0
    assert report.duplicate_ids_test == 0


@requires_official_csv
def test_official_target_is_binary_with_no_nulls():
    train = load_train()
    report = assess_quality(train, load_test())
    assert set(report.target_counts) <= {"0", "1"}
    assert sum(report.target_counts.values()) == len(train)
    assert int(train[TARGET_COL].isna().sum()) == 0


@requires_official_csv
def test_official_constant_and_near_constant_columns():
    train = load_train()
    assert "toCoupon_GEQ5min" in constant_columns(train)
    near = near_constant_columns(train)
    assert "car" in near
    assert near["car"] > 0.99


@requires_official_csv
def test_official_no_category_mismatch_or_whitespace():
    report = assess_quality(load_train(), load_test())
    assert report.unexpected_categories == {}
    assert report.whitespace_issues == {}
    assert report.case_collisions == {}
    assert report.test_only_categories == {}
    assert report.direction_are_complements is True


@requires_official_csv
def test_official_feature_duplicate_pairs_are_documented():
    train = load_train()
    test = load_test()
    n_pairs, n_mixed = feature_duplicate_summary(train)
    n_test_pairs, n_test_mixed = feature_duplicate_summary(test)
    assert n_pairs == 63
    assert n_mixed == 15
    assert n_test_pairs == 4
    assert n_test_mixed == 0


@requires_official_csv
def test_official_missing_counts():
    report = assess_quality(load_train(), load_test())
    assert report.missing_train["car"] == 10063
    assert report.missing_test["car"] == 2513
    assert report.missing_train["Bar"] == 88
    assert report.missing_train["CoffeeHouse"] == 172


def test_whitespace_detection_on_synthetic_column():
    df = pd.DataFrame({"city": ["Home", " Home"]})
    assert whitespace_issues(df) == {"city": 1}


def test_case_collision_detection_on_synthetic_column():
    df = pd.DataFrame({"weather": ["Sunny", "sunny"]})
    assert case_collisions(df) == {"weather": ["Sunny/sunny"]}


def test_unexpected_category_detection():
    df = pd.DataFrame({"destination": ["Home", "Airport"]})
    assert unexpected_categories(df) == {"destination": ["Airport"]}


def test_feature_duplicate_mixed_label_on_synthetic():
    df = pd.DataFrame(
        {
            IDENTIFIER_COL: [1, 2],
            "destination": ["Home", "Home"],
            TARGET_COL: [0, 1],
        }
    )
    n_pairs, n_mixed = feature_duplicate_summary(df)
    assert n_pairs == 1
    assert n_mixed == 1
