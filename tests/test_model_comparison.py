"""Comparison table tests. Uses the saved untuned records."""

from __future__ import annotations

import pandas as pd

from src.model_comparison import load_untuned_records, shortlist_models


def test_untuned_table_contains_the_four_models():
    table = load_untuned_records()
    assert set(table["model"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
    assert table.iloc[0]["model"] == "xgboost"
    assert table["roc_auc_mean"].is_monotonic_decreasing


def test_shortlist_keeps_a_close_runner_up():
    table = pd.DataFrame(
        {
            "model": ["xgboost", "random_forest", "logistic_regression"],
            "roc_auc_mean": [0.82, 0.81, 0.75],
        }
    )
    assert shortlist_models(table) == ["xgboost", "random_forest"]


def test_shortlist_drops_a_distant_runner_up():
    table = pd.DataFrame(
        {
            "model": ["xgboost", "logistic_regression"],
            "roc_auc_mean": [0.82, 0.76],
        }
    )
    assert shortlist_models(table) == ["xgboost"]
