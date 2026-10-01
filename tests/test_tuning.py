"""Tuning helpers. The search itself is recorded outside the unit tests."""

from __future__ import annotations

from src.tuning import SEARCH_SPACES, estimator_from_search


def test_search_spaces_target_the_model_step():
    for space in SEARCH_SPACES.values():
        assert space
        assert all(key.startswith("model__") for key in space)


def test_best_params_are_applied_without_the_pipeline_prefix():
    estimator = estimator_from_search(
        "random_forest",
        {"model__n_estimators": 100, "model__max_depth": 8, "model__min_samples_leaf": 10},
    )
    assert estimator.n_estimators == 100
    assert estimator.max_depth == 8
    assert estimator.min_samples_leaf == 10
