"""Tests for the shared validation protocol."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_train
from src.experiment import (
    metric_bundle,
    run_experiment,
    stratified_validation_split,
)


def test_metric_bundle_on_a_known_case():
    scores = metric_bundle([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert scores["accuracy"] == 0.75
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == 1.0
    assert 0 < scores["roc_auc"] <= 1
    assert 0 < scores["pr_auc"] <= 1


def test_validation_split_is_stratified_and_disjoint():
    train = load_train()
    X_train, X_valid, y_train, y_valid = stratified_validation_split(train)
    assert len(X_train) + len(X_valid) == len(train)
    assert set(X_train.index).isdisjoint(set(X_valid.index))
    assert y_train.mean() == pytest.approx(y_valid.mean(), abs=0.02)


def test_run_experiment_writes_a_complete_record(tmp_path, monkeypatch):
    def _forbid_test_load(*args, **kwargs):
        raise AssertionError("official test data must not be loaded")

    monkeypatch.setattr("src.data_loader.load_test", _forbid_test_load)
    sample = load_train().sample(n=400, random_state=0)
    payload = run_experiment(
        "smoke_tree",
        DecisionTreeClassifier(max_depth=2, random_state=0),
        sample,
        output_dir=tmp_path / "experiments",
        figures_dir=tmp_path / "figures",
        n_splits=2,
    )
    saved = json.loads((tmp_path / "experiments" / "smoke_tree.json").read_text(encoding="utf-8"))
    assert saved["model"] == "smoke_tree"
    assert saved["n_splits"] == 2
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= saved["cv"][metric]["mean"] <= 1.0
        assert len(saved["cv"][metric]["folds"]) == 2
    matrix = np.array(payload["confusion_matrix"])
    assert matrix.shape == (2, 2)
    assert int(matrix.sum()) == 400
    assert (tmp_path / "figures" / "cm_smoke_tree.png").stat().st_size > 500
    assert (tmp_path / "figures" / "calibration_smoke_tree.png").stat().st_size > 500
    assert (tmp_path / "experiments" / "results.csv").is_file()
