"""Bounded hyperparameter search for the shortlisted models.

The search uses training rows only. Official test data is not read.
After the search, the best settings are scored again with the shared 5-fold protocol.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data_loader import PROJECT_ROOT, split_xy
from src.experiment import run_experiment
from src.experiment import make_model_pipeline
from src.model_comparison import load_untuned_records, shortlist_models
from src.model_specs import random_forest, xgboost_model

TUNING_DIR = PROJECT_ROOT / "reports" / "tuning"
N_ITER = 8
SEARCH_SPLITS = 3
SEARCH_SCORING = "roc_auc"

SEARCH_SPACES = {
    "xgboost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__min_child_weight": [1, 5],
        "model__subsample": [0.8, 1.0],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [8, 12, 16, None],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", 0.5],
    },
}

FACTORIES = {
    "xgboost": xgboost_model,
    "random_forest": random_forest,
}


def search_one(name: str, train_df: pd.DataFrame) -> dict:
    X, y = split_xy(train_df)
    pipeline = make_model_pipeline(FACTORIES[name]())
    cv = StratifiedKFold(n_splits=SEARCH_SPLITS, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=SEARCH_SPACES[name],
        n_iter=N_ITER,
        scoring=SEARCH_SCORING,
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=False,
    )
    started = time.perf_counter()
    search.fit(X, y)
    runtime = time.perf_counter() - started
    record = {
        "model": name,
        "n_iter": N_ITER,
        "n_splits": SEARCH_SPLITS,
        "scoring": SEARCH_SCORING,
        "search_space": SEARCH_SPACES[name],
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "runtime_seconds": round(runtime, 1),
    }
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    (TUNING_DIR / f"{name}_search.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
    return record


def estimator_from_search(name: str, best_params: dict):
    estimator = FACTORIES[name]()
    cleaned = {key.replace("model__", ""): value for key, value in best_params.items()}
    estimator.set_params(**cleaned)
    return estimator


def tuned_shortlist(train_df: pd.DataFrame) -> list[str]:
    return shortlist_models(load_untuned_records())


def run_pending(train_df: pd.DataFrame) -> None:
    """Search each shortlisted model once, then score the best settings with 5 folds."""
    for name in tuned_shortlist(train_df):
        search_path = TUNING_DIR / f"{name}_search.json"
        if search_path.is_file():
            record = json.loads(search_path.read_text(encoding="utf-8"))
        else:
            record = search_one(name, train_df)
        experiment_path = PROJECT_ROOT / "reports" / "experiments" / f"{name}_tuned.json"
        if not experiment_path.is_file():
            run_experiment(
                f"{name}_tuned",
                estimator_from_search(name, record["best_params"]),
                train_df,
            )
        print(name, "ready", round(record["best_cv_roc_auc"], 4), record["runtime_seconds"])
