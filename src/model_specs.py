"""Estimator settings for the shared experiment protocol.

Each model uses these initial settings. Tuning belongs to a later phase.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def logistic_regression() -> LogisticRegression:
    """Transparent linear baseline."""
    return LogisticRegression(max_iter=2000, random_state=42)


def decision_tree() -> DecisionTreeClassifier:
    """Initial tree. Depth and leaf size limit overfitting before any search."""
    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        min_samples_split=100,
        random_state=42,
    )


def random_forest() -> RandomForestClassifier:
    """Initial forest. More trees than one tree, without a large search."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def xgboost_model():
    """Initial boosted trees. Small fixed settings, not a search."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
