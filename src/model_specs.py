"""Estimator settings for the shared experiment protocol.

Each model uses these initial settings. Tuning belongs to a later phase.
"""

from __future__ import annotations

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
