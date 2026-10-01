"""Row-level features for coupon acceptance.

Mappings are fixed from the meaning of the labels. Nothing is learned from
Y or from the test file. customer_id is unique per row, so there is no
customer history to turn into an affinity score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    NormalizeMissing,
    make_column_transformer,
    make_preprocessing_pipeline,
)

TIME_TO_HOUR = {"7AM": 7, "10AM": 10, "2PM": 14, "6PM": 18, "10PM": 22}
TIME_TO_DAYPART = {
    "7AM": "morning",
    "10AM": "morning",
    "2PM": "afternoon",
    "6PM": "evening",
    "10PM": "night",
}
AGE_ORDINAL = {
    "below21": 0,
    "21": 1,
    "26": 2,
    "31": 3,
    "36": 4,
    "41": 5,
    "46": 6,
    "50plus": 7,
}
INCOME_ORDINAL = {
    "Less than $12500": 0,
    "$12500 - $24999": 1,
    "$25000 - $37499": 2,
    "$37500 - $49999": 3,
    "$50000 - $62499": 4,
    "$62500 - $74999": 5,
    "$75000 - $87499": 6,
    "$87500 - $99999": 7,
    "$100000 or More": 8,
}
FREQUENCY_ORDINAL = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
FREQUENCY_SOURCE = {
    "bar_ord": "Bar",
    "coffeehouse_ord": "CoffeeHouse",
    "carryaway_ord": "CarryAway",
    "restaurant_lt20_ord": "RestaurantLessThan20",
    "restaurant_20to50_ord": "Restaurant20To50",
}
COUPON_TO_FREQUENCY = {
    "Bar": "Bar",
    "Coffee House": "CoffeeHouse",
    "Carry out & Take away": "CarryAway",
    "Restaurant(<20)": "RestaurantLessThan20",
    "Restaurant(20-50)": "Restaurant20To50",
}

ENGINEERED_NOMINAL = ["daypart", "distance_band", "destination_direction"]
ENGINEERED_SCALED = [
    "hour",
    "age_ord",
    "income_ord",
    "bar_ord",
    "coffeehouse_ord",
    "carryaway_ord",
    "restaurant_lt20_ord",
    "restaurant_20to50_ord",
    "coupon_venue_freq_ord",
]


def _ordinal(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).astype("float64")


class CouponFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add documented columns. fit does not look at y."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        extra = [name for name in ENGINEERED_NOMINAL + ENGINEERED_SCALED if name not in frame.columns]
        self.feature_names_out_ = np.asarray(list(frame.columns) + extra, dtype=object)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame["hour"] = _ordinal(frame["time"], TIME_TO_HOUR)
        frame["daypart"] = frame["time"].map(TIME_TO_DAYPART)
        frame["age_ord"] = _ordinal(frame["age"], AGE_ORDINAL)
        frame["income_ord"] = _ordinal(frame["income"], INCOME_ORDINAL)
        for new_name, source in FREQUENCY_SOURCE.items():
            frame[new_name] = _ordinal(frame[source], FREQUENCY_ORDINAL)
        frame["coupon_venue_freq_ord"] = _coupon_venue_frequency(frame)
        frame["distance_band"] = _distance_band(frame)
        frame["destination_direction"] = _destination_direction(frame)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def _coupon_venue_frequency(frame: pd.DataFrame) -> pd.Series:
    """Ordinal visit frequency of the venue family named by the coupon."""
    scores = pd.Series(np.nan, index=frame.index, dtype="float64")
    for coupon, source in COUPON_TO_FREQUENCY.items():
        mask = frame["coupon"].eq(coupon)
        scores.loc[mask] = _ordinal(frame.loc[mask, source], FREQUENCY_ORDINAL).to_numpy()
    return scores


def _distance_band(frame: pd.DataFrame) -> pd.Series:
    geq25 = frame["toCoupon_GEQ25min"].astype("float64").eq(1)
    geq15 = frame["toCoupon_GEQ15min"].astype("float64").eq(1)
    band = pd.Series("5_to_15", index=frame.index, dtype="object")
    band.loc[geq15] = "15_to_25"
    band.loc[geq25] = "25_plus"
    return band


def _destination_direction(frame: pd.DataFrame) -> pd.Series:
    destination = frame["destination"].astype("object").where(pd.notna(frame["destination"]), "missing")
    same = frame["direction_same"].astype("float64").eq(1)
    side = pd.Series(np.where(same, "same", "opposite"), index=frame.index)
    return destination.astype(str) + "|" + side


def make_engineered_pipeline() -> Pipeline:
    """Feature step, missing-value cleanup, then the shared column transformer."""
    return Pipeline(
        steps=[
            ("engineer", CouponFeatureEngineer()),
            ("normalize", NormalizeMissing()),
            (
                "columns",
                make_column_transformer(
                    extra_nominal=ENGINEERED_NOMINAL,
                    extra_scaled=ENGINEERED_SCALED,
                ),
            ),
        ]
    )


def compare_feature_sets(train_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Probe logistic regression on training folds only.

    This compares feature sets. It is not the Phase 8 model record.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate

    from src.data_loader import split_xy

    X, y = split_xy(train_df)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    pipelines = {
        "baseline": Pipeline(
            steps=[
                ("prep", make_preprocessing_pipeline()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "engineered": Pipeline(
            steps=[
                ("prep", make_engineered_pipeline()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
    }
    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "roc_auc"],
            n_jobs=1,
        )
        rows.append(
            {
                "feature_set": name,
                "n_splits": n_splits,
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std(ddof=0)),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "roc_auc_std": float(scores["test_roc_auc"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)
