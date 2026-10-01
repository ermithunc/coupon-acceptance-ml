"""Training-only sklearn preprocessing for coupon acceptance.

Fit this pipeline on training rows or training folds only. Do not fit it
on the official test file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import IDENTIFIER_COL, TARGET_COL

# Nominal fields. Frequency and age stay categorical here. Ordered numeric
# versions are a Phase 6 experiment, not the default encoding.
NOMINAL_COLUMNS = [
    "destination",
    "passanger",
    "weather",
    "time",
    "coupon",
    "expiration",
    "gender",
    "age",
    "maritalStatus",
    "education",
    "occupation",
    "income",
    "car",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

# Continuous-looking value. Binary flags are left unscaled.
SCALED_COLUMNS = ["temperature"]

BINARY_COLUMNS = [
    "has_children",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
]

# Dropped before the transformer. Documented in reports/preprocessing_decisions.md.
DROPPED_COLUMNS = [
    IDENTIFIER_COL,
    "toCoupon_GEQ5min",
    "direction_opp",
]


class NormalizeMissing(BaseEstimator, TransformerMixin):
    """DataFrame cleanup: pandas missing strings become the category `missing`.

    Filling before imputation keeps a level even when a fit slice has no
    observed value, which happens for `car` on small slices.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for column in frame.columns:
            if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(
                frame[column]
            ):
                filled = frame[column].astype("object")
                frame[column] = filled.where(pd.notna(filled), "missing")
        return frame

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray(input_features, dtype=object)


def _nominal_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def _scaled_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def _binary_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )


def make_column_transformer(
    *,
    extra_nominal: list[str] | None = None,
    extra_scaled: list[str] | None = None,
) -> ColumnTransformer:
    """ColumnTransformer over the modeling columns. Identifier and target are absent."""
    nominal = list(NOMINAL_COLUMNS) + list(extra_nominal or [])
    scaled = list(SCALED_COLUMNS) + list(extra_scaled or [])
    binary = list(BINARY_COLUMNS)
    return ColumnTransformer(
        transformers=[
            ("nominal", _nominal_pipeline(), nominal),
            ("scaled", _scaled_pipeline(), scaled),
            ("binary", _binary_pipeline(), binary),
        ],
        remainder="drop",
    )


def make_preprocessing_pipeline(
    *,
    extra_nominal: list[str] | None = None,
    extra_scaled: list[str] | None = None,
) -> Pipeline:
    """Normalize missing values, then impute, encode, and scale."""
    return Pipeline(
        steps=[
            ("normalize", NormalizeMissing()),
            (
                "columns",
                make_column_transformer(
                    extra_nominal=extra_nominal,
                    extra_scaled=extra_scaled,
                ),
            ),
        ]
    )


def modeling_feature_names() -> list[str]:
    """Input columns the default pipeline reads. Excludes the identifier and Y."""
    return list(NOMINAL_COLUMNS) + list(SCALED_COLUMNS) + list(BINARY_COLUMNS)
