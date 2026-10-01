"""Preprocessing pipeline tests. Fits use training rows only."""

from __future__ import annotations

import numpy as np

from src.data_loader import IDENTIFIER_COL, TARGET_COL, load_train, split_xy
from src.preprocessing import (
    DROPPED_COLUMNS,
    make_preprocessing_pipeline,
    modeling_feature_names,
)


def _xy():
    train = load_train()
    return split_xy(train)


def test_pipeline_drops_identifier_target_and_constant_columns():
    X, y = _xy()
    pipeline = make_preprocessing_pipeline().fit(X, y)
    names = list(pipeline.get_feature_names_out())
    blob = " ".join(names)
    assert TARGET_COL not in X.columns
    assert IDENTIFIER_COL not in modeling_feature_names()
    assert "toCoupon_GEQ5min" in DROPPED_COLUMNS
    assert "direction_opp" in DROPPED_COLUMNS
    for banned in (IDENTIFIER_COL, TARGET_COL, "toCoupon_GEQ5min", "direction_opp"):
        assert banned not in blob
    assert "passanger" in blob


def test_fit_on_train_slice_transforms_later_rows_without_refitting():
    X, y = _xy()
    pipeline = make_preprocessing_pipeline().fit(X.iloc[:8000], y.iloc[:8000])
    transformed = pipeline.transform(X.iloc[8000:])
    assert transformed.shape[0] == len(X) - 8000
    assert np.isfinite(transformed).all()


def test_temperature_scaler_uses_only_fitted_rows():
    X, y = _xy()
    small = make_preprocessing_pipeline().fit(X.iloc[:400], y.iloc[:400])
    large = make_preprocessing_pipeline().fit(X.iloc[:8000], y.iloc[:8000])
    small_mean = small.named_steps["columns"].named_transformers_["scaled"].named_steps["scale"].mean_
    large_mean = large.named_steps["columns"].named_transformers_["scaled"].named_steps["scale"].mean_
    assert small_mean.shape == large_mean.shape
    assert not np.allclose(small_mean, large_mean)


def test_unknown_category_is_ignored():
    X, y = _xy()
    pipeline = make_preprocessing_pipeline().fit(X.iloc[:1000], y.iloc[:1000])
    row = X.iloc[[0]].copy()
    row["coupon"] = "not-a-real-coupon"
    transformed = pipeline.transform(row)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_car_missing_becomes_its_own_level():
    X, y = _xy()
    pipeline = make_preprocessing_pipeline().fit(X, y)
    names = list(pipeline.get_feature_names_out())
    assert any("car_missing" in name or name.endswith("car_missing") for name in names)
