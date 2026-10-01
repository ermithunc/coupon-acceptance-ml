"""Feature engineering tests. Mappings are fixed and ignore the target."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_train
from tests.csv_requirement import requires_official_csv
from src.feature_engineering import (
    CouponFeatureEngineer,
    compare_feature_sets,
    make_engineered_pipeline,
)


def _row(**overrides):
    base = {
        "destination": "Home",
        "passanger": "Alone",
        "weather": "Sunny",
        "temperature": 80,
        "time": "2PM",
        "coupon": "Coffee House",
        "expiration": "1d",
        "gender": "Female",
        "age": "21",
        "maritalStatus": "Single",
        "has_children": 0,
        "education": "Some college - no degree",
        "occupation": "Student",
        "income": "Less than $12500",
        "car": None,
        "Bar": "never",
        "CoffeeHouse": "1~3",
        "CarryAway": "4~8",
        "RestaurantLessThan20": "less1",
        "Restaurant20To50": "gt8",
        "toCoupon_GEQ5min": 1,
        "toCoupon_GEQ15min": 1,
        "toCoupon_GEQ25min": 0,
        "direction_same": 0,
        "direction_opp": 1,
        "customer_id": 1,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_time_age_income_and_frequency_maps():
    frame = CouponFeatureEngineer().fit_transform(_row())
    assert frame.loc[0, "hour"] == 14
    assert frame.loc[0, "daypart"] == "afternoon"
    assert frame.loc[0, "age_ord"] == 1
    assert frame.loc[0, "income_ord"] == 0
    assert frame.loc[0, "bar_ord"] == 0
    assert frame.loc[0, "coffeehouse_ord"] == 2
    assert frame.loc[0, "restaurant_20to50_ord"] == 4


def test_coupon_frequency_uses_the_matching_venue_only():
    coffee = CouponFeatureEngineer().fit_transform(_row(coupon="Coffee House", CoffeeHouse="never"))
    bar = CouponFeatureEngineer().fit_transform(_row(coupon="Bar", Bar="4~8", CoffeeHouse="never"))
    assert coffee.loc[0, "coupon_venue_freq_ord"] == 0
    assert bar.loc[0, "coupon_venue_freq_ord"] == 3


def test_distance_band_and_direction_label():
    mid = CouponFeatureEngineer().fit_transform(_row(toCoupon_GEQ15min=1, toCoupon_GEQ25min=0))
    far = CouponFeatureEngineer().fit_transform(
        _row(destination="Work", direction_same=1, toCoupon_GEQ25min=1)
    )
    assert mid.loc[0, "distance_band"] == "15_to_25"
    assert far.loc[0, "distance_band"] == "25_plus"
    assert far.loc[0, "destination_direction"] == "Work|same"


def test_missing_frequency_stays_missing():
    frame = CouponFeatureEngineer().fit_transform(_row(Bar=None))
    assert pd.isna(frame.loc[0, "bar_ord"])


def test_target_values_do_not_change_features():
    raw = _row()
    first = CouponFeatureEngineer().fit_transform(raw, y=pd.Series([0]))
    second = CouponFeatureEngineer().fit_transform(raw, y=pd.Series([1]))
    pd.testing.assert_frame_equal(first, second)


@requires_official_csv
def test_engineered_pipeline_transforms_training_rows():
    train = load_train().iloc[:300]
    X = train.drop(columns=["Y"])
    y = train["Y"]
    transformed = make_engineered_pipeline().fit(X, y).transform(X.iloc[:5])
    assert transformed.shape[0] == 5
    assert transformed.shape[1] > 10


@requires_official_csv
def test_feature_comparison_runs_on_a_training_sample():
    train = load_train().sample(n=1200, random_state=0)
    table = compare_feature_sets(train, n_splits=2)
    assert set(table["feature_set"]) == {"baseline", "engineered"}
    assert table["roc_auc_mean"].between(0, 1).all()
    assert table["accuracy_mean"].between(0, 1).all()
