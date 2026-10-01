"""Phase 18 checks for the saved inference pipeline.

These tests load models/inference_pipeline.joblib. They do not refit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT, TEST_COLUMNS, load_test
from src.demo_text import build_raw_row, classify
from src.predict import DECISION_THRESHOLD, SUBMISSION_PATH, load_pipeline
from src.validation import KNOWN_CATEGORIES


def _pipeline():
    return load_pipeline()


def _demo_row() -> pd.DataFrame:
    """Same defaults as the Streamlit form."""
    return build_raw_row(
        {
            "destination": "No Urgent Place",
            "passanger": "Friend(s)",
            "weather": "Sunny",
            "temperature": 80,
            "time": "6PM",
            "coupon": "Coffee House",
            "expiration": "1d",
            "gender": "Male",
            "age": "21",
            "maritalStatus": "Single",
            "has_children": 0,
            "education": "Some college - no degree",
            "occupation": "Student",
            "income": "Less than $12500",
            "car": "Not reported",
            "Bar": "never",
            "CoffeeHouse": "1~3",
            "CarryAway": "1~3",
            "RestaurantLessThan20": "1~3",
            "Restaurant20To50": "less1",
            "toCoupon_GEQ15min": 0,
            "toCoupon_GEQ25min": 0,
            "direction_same": 0,
        }
    )


def test_model_loads_the_saved_xgboost_pipeline():
    pipeline = _pipeline()
    assert pipeline.named_steps["model"].__class__.__name__ == "XGBClassifier"
    assert callable(pipeline.predict_proba)


def test_feature_schema_matches_the_official_test_columns():
    pipeline = _pipeline()
    assert list(pipeline.feature_names_in_) == TEST_COLUMNS
    assert "Y" not in pipeline.feature_names_in_
    assert "passanger" in pipeline.feature_names_in_


def test_single_row_inference_returns_a_probability_and_class():
    proba = _pipeline().predict_proba(_demo_row())
    assert proba.shape == (1, 2)
    assert np.issubdtype(proba.dtype, np.floating)
    acceptance = float(proba[0, 1])
    label = classify(acceptance)
    assert isinstance(label, int)
    assert label in (0, 1)
    assert 0.0 <= acceptance <= 1.0
    assert label == int(acceptance >= DECISION_THRESHOLD)


def test_batch_inference_matches_the_saved_submission():
    test_df = load_test()
    proba = _pipeline().predict_proba(test_df)
    assert proba.shape == (len(test_df), 2)
    assert np.isfinite(proba).all()
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    labels = (proba[:, 1] >= DECISION_THRESHOLD).astype(int)
    submission = pd.read_csv(SUBMISSION_PATH)
    assert labels.tolist() == submission["Y"].astype(int).tolist()


def test_missing_values_still_return_a_probability_in_range():
    row = load_test().iloc[[0]].copy()
    row["car"] = None
    row["Bar"] = np.nan
    row["CoffeeHouse"] = np.nan
    acceptance = float(_pipeline().predict_proba(row)[0, 1])
    assert 0.0 <= acceptance <= 1.0
    assert classify(acceptance) in (0, 1)


def test_unknown_categories_still_return_a_probability_in_range():
    row = load_test().iloc[[0]].copy()
    row["coupon"] = "Not A Real Coupon"
    row["destination"] = "Mars"
    assert row["coupon"].iloc[0] not in KNOWN_CATEGORIES["coupon"]
    assert row["destination"].iloc[0] not in KNOWN_CATEGORIES["destination"]
    acceptance = float(_pipeline().predict_proba(row)[0, 1])
    assert 0.0 <= acceptance <= 1.0


def test_streamlit_smoke_scores_the_default_form():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=90)
    app.run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    metric_values = [metric.value for metric in app.metric]
    assert any(value in ("Accept", "Not accept") for value in metric_values)
    assert any("%" in str(value) for value in metric_values)
    direct = float(_pipeline().predict_proba(_demo_row())[0, 1])
    assert f"{direct:.1%}" in metric_values
