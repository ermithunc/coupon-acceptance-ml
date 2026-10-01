"""Tests for the performance claim and the demo profile text."""

from __future__ import annotations

import json

from src.data_loader import PROJECT_ROOT
from src.demo_text import build_raw_row, claim_sentence, profile_sentence
from src.model_card import build_model_card


def test_model_card_matches_the_saved_experiment():
    card = build_model_card()
    payload = json.loads(
        (PROJECT_ROOT / "reports" / "experiments" / "xgboost_tuned.json").read_text(encoding="utf-8")
    )
    assert card["cv"]["roc_auc"]["mean"] == payload["cv"]["roc_auc"]["mean"]
    assert card["not_a_performance_claim"]["train_accuracy_full_refit"] == payload["train_accuracy_full_refit"]
    assert "5-fold" in card["performance_claim"]


def test_claim_sentence_quotes_cross_validation_not_the_refit():
    text = claim_sentence()
    assert "0.8381" in text
    assert "not the performance claim" in text


def test_profile_mentions_the_entered_age_and_coupon():
    row = build_raw_row(
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
    ).iloc[0]
    text = profile_sentence(row)
    assert "21 years old" in text
    assert "Coffee House" in text
    assert row["direction_opp"] == 1
    assert row["toCoupon_GEQ5min"] == 1
