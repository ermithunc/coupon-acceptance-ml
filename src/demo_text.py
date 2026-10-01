"""Plain-language profile for one raw input row.

This does not transform features and does not score the model.
"""

from __future__ import annotations

import pandas as pd

from src.model_card import load_model_card
from src.predict import DECISION_THRESHOLD


def profile_sentence(row: pd.Series) -> str:
    age = str(row["age"])
    if age == "below21":
        age_text = "younger than 21"
    elif age == "50plus":
        age_text = "50 or older"
    else:
        age_text = f"{age} years old"
    return (
        f"This person is {age_text}, {str(row['gender']).lower()}, "
        f"going to {row['destination']} with {row['passanger']}. "
        f"The coupon is {row['coupon']} at {row['time']}, weather is {row['weather']}, "
        f"and it expires in {row['expiration']}."
    )


def prediction_sentence(probability: float, label: int) -> str:
    decision = "accept" if label == 1 else "not accept"
    return (
        f"The saved tuned XGBoost pipeline estimates a {probability:.0%} chance of acceptance, "
        f"so the predicted class is {decision}."
    )


def claim_sentence() -> str:
    card = load_model_card()
    roc = card["cv"]["roc_auc"]["mean"]
    return (
        f"The performance claim is a 5-fold cross-validation ROC-AUC of {roc:.4f} "
        "on the training file. The saved file was refit on all training rows after that "
        "score was locked, and that refit accuracy is not the performance claim."
    )


def build_raw_row(values: dict) -> pd.DataFrame:
    """One raw row with the columns the inference pipeline expects."""
    row = dict(values)
    row["customer_id"] = 0
    row["toCoupon_GEQ5min"] = 1
    same = int(row["direction_same"])
    row["direction_opp"] = 0 if same == 1 else 1
    if row.get("car") in (None, "", "Not reported"):
        row["car"] = None
    return pd.DataFrame([row])


def classify(probability: float, threshold: float = DECISION_THRESHOLD) -> int:
    return int(probability >= threshold)
