"""Coupon acceptance demo.

Loads the saved pipeline. It does not fit a model and does not reimplement
feature engineering or preprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.demo_text import build_raw_row, claim_sentence, classify, prediction_sentence, profile_sentence
from src.predict import load_pipeline
from src.validation import KNOWN_CATEGORIES

AGE_ORDER = ["below21", "21", "26", "31", "36", "41", "46", "50plus"]
TIME_ORDER = ["7AM", "10AM", "2PM", "6PM", "10PM"]
INCOME_ORDER = [
    "Less than $12500",
    "$12500 - $24999",
    "$25000 - $37499",
    "$37500 - $49999",
    "$50000 - $62499",
    "$62500 - $74999",
    "$75000 - $87499",
    "$87500 - $99999",
    "$100000 or More",
]
EDUCATION_ORDER = [
    "Some High School",
    "High School Graduate",
    "Some college - no degree",
    "Associates degree",
    "Bachelors degree",
    "Graduate degree (Masters or Doctorate)",
]
FREQ_ORDER = ["never", "less1", "1~3", "4~8", "gt8"]


def _ordered(name: str, order: list[str] | None = None) -> list[str]:
    values = set(KNOWN_CATEGORIES[name])
    if order is None:
        return sorted(values)
    return [item for item in order if item in values]


@st.cache_resource
def get_pipeline():
    return load_pipeline()


def _index(options: list, value) -> int:
    return options.index(value)


def main() -> None:
    st.set_page_config(page_title="Coupon acceptance", layout="wide")
    st.title("Will this customer accept the coupon?")
    st.write(claim_sentence())

    with st.form("coupon_form"):
        left, right = st.columns(2)
        destinations = _ordered("destination")
        passangers = _ordered("passanger")
        ages = _ordered("age", AGE_ORDER)
        coupons = _ordered("coupon")
        genders = _ordered("gender")
        statuses = _ordered("maritalStatus")
        occupations = _ordered("occupation")
        with left:
            destination = st.selectbox("Destination", destinations, index=_index(destinations, "No Urgent Place"))
            passanger = st.selectbox("Passanger", passangers, index=_index(passangers, "Friend(s)"))
            weather = st.selectbox("Weather", _ordered("weather", ["Sunny", "Rainy", "Snowy"]))
            temperature = st.selectbox("Temperature", [30, 55, 80], index=2)
            time = st.selectbox("Time", _ordered("time", TIME_ORDER), index=3)
            coupon = st.selectbox("Coupon", coupons, index=_index(coupons, "Coffee House"))
            expiration = st.selectbox("Expiration", _ordered("expiration", ["2h", "1d"]), index=1)
            gender = st.selectbox("Gender", genders, index=_index(genders, "Male"))
            age = st.selectbox("Age", ages, index=_index(ages, "21"))
        with right:
            marital = st.selectbox("Marital status", statuses, index=_index(statuses, "Single"))
            has_children = st.selectbox("Has children", [0, 1])
            education = st.selectbox("Education", _ordered("education", EDUCATION_ORDER), index=2)
            occupation = st.selectbox("Occupation", occupations, index=_index(occupations, "Student"))
            income = st.selectbox("Income", _ordered("income", INCOME_ORDER))
            car = st.selectbox("Car", ["Not reported"] + _ordered("car"))
            bar = st.selectbox("Bar visits", _ordered("Bar", FREQ_ORDER))
            coffee = st.selectbox("Coffee house visits", _ordered("CoffeeHouse", FREQ_ORDER), index=2)
            carry = st.selectbox("Carry-out visits", _ordered("CarryAway", FREQ_ORDER), index=2)
            rest_low = st.selectbox("Restaurant under $20 visits", _ordered("RestaurantLessThan20", FREQ_ORDER), index=2)
            rest_high = st.selectbox("Restaurant $20–50 visits", _ordered("Restaurant20To50", FREQ_ORDER), index=1)
            geq15 = st.selectbox("Drive is at least 15 minutes", [0, 1])
            geq25 = st.selectbox("Drive is at least 25 minutes", [0, 1])
            direction_same = st.selectbox("Same direction as the venue", [0, 1])
        submitted = st.form_submit_button("Predict acceptance")

    if not submitted:
        st.info("Choose the trip and the coupon, then predict. The form uses the categories from the training file.")
        return

    row = build_raw_row(
        {
            "destination": destination,
            "passanger": passanger,
            "weather": weather,
            "temperature": int(temperature),
            "time": time,
            "coupon": coupon,
            "expiration": expiration,
            "gender": gender,
            "age": age,
            "maritalStatus": marital,
            "has_children": int(has_children),
            "education": education,
            "occupation": occupation,
            "income": income,
            "car": car,
            "Bar": bar,
            "CoffeeHouse": coffee,
            "CarryAway": carry,
            "RestaurantLessThan20": rest_low,
            "Restaurant20To50": rest_high,
            "toCoupon_GEQ15min": int(geq15),
            "toCoupon_GEQ25min": int(geq25),
            "direction_same": int(direction_same),
        }
    )
    probability = float(get_pipeline().predict_proba(row)[0, 1])
    label = classify(probability)

    score, decision = st.columns(2)
    score.metric("Acceptance probability", f"{probability:.1%}")
    decision.metric("Predicted class", "Accept" if label == 1 else "Not accept")
    st.subheader("Who this is")
    st.write(profile_sentence(row.iloc[0]))
    st.write(prediction_sentence(probability, label))

    st.subheader("Where the number comes from")
    st.write(
        "This row goes through the saved pipeline once: the same feature engineering and preprocessing used in training, then tuned XGBoost. "
        "Logistic regression, the decision tree, and random forest were comparison models. Their validation scores are below. They do not each score this row."
    )
    comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    show = comparison[
        [
            "model",
            "accuracy_mean",
            "f1_mean",
            "roc_auc_mean",
            "pr_auc_mean",
        ]
    ].copy()
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Untuned 5-fold means on the training file. Tuned XGBoost, the selected model, has cross-validation ROC-AUC 0.8381.")

    st.subheader("Project path")
    st.write(
        "Real training data, then data quality, EDA, and engineered features. "
        "Logistic regression, a decision tree, random forest, and XGBoost were compared. "
        "Bounded tuning selected tuned XGBoost from the cross-validation scores. "
        "SHAP describes associations with the predicted probability. "
        "GitHub hosting and the AWS app are the remaining deployment steps."
    )


if __name__ == "__main__":
    main()
