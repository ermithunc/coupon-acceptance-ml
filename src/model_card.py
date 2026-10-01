"""Official performance claim for the saved inference pipeline.

The cross-validation score is the claim. The joblib file is a later refit
on all training rows and is used only to score new rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data_loader import PROJECT_ROOT

EXPERIMENT_PATH = PROJECT_ROOT / "reports" / "experiments" / "xgboost_tuned.json"
MODEL_CARD_PATH = PROJECT_ROOT / "models" / "model_card.json"


def build_model_card() -> dict:
    payload = json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    cv = payload["cv"]
    return {
        "selected_model": "Tuned XGBoost",
        "performance_claim": "5-fold cross-validation on the training file, before the deployment refit",
        "decision_threshold": payload["decision_threshold"],
        "cv": {
            metric: {"mean": cv[metric]["mean"], "std": cv[metric]["std"]}
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
        },
        "oof_brier": payload["oof_brier"],
        "artifact": "models/inference_pipeline.joblib",
        "artifact_fit": "all 10,147 training rows, after the cross-validation score was locked",
        "not_a_performance_claim": {
            "train_accuracy_full_refit": payload["train_accuracy_full_refit"],
            "reason": "Accuracy on the same rows used to fit the saved pipeline. Do not quote it as model performance.",
        },
        "global_associations": [
            "coupon_venue_freq_ord",
            "expiration_1d",
            "coupon_Carry out & Take away",
            "destination_No Urgent Place",
            "weather_Sunny",
        ],
    }


def write_model_card(path: Path = MODEL_CARD_PATH) -> dict:
    card = build_model_card()
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def load_model_card(path: Path = MODEL_CARD_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
