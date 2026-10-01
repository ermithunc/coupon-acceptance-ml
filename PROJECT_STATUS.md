# Project Status

## Current Phase

15 — SHAP / Explainability

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_explain.py` — 1 passed. SHAP sample of 400 training rows on tuned XGBoost.

## Last Git Commit

feat: add SHAP explanations for tuned xgboost

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Reported model performance remains the 5-fold score (ROC-AUC 0.8381), not the 0.8900 training refit accuracy.
- SHAP describes associations with predicted probability. One local example is dominated by a single occupation flag; the global pattern is coupon-venue frequency and a 1-day expiration.

## Next Action

Phase 16 — Final Model + Test Prediction. Do not start until explicitly continued.
