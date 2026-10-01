# Project Status

## Current Phase

6 — Feature Engineering

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_feature_engineering.py tests/test_preprocessing.py` — 12 passed. Feature-set probe: stratified 5-fold logistic regression on the full training file.

## Last Git Commit

feat: add coupon features and compare them to the baseline

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Full `requirements.txt` is not installed. pandas, pytest, matplotlib, seaborn, and scikit-learn are in `.venv`.
- `car` stays a categorical level `missing`. `toCoupon_GEQ5min` and `direction_opp` stay out of the model matrix.
- Engineered features improved a logistic-regression probe (ROC-AUC 0.7590 vs 0.7360). That is not a final model choice.

## Next Action

Phase 7 — Validation / Experiment Framework. Do not start until explicitly continued.
