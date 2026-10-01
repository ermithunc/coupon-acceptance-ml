# Project Status

## Current Phase

11 — XGBoost / Gradient Boosting

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — stratified 5-fold XGBoost on the training file. CV ROC-AUC 0.8214, accuracy 0.7526.

## Last Git Commit

feat: record xgboost benchmark

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Full `requirements.txt` is not installed. xgboost and shap are now in `.venv`.
- XGBoost leads the untuned comparison. It is not the selected model.

## Next Action

Phase 12 — Model Comparison.
