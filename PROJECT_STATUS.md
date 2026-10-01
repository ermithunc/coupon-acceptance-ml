# Project Status

## Current Phase

14 — Final Model Selection

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — selection checked against `reports/experiments/` 5-fold records. Selected model: tuned XGBoost, ROC-AUC 0.8381, PR-AUC 0.8596, Brier 0.1606.

## Last Git Commit

docs: select tuned xgboost from validation evidence

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Reported performance is the 5-fold score. The 0.8900 full-refit training accuracy is not the claim.
- SHAP explanations are not written yet.

## Next Action

Phase 15 — SHAP / Explainability.
