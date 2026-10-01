# Project Status

## Current Phase

12 — Model Comparison

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_model_comparison.py` — 3 passed.

## Last Git Commit

docs: compare untuned models on shared validation metrics

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Untuned XGBoost leads ROC-AUC, PR-AUC, F1, accuracy, and Brier. Random forest leads recall and is within 0.02 ROC-AUC, so both are shortlisted. Neither is the final model yet.

## Next Action

Phase 13 — Hyperparameter Tuning of the shortlist.
