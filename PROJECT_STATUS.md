# Project Status

## Current Phase

13 — Hyperparameter Tuning

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_tuning.py` — 2 passed. Bounded search on the training file, then 5-fold rescore. Tuned XGBoost ROC-AUC 0.8381. Tuned random forest ROC-AUC 0.8201.

## Last Git Commit

feat: tune shortlisted models with a bounded search

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Tuned models fit the training rows more tightly (accuracy gaps 0.1247 and 0.1423). Reported performance is the 5-fold score.
- No final model is selected yet.

## Next Action

Phase 14 — Final Model Selection.
