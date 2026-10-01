# Project Status

## Current Phase

17 — Streamlit Application

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_demo_text.py` — 3 passed. Streamlit AppTest predict on the default form returned 84.6% and class Accept.

## Last Git Commit

feat: add coupon acceptance streamlit demo

## Known Issues

- GitHub CLI 2.102.0 is installed, but `gh auth login` has not been run, so no GitHub remote exists yet.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Performance claim is recorded in `models/model_card.json`: 5-fold ROC-AUC 0.8381. The saved pipeline is the deployment refit.

## Next Action

Phase 18 — Inference and Application Testing. Before that, run `gh auth login` in a terminal if you want the GitHub remote created.
