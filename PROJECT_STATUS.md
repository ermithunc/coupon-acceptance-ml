# Project Status

## Current Phase

17 — Streamlit Application

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_demo_text.py` — 3 passed. Streamlit AppTest predict on the default form returned 84.6% and class Accept.

## Last Git Commit

docs: record the handoff and align raw data with the meta prompt layout

## Known Issues

- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Performance claim is recorded in `models/model_card.json`: 5-fold ROC-AUC 0.8381. The saved pipeline is the deployment refit.

## GitHub

- Remote: `origin` → https://github.com/ermithunc/coupon-acceptance-ml
- Visibility: public
- Branch: `main`

## Next Action

Phase 18 — Inference and Application Testing.
