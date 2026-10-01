# Project Status

## Current Phase

18 — Inference and Application Testing

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_inference.py` — 7 passed. The default Streamlit form scored 84.6% and class Accept, matching the saved pipeline.

## Last Git Commit

test: check saved-pipeline inference and the streamlit form

## Known Issues

- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Performance claim is recorded in `models/model_card.json`: 5-fold ROC-AUC 0.8381. The saved pipeline is the deployment refit.
- Unknown categories are ignored by the one-hot encoder and still receive a probability. The Streamlit form only offers training-file categories.

## GitHub

- Remote: `origin` → https://github.com/ermithunc/coupon-acceptance-ml
- Visibility: public
- Default branch: `main`
- This phase branch: `phase/18-application-testing`

## Next Action

Phase 19 — GitHub Actions CI.
