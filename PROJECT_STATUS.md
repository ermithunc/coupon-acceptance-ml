# Project Status

## Current Phase

19 — GitHub Actions CI

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest` — 59 passed, 0 skipped. Official CSVs are on this machine, so every data check ran.

## Last Git Commit

ci: run pytest on push and pull request

## Known Issues

- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Performance claim is recorded in `models/model_card.json`: 5-fold ROC-AUC 0.8381. The saved pipeline is the deployment refit.
- Unknown categories are ignored by the one-hot encoder and still receive a probability. The Streamlit form only offers training-file categories.
- GitHub Actions cannot see `train.csv` or `test.csv` because those files are gitignored. On run 36851866099 the job passed with 31 tests and skipped 28. Those 28 still run locally.

## GitHub

- Remote: `origin` → https://github.com/ermithunc/coupon-acceptance-ml
- Visibility: public
- Default branch: `main`
- This phase branch: `phase/19-github-actions`

## Next Action

Phase 20 — AWS Deployment Preparation. Do not create paid infrastructure in that phase.
