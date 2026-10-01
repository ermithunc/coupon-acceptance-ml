# Project Status

## Current Phase

20 — AWS Deployment Preparation

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_deploy_prep.py` — 3 passed. No AWS API call was made.

## Last Git Commit

docs: prepare lightweight hosting without creating AWS resources

## Known Issues

- AWS CLI (`aws`) is not installed, so `aws sts get-caller-identity` could not run. Phase 21 stays blocked until the CLI is installed and the account is authenticated locally.
- Performance claim is recorded in `models/model_card.json`: 5-fold ROC-AUC 0.8381. The saved pipeline is the deployment refit.
- Unknown categories are ignored by the one-hot encoder and still receive a probability. The Streamlit form only offers training-file categories.
- GitHub Actions cannot see `train.csv` or `test.csv` because those files are gitignored. On run 36851866099 the job passed with 31 tests and skipped 28. Those 28 still run locally.

## GitHub

- Remote: `origin` → https://github.com/ermithunc/coupon-acceptance-ml
- Visibility: public
- Default branch: `main`
- This phase branch: `phase/20-aws-preparation`

## Next Action

Phase 21 — AWS Deployment. Blocked until the AWS CLI is installed and `aws sts get-caller-identity` succeeds. Do not create paid resources before that.
