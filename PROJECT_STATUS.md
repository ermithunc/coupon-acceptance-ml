# Project Status

## Current Phase

4 — Exploratory Data Analysis

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_eda.py tests/test_data_loader.py tests/test_validation.py` — 23 passed.

## Last Git Commit

feat: add training-set EDA charts and findings

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Full `requirements.txt` is not installed. pandas, pytest, matplotlib, and seaborn are in `.venv`.
- `car` is 99.17% missing (train). `toCoupon_GEQ5min` is constant. 15 train feature-duplicate pairs have mixed `Y`. Not cleaned yet.

## Next Action

Phase 5 — Preprocessing Pipeline. Do not start until explicitly continued.
