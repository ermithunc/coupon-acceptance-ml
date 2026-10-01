# Project Status

## Current Phase

5 — Preprocessing Pipeline

## Status

PASS WITH WARNING

## Last Successful Test

2026-10-01 — `pytest tests/test_preprocessing.py` — 5 passed.

## Last Git Commit

feat: add training-only preprocessing pipeline

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created.
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- Full `requirements.txt` is not installed. pandas, pytest, matplotlib, seaborn, and scikit-learn are in `.venv`.
- `car` is 99.17% missing (train). Encoded as its own level, not dropped.
- `toCoupon_GEQ5min` is dropped as a constant. `direction_opp` is dropped as the complement of `direction_same`.

## Next Action

Phase 6 — Feature Engineering. In progress in the same requested batch.
