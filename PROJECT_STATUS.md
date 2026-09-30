# Project Status

## Current Phase

2 — Dataset Ingestion & Data Contract

## Status

PASS WITH WARNING

## Last Successful Test

2026-09-30 — `pytest tests/test_data_loader.py` — 8 passed.

## Last Git Commit

Pending Phase 2 commit (this file is included in that commit).

## Known Issues

- GitHub CLI (`gh`) is not installed, so a GitHub remote was not created (Phase 1 warning, still open).
- AWS CLI (`aws`) is not installed. Needed from Phase 20 only.
- `python` is not on PATH; use `py -3.13` or `.venv\Scripts\python.exe`.
- Full `requirements.txt` is not installed; pandas and pytest are installed in `.venv`.

## Data contract notes

- Train 10,147 × 27 and test 2,537 × 26 match the official sizes.
- `passanger` spelling preserved. `car` and `toCoupon_GEQ5min` inspected, not dropped.
- Test has no `Y`. Train/test `customer_id` sets do not overlap.

## Next Action

Phase 3 — Data Quality Assessment. Do not start until explicitly continued.
