# Coupon Acceptance ML

Predict whether a customer will accept a recommended restaurant/bar coupon (`Y = 1` accepted, `Y = 0` not accepted).

This repository is built phase by phase from the official hackathon dataset. Training stays local. GitHub is used for source control and CI. AWS is reserved for a lightweight live application.

## Dataset

Official sizes (to be verified in Phase 2 against the actual CSVs):

- Train: 10,147 records
- Test: 2,537 records

Raw files live outside version control:

- `Dataset/Datasets/train.csv`
- `Dataset/Datasets/test.csv`
- `Dataset/Datasets/sample_submission.csv`

## Project structure

```text
coupon-acceptance-ml/
├── data/raw/          # local copies only; not committed
├── data/processed/    # derived artifacts; not committed
├── notebooks/
├── src/
├── models/
├── reports/figures/
├── app/
├── tests/
├── .github/workflows/
├── requirements.txt
├── README.md
└── PROJECT_STATUS.md
```

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Status

See `PROJECT_STATUS.md` for the current phase. Modeling, Streamlit, CI, and AWS work have not started yet.
