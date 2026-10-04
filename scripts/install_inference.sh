#!/bin/bash
# Install the CPU inference stack from requirements-prod.txt.
# XGBoost 3.4.1 declares nvidia-nccl-cu13 on Linux. That library is used for
# GPU collectives. This host runs tree_method=hist and does not load it.
# The application start command stays:
#   python -m streamlit run app/app.py
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python3.13 -m pip install --upgrade pip

filtered="$(mktemp)"
trap 'rm -f "$filtered"' EXIT
grep -E -v '^[[:space:]]*(#|$|xgboost==)' requirements-prod.txt > "$filtered"
python3.13 -m pip install -r "$filtered"

xgb="$(grep -E '^xgboost==[0-9]+\.[0-9]+\.[0-9]+[[:space:]]*$' requirements-prod.txt)"
python3.13 -m pip install --no-deps "${xgb}"
