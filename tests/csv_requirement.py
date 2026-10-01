"""Skip official-file checks when the CSVs are not in the checkout.

The raw files are gitignored. A local run sees them. GitHub Actions does not.
"""

from __future__ import annotations

import pytest

from src.data_loader import official_files_present

requires_official_csv = pytest.mark.skipif(
    not official_files_present(),
    reason="Official train.csv and test.csv stay on the local machine and are not in the GitHub checkout.",
)
