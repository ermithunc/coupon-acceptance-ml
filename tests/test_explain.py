"""Checks for explanation wording. The full SHAP run is recorded separately."""

from __future__ import annotations

import numpy as np

from src.explain import _local_lines


def test_local_lines_use_association_language():
    lines = _local_lines(
        np.array([0.2, -0.1]),
        ["coupon_Bar", "hour"],
        np.array([1.0, 18.0]),
    )
    text = " ".join(lines)
    assert "associated with a higher" in text
    assert "associated with a lower" in text
    assert "cause" not in text.lower()
