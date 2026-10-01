"""EDA helper tests. Figure generation uses the official training file only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import TARGET_COL, load_train
from src.eda import (
    acceptance_table,
    car_observed_table,
    generate_figures,
    overall_acceptance_rate,
)


def test_acceptance_rate_matches_counts():
    frame = pd.DataFrame(
        {"coupon": ["Bar", "Bar", "Coffee House"], TARGET_COL: [1, 0, 1]}
    )
    table = acceptance_table(frame, "coupon").set_index("level")
    assert table.loc["Bar", "n"] == 2
    assert table.loc["Bar", "acceptance_rate"] == 0.5
    assert table.loc["Coffee House", "acceptance_rate"] == 1.0


def test_missing_levels_are_kept():
    frame = pd.DataFrame({"Bar": [None, "never"], TARGET_COL: [1, 0]})
    levels = set(acceptance_table(frame, "Bar")["level"])
    assert levels == {"(missing)", "never"}


def test_training_overall_rate_matches_official_counts():
    train = load_train()
    assert len(train) == 10147
    assert abs(overall_acceptance_rate(train) - (5768 / 10147)) < 1e-12


def test_car_observed_split_counts():
    table = car_observed_table(load_train()).set_index("level")
    assert int(table.loc["missing", "n"]) == 10063
    assert int(table.loc["observed", "n"]) == 84


def test_generate_figures_writes_nonempty_pngs(tmp_path: Path):
    paths = generate_figures(load_train(), tmp_path)
    assert len(paths) == 10
    for path in paths:
        assert path.suffix == ".png"
        assert path.stat().st_size > 1000
        assert path.parent == tmp_path
