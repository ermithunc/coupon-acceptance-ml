"""Training-set exploratory summaries and charts.

Descriptive only. Does not fit models and does not read the test file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import PROJECT_ROOT, TARGET_COL

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FREQ_ORDER = ["never", "less1", "1~3", "4~8", "gt8", "(missing)"]
AGE_ORDER = ["below21", "21", "26", "31", "36", "41", "46", "50plus"]
TIME_ORDER = ["7AM", "10AM", "2PM", "6PM", "10PM"]
INCOME_ORDER = [
    "Less than $12500",
    "$12500 - $24999",
    "$25000 - $37499",
    "$37500 - $49999",
    "$50000 - $62499",
    "$62500 - $74999",
    "$75000 - $87499",
    "$87500 - $99999",
    "$100000 or More",
]
EDUCATION_ORDER = [
    "Some High School",
    "High School Graduate",
    "Some college - no degree",
    "Associates degree",
    "Bachelors degree",
    "Graduate degree (Masters or Doctorate)",
]
FREQUENCY_COLS = (
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
)

_BAR = "#3d5a80"
_LINE = "#9b2226"


def overall_acceptance_rate(train: pd.DataFrame) -> float:
    return float(train[TARGET_COL].mean())


def acceptance_table(
    train: pd.DataFrame,
    column: str,
    order: list[str] | None = None,
    *,
    missing_label: str = "(missing)",
) -> pd.DataFrame:
    """Acceptance count and rate by one column. Missing values stay visible."""
    labels = train[column].astype("string").fillna(missing_label)
    frame = pd.DataFrame({"level": labels, TARGET_COL: train[TARGET_COL].to_numpy()})
    grouped = (
        frame.groupby("level", dropna=False)[TARGET_COL]
        .agg(n="size", accepted="sum")
        .reset_index()
    )
    grouped["acceptance_rate"] = grouped["accepted"] / grouped["n"]
    if order is not None:
        rank = {name: i for i, name in enumerate(order)}
        grouped["_rank"] = grouped["level"].map(lambda v: rank.get(str(v), len(order)))
        grouped = grouped.sort_values(["_rank", "level"]).drop(columns="_rank")
    else:
        grouped = grouped.sort_values("acceptance_rate", ascending=False)
    return grouped.reset_index(drop=True)


def car_observed_table(train: pd.DataFrame) -> pd.DataFrame:
    """Collapse rare car labels into observed vs missing."""
    labels = train["car"].notna().map({True: "observed", False: "missing"})
    frame = pd.DataFrame({"level": labels, TARGET_COL: train[TARGET_COL].to_numpy()})
    grouped = (
        frame.groupby("level", dropna=False)[TARGET_COL]
        .agg(n="size", accepted="sum")
        .reset_index()
    )
    grouped["acceptance_rate"] = grouped["accepted"] / grouped["n"]
    order = {"missing": 0, "observed": 1}
    grouped["_rank"] = grouped["level"].map(order)
    return grouped.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)


def _style_rate_axis(ax: plt.Axes, overall: float) -> None:
    ax.axvline(overall, color=_LINE, linestyle="--", linewidth=1, label="overall")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Acceptance rate")


def _barh(ax: plt.Axes, table: pd.DataFrame, overall: float) -> None:
    plot = table.iloc[::-1]
    ax.barh(plot["level"].astype(str), plot["acceptance_rate"], color=_BAR)
    _style_rate_axis(ax, overall)
    for y, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            min(row["acceptance_rate"] + 0.02, 0.98),
            y,
            f"{row['acceptance_rate']:.0%} (n={int(row['n'])})",
            va="center",
            fontsize=8,
            color="#1d1d1d",
        )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_target_balance(train: pd.DataFrame, path: Path) -> Path:
    counts = train[TARGET_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    labels = ["Not accepted (0)", "Accepted (1)"]
    ax.bar(labels, [int(counts.get(0, 0)), int(counts.get(1, 0))], color=["#6c757d", _BAR])
    ax.set_ylabel("Training rows")
    ax.set_title("Training target balance")
    for i, key in enumerate([0, 1]):
        n = int(counts.get(key, 0))
        ax.text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, path)


def plot_single(
    train: pd.DataFrame,
    column: str,
    path: Path,
    title: str,
    order: list[str] | None = None,
) -> Path:
    overall = overall_acceptance_rate(train)
    table = acceptance_table(train, column, order)
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.38 * len(table) + 1.2)))
    _barh(ax, table, overall)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_panels(
    train: pd.DataFrame,
    specs: list[tuple[str, str, list[str] | None]],
    path: Path,
    title: str,
    ncols: int = 2,
) -> Path:
    overall = overall_acceptance_rate(train)
    n = len(specs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4 * nrows), squeeze=False)
    for ax, (column, panel_title, order) in zip(axes.ravel(), specs):
        _barh(ax, acceptance_table(train, column, order), overall)
        ax.set_title(panel_title)
        ax.legend(loc="lower right", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    return _save(fig, path)


def plot_car_observed(train: pd.DataFrame, path: Path) -> Path:
    overall = overall_acceptance_rate(train)
    table = car_observed_table(train)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    _barh(ax, table, overall)
    ax.set_title("Acceptance when car is missing vs observed")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def generate_figures(train: pd.DataFrame, output_dir: Path | None = None) -> list[Path]:
    """Write the focused training-set chart set. Returns saved paths."""
    out = Path(output_dir) if output_dir is not None else FIGURES_DIR
    saved = [
        plot_target_balance(train, out / "01_target_balance.png"),
        plot_single(
            train,
            "coupon",
            out / "02_acceptance_by_coupon.png",
            "Acceptance rate by coupon type",
        ),
        plot_panels(
            train,
            [
                ("destination", "Destination", None),
                ("passanger", "Passanger", None),
                ("weather", "Weather", None),
                ("temperature", "Temperature", ["30", "55", "80"]),
                ("time", "Time", TIME_ORDER),
                ("expiration", "Expiration", ["2h", "1d"]),
            ],
            out / "03_acceptance_by_context.png",
            "Acceptance rate by trip context",
        ),
        plot_panels(
            train,
            [
                ("age", "Age", AGE_ORDER),
                ("gender", "Gender", None),
                ("maritalStatus", "Marital status", None),
                ("has_children", "Has children", ["0", "1"]),
            ],
            out / "04_acceptance_by_demographics.png",
            "Acceptance rate by demographics",
        ),
        plot_single(
            train,
            "income",
            out / "05_acceptance_by_income.png",
            "Acceptance rate by income",
            INCOME_ORDER,
        ),
        plot_single(
            train,
            "education",
            out / "06_acceptance_by_education.png",
            "Acceptance rate by education",
            EDUCATION_ORDER,
        ),
        plot_single(
            train,
            "occupation",
            out / "07_acceptance_by_occupation.png",
            "Acceptance rate by occupation",
        ),
        plot_panels(
            train,
            [(col, col, FREQ_ORDER) for col in FREQUENCY_COLS],
            out / "08_acceptance_by_venue_frequency.png",
            "Acceptance rate by venue visit frequency",
            ncols=3,
        ),
        plot_panels(
            train,
            [
                ("toCoupon_GEQ15min", "At least 15 minutes", ["0", "1"]),
                ("toCoupon_GEQ25min", "At least 25 minutes", ["0", "1"]),
                ("direction_same", "Same direction", ["0", "1"]),
            ],
            out / "09_acceptance_by_distance_direction.png",
            "Acceptance rate by distance and direction",
        ),
        plot_car_observed(train, out / "10_acceptance_by_car_observed.png"),
    ]
    return saved


def format_table(table: pd.DataFrame) -> str:
    lines = ["level | n | accepted | rate"]
    for _, row in table.iterrows():
        lines.append(
            f"{row['level']} | {int(row['n'])} | {int(row['accepted'])} | {row['acceptance_rate']:.4f}"
        )
    return "\n".join(lines)
