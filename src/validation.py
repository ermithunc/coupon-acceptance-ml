"""Data-quality checks for the official coupon-acceptance tables.

Findings are reported, not auto-repaired. Test labels are never used
because the official test file has no target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from src.data_loader import IDENTIFIER_COL, TARGET_COL

NEAR_CONSTANT_SHARE = 0.95

# Observed non-null categories from the official train/test files (Phase 2).
KNOWN_CATEGORIES: dict[str, frozenset[str]] = {
    "destination": frozenset({"Home", "Work", "No Urgent Place"}),
    "passanger": frozenset({"Alone", "Friend(s)", "Kid(s)", "Partner"}),
    "weather": frozenset({"Sunny", "Rainy", "Snowy"}),
    "time": frozenset({"7AM", "10AM", "2PM", "6PM", "10PM"}),
    "coupon": frozenset(
        {
            "Bar",
            "Coffee House",
            "Carry out & Take away",
            "Restaurant(<20)",
            "Restaurant(20-50)",
        }
    ),
    "expiration": frozenset({"1d", "2h"}),
    "gender": frozenset({"Male", "Female"}),
    "age": frozenset({"below21", "21", "26", "31", "36", "41", "46", "50plus"}),
    "maritalStatus": frozenset(
        {"Single", "Married partner", "Unmarried partner", "Divorced", "Widowed"}
    ),
    "education": frozenset(
        {
            "Some High School",
            "High School Graduate",
            "Some college - no degree",
            "Associates degree",
            "Bachelors degree",
            "Graduate degree (Masters or Doctorate)",
        }
    ),
    "income": frozenset(
        {
            "Less than $12500",
            "$12500 - $24999",
            "$25000 - $37499",
            "$37500 - $49999",
            "$50000 - $62499",
            "$62500 - $74999",
            "$75000 - $87499",
            "$87500 - $99999",
            "$100000 or More",
        }
    ),
    "car": frozenset(
        {
            "Mazda5",
            "crossover",
            "do not drive",
            "Scooter and motorcycle",
            "Car that is too old to install Onstar :D",
        }
    ),
    "Bar": frozenset({"never", "less1", "1~3", "4~8", "gt8"}),
    "CoffeeHouse": frozenset({"never", "less1", "1~3", "4~8", "gt8"}),
    "CarryAway": frozenset({"never", "less1", "1~3", "4~8", "gt8"}),
    "RestaurantLessThan20": frozenset({"never", "less1", "1~3", "4~8", "gt8"}),
    "Restaurant20To50": frozenset({"never", "less1", "1~3", "4~8", "gt8"}),
    "occupation": frozenset(
        {
            "Architecture & Engineering",
            "Arts Design Entertainment Sports & Media",
            "Building & Grounds Cleaning & Maintenance",
            "Business & Financial",
            "Community & Social Services",
            "Computer & Mathematical",
            "Construction & Extraction",
            "Education&Training&Library",
            "Farming Fishing & Forestry",
            "Food Preparation & Serving Related",
            "Healthcare Practitioners & Technical",
            "Healthcare Support",
            "Installation Maintenance & Repair",
            "Legal",
            "Life Physical Social Science",
            "Management",
            "Office & Administrative Support",
            "Personal Care & Service",
            "Production Occupations",
            "Protective Service",
            "Retired",
            "Sales & Related",
            "Student",
            "Transportation & Material Moving",
            "Unemployed",
        }
    ),
}

FREQUENCY_COLS = (
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
)


@dataclass
class QualityReport:
    n_train: int
    n_test: int
    missing_train: dict[str, int]
    missing_test: dict[str, int]
    duplicate_rows_train: int
    duplicate_rows_test: int
    duplicate_ids_train: int
    duplicate_ids_test: int
    feature_duplicate_pairs_train: int
    feature_duplicate_pairs_test: int
    feature_duplicate_pairs_mixed_y: int
    constant_columns_train: list[str]
    near_constant_columns_train: dict[str, float]
    target_counts: dict[str, int]
    unexpected_categories: dict[str, list[str]]
    whitespace_issues: dict[str, int]
    case_collisions: dict[str, list[str]]
    train_only_categories: dict[str, list[str]]
    test_only_categories: dict[str, list[str]]
    direction_are_complements: bool
    dtypes_train: dict[str, str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_text_series(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def missing_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df.isna().sum()
    return {col: int(n) for col, n in counts.items() if n > 0}


def duplicate_row_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def duplicate_id_count(df: pd.DataFrame) -> int:
    return int(df[IDENTIFIER_COL].duplicated().sum())


def feature_duplicate_summary(df: pd.DataFrame) -> tuple[int, int]:
    """Count extra rows that match another row on all non-identifier features.

    Returns (extra_rows, mixed_label_groups). mixed_label_groups is 0 when Y
    is absent (test). Extra rows equal sum(group_size - 1) over duplicate groups.
    """
    feature_only = [
        c for c in df.columns if c not in {IDENTIFIER_COL, TARGET_COL}
    ]
    grouped = df.groupby(feature_only, dropna=False, sort=False)
    extra_rows = 0
    mixed_groups = 0
    has_y = TARGET_COL in df.columns
    for _, group in grouped:
        if len(group) < 2:
            continue
        extra_rows += len(group) - 1
        if has_y and group[TARGET_COL].nunique(dropna=False) > 1:
            mixed_groups += 1
    return extra_rows, mixed_groups


def constant_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col == IDENTIFIER_COL:
            continue
        nun = int(df[col].nunique(dropna=True))
        if nun <= 1:
            out.append(col)
    return out


def near_constant_columns(
    df: pd.DataFrame, threshold: float = NEAR_CONSTANT_SHARE
) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in df.columns:
        if col in {IDENTIFIER_COL, TARGET_COL}:
            continue
        share = float(df[col].value_counts(dropna=False, normalize=True).iloc[0])
        nun = int(df[col].nunique(dropna=True))
        if share >= threshold and nun > 1:
            out[col] = round(share, 6)
    return out


def unexpected_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for col, allowed in KNOWN_CATEGORIES.items():
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).unique())
        extra = sorted(values - allowed)
        if extra:
            found[col] = extra
    return found


def whitespace_issues(df: pd.DataFrame) -> dict[str, int]:
    issues: dict[str, int] = {}
    for col in df.columns:
        if not _is_text_series(df[col]):
            continue
        vals = df[col].dropna().astype(str)
        n = int((vals != vals.str.strip()).sum())
        if n:
            issues[col] = n
    return issues


def case_collisions(df: pd.DataFrame) -> dict[str, list[str]]:
    collisions: dict[str, list[str]] = {}
    for col in df.columns:
        if not _is_text_series(df[col]):
            continue
        mapping: dict[str, set[str]] = {}
        for raw in df[col].dropna().astype(str).str.strip():
            mapping.setdefault(raw.lower(), set()).add(raw)
        collided = sorted(
            "/".join(sorted(variants))
            for variants in mapping.values()
            if len(variants) > 1
        )
        if collided:
            collisions[col] = collided
    return collisions


def category_set_diffs(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    train_only: dict[str, list[str]] = {}
    test_only: dict[str, list[str]] = {}
    for col in test.columns:
        if col == IDENTIFIER_COL:
            continue
        tr = set(train[col].dropna().astype(str).unique())
        te = set(test[col].dropna().astype(str).unique())
        only_tr = sorted(tr - te)
        only_te = sorted(te - tr)
        if only_tr:
            train_only[col] = only_tr
        if only_te:
            test_only[col] = only_te
    return train_only, test_only


def target_value_counts(train: pd.DataFrame) -> dict[str, int]:
    counts = train[TARGET_COL].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def directions_are_complements(df: pd.DataFrame) -> bool:
    same = df["direction_same"].astype("Int64")
    opp = df["direction_opp"].astype("Int64")
    return bool(((same + opp) == 1).all() and same.notna().all() and opp.notna().all())


def assess_quality(train: pd.DataFrame, test: pd.DataFrame) -> QualityReport:
    n_pairs, n_mixed = feature_duplicate_summary(train)
    n_pairs_test, _ = feature_duplicate_summary(test)
    train_only, test_only = category_set_diffs(train, test)
    direction_ok = directions_are_complements(train) and directions_are_complements(test)
    notes = [
        "Raw files were not modified.",
        "Issues are documented for later preprocessing; nothing is auto-cleaned here.",
        "passanger spelling is treated as valid, not as a defect.",
    ]
    return QualityReport(
        n_train=len(train),
        n_test=len(test),
        missing_train=missing_counts(train),
        missing_test=missing_counts(test),
        duplicate_rows_train=duplicate_row_count(train),
        duplicate_rows_test=duplicate_row_count(test),
        duplicate_ids_train=duplicate_id_count(train),
        duplicate_ids_test=duplicate_id_count(test),
        feature_duplicate_pairs_train=n_pairs,
        feature_duplicate_pairs_test=n_pairs_test,
        feature_duplicate_pairs_mixed_y=n_mixed,
        constant_columns_train=constant_columns(train),
        near_constant_columns_train=near_constant_columns(train),
        target_counts=target_value_counts(train),
        unexpected_categories={
            **{f"train.{k}": v for k, v in unexpected_categories(train).items()},
            **{f"test.{k}": v for k, v in unexpected_categories(test).items()},
        },
        whitespace_issues={
            **{f"train.{k}": v for k, v in whitespace_issues(train).items()},
            **{f"test.{k}": v for k, v in whitespace_issues(test).items()},
        },
        case_collisions={
            **{f"train.{k}": v for k, v in case_collisions(train).items()},
            **{f"test.{k}": v for k, v in case_collisions(test).items()},
        },
        train_only_categories=train_only,
        test_only_categories=test_only,
        direction_are_complements=direction_ok,
        dtypes_train={c: str(t) for c, t in train.dtypes.items()},
        notes=notes,
    )
