# Data Quality Report

Inspection date: 2026-09-30.

Sources: official `train.csv` (10,147 rows) and `test.csv` (2,537 rows), loaded via `src/data_loader.py`. Raw files were **not** modified. Nothing was auto-cleaned.

Code: `src/validation.py`.

## Summary

| Check | Train | Test | Treatment (later phases) |
| --- | --- | --- | --- |
| Full-row duplicates | 0 | 0 | None |
| Duplicate `customer_id` | 0 | 0 | Keep as identifier |
| Unexpected categories vs known sets | none | none | Keep labels as stored |
| Leading/trailing whitespace | none | none | None |
| Case collisions (same token, different case) | none | none | None |
| Train/test category mismatch | none | none | Safe to share category levels |
| Target `Y` | 5,768 accepted (`1`), 4,379 not (`0`); no nulls | column absent | Binary classification; accept-class share 56.84% |

## Missing values

| Column | Train missing | Train % | Test missing | Test % | Recommended treatment |
| --- | ---: | ---: | ---: | ---: | --- |
| car | 10,063 | 99.17% | 2,513 | 99.05% | Do not drop yet. 84 non-null train rows. Encode `missing` plus rare observed types, or a missing indicator. Revisit in EDA. |
| CoffeeHouse | 172 | 1.70% | 45 | 1.77% | Impute inside the training/CV pipeline (most frequent or explicit `missing`). Fit on train folds only. |
| Restaurant20To50 | 148 | 1.46% | 41 | 1.62% | Same as other frequency columns. |
| CarryAway | 122 | 1.20% | 29 | 1.14% | Same. |
| RestaurantLessThan20 | 97 | 0.96% | 33 | 1.30% | Same. |
| Bar | 88 | 0.87% | 19 | 0.75% | Same. |

No other columns have missing values.

## Constant and near-constant columns

| Column | Finding | Evidence | Recommended treatment |
| --- | --- | --- | --- |
| toCoupon_GEQ5min | Constant | Always `1` in train and test | Candidate to drop in preprocessing; it cannot discriminate. Confirm in EDA. |
| car | Near-constant | 99.17% of train rows are missing | Keep for EDA. A missing/observed flag may be more useful than five rare labels. |
| toCoupon_GEQ25min | Imbalanced binary, **not** near-constant at the 95% rule | 88.02% are `0` | Keep. Minority class (GEQ 25 min) is 11.98%. |
| weather | Imbalanced, not near-constant | Sunny 78.99%, Snowy 11.25%, Rainy 9.76% | Keep. |
| direction_same / direction_opp | Exact complements | Every train and test row is `(1,0)` or `(0,1)` | Keep one of the two in modeling to avoid redundant collinearity. |

Near-constant rule in code: most frequent value (including NA) share ≥ 0.95 **and** more than one non-null level.

## Feature-duplicate profiles (not full-row duplicates)

`customer_id` is unique, so there are no duplicate IDs. After ignoring `customer_id`:

- Train: 63 groups of size 2 (63 extra rows). 15 of those groups have **disagreeing `Y`**.
- Test: 4 groups of size 2 (4 extra rows). No `Y` to compare.

This is a data-quality finding, not a license to drop rows. Possible readings: repeated contexts with different outcomes, or label noise. **Do not drop automatically.** Later modeling can keep all rows unless a training-only experiment shows a benefit to de-duplication.

## Data types (pandas after load)

| Columns | Dtype |
| --- | --- |
| customer_id, temperature, has_children, toCoupon_GEQ5min, toCoupon_GEQ15min, toCoupon_GEQ25min, direction_same, direction_opp, Y | int64 |
| destination, passanger, weather, time, coupon, expiration, gender, age, maritalStatus, education, occupation, income, car, Bar, CoffeeHouse, CarryAway, RestaurantLessThan20, Restaurant20To50 | string |

`age` is stored as text because of `below21` and `50plus`. `temperature` has only three numeric levels (30, 55, 80). Encoding choices belong to preprocessing, not this phase.

`passanger` is a valid column name in the files. It is not treated as a quality defect.

## Occupation

25 occupation labels in train and the same 25 in test. No unexpected values. Cardinality is higher than other categoricals; encoding strategy is deferred.

## What was not done

- No imputation
- No row drops
- No column drops in the raw files
- No recoding of `passanger`
- No use of test `Y` (none exists)
- No model training
