# Data Dictionary

Source files (unchanged raw CSVs):

- `Dataset/Datasets/train.csv`
- `Dataset/Datasets/test.csv`
- `Dataset/Datasets/sample_submission.csv`

Inspection date: 2026-09-30.

Loader: `src/data_loader.py`. Raw files are not copied or rewritten.

## Row counts

| File | Official expected | Actual rows | Actual columns |
| --- | ---: | ---: | ---: |
| train.csv | 10,147 | 10,147 | 27 |
| test.csv | 2,537 | 2,537 | 26 |
| sample_submission.csv | 2,537 | 2,537 | 2 |

Actual sizes **match** the official statement. No silent correction was required.

## Identifier and target

| Column | Role | Train | Test | Notes from actual files |
| --- | --- | --- | --- | --- |
| `customer_id` | identifier | present, unique (10,147) | present, unique (2,537) | Integer-like IDs. No train/test ID overlap. Not a modeling feature unless a later experiment documents otherwise. |
| `Y` | target | present, values `{0, 1}` | **absent** | `1` = coupon accepted (5,768). `0` = not accepted (4,379). |

`sample_submission.csv` has columns `customer_id`, `Y`. Its `customer_id` values match `test.csv` **in the same order**. Its `Y` values (`0`: 1,283; `1`: 1,254) are a template, **not** test labels. Do not train or evaluate on them.

## Feature columns

Spelling is taken from the files. `passanger` is **not** renamed.

| Column | Observed type in CSV | nunique (train) | Missing (train) | Missing (test) | Observed values / notes |
| --- | --- | ---: | ---: | ---: | --- |
| destination | string | 3 | 0 | 0 | Home, Work, No Urgent Place |
| passanger | string | 4 | 0 | 0 | Alone, Friend(s), Kid(s), Partner |
| weather | string | 3 | 0 | 0 | Sunny, Rainy, Snowy |
| temperature | integer-like | 3 | 0 | 0 | 30, 55, 80 (few discrete levels; treat as numeric or categorical in later phases) |
| time | string | 5 | 0 | 0 | 7AM, 10AM, 2PM, 6PM, 10PM |
| coupon | string | 5 | 0 | 0 | Bar, Coffee House, Carry out & Take away, Restaurant(<20), Restaurant(20-50) |
| expiration | string | 2 | 0 | 0 | 1d, 2h |
| gender | string | 2 | 0 | 0 | Male, Female |
| age | mixed string | 8 | 0 | 0 | below21, 21, 26, 31, 36, 41, 46, 50plus |
| maritalStatus | string | 5 | 0 | 0 | Single, Married partner, Unmarried partner, Divorced, Widowed |
| has_children | integer-like | 2 | 0 | 0 | 0, 1 |
| education | string | 6 | 0 | 0 | Some High School, High School Graduate, Some college - no degree, Associates degree, Bachelors degree, Graduate degree (Masters or Doctorate) |
| occupation | string | 25 | 0 | 0 | 25 occupation labels (same set in train and test) |
| income | string | 9 | 0 | 0 | Less than $12500 through $100000 or More |
| car | string | 5 | 10,063 | 2,513 | Rare non-missing values: Mazda5, crossover, do not drive, Scooter and motorcycle, Car that is too old to install Onstar :D. **Inspected, not dropped.** |
| Bar | string | 5 | 88 | 19 | never, less1, 1~3, 4~8, gt8 |
| CoffeeHouse | string | 5 | 172 | 45 | never, less1, 1~3, 4~8, gt8 |
| CarryAway | string | 5 | 122 | 29 | never, less1, 1~3, 4~8, gt8 |
| RestaurantLessThan20 | string | 5 | 97 | 33 | never, less1, 1~3, 4~8, gt8 |
| Restaurant20To50 | string | 5 | 148 | 41 | never, less1, 1~3, 4~8, gt8 |
| toCoupon_GEQ5min | integer-like | 1 | 0 | 0 | Always `1` in train and test. **Inspected; constant column.** |
| toCoupon_GEQ15min | integer-like | 2 | 0 | 0 | 0, 1 |
| toCoupon_GEQ25min | integer-like | 2 | 0 | 0 | 0, 1 |
| direction_same | integer-like | 2 | 0 | 0 | 0, 1 |
| direction_opp | integer-like | 2 | 0 | 0 | 0, 1 |

Frequency columns (`Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`) share the same category set. They are **not** converted to numbers in this phase.

## Isolation rules

- Training analysis and any later fitting use `train.csv` only.
- `test.csv` has no `Y` and must not be used to learn encodings, impute statistics, or select models.
- This phase does not train models and does not write processed datasets.

## Open items for later phases (not cleaned here)

- High missingness on `car`.
- Modest missingness on venue-frequency columns.
- Constant `toCoupon_GEQ5min`.
- Whether `temperature` and binary flags stay numeric.
