# Preprocessing decisions

Code: `src/preprocessing.py`. Fit on training rows or training folds only.

## Target and identifier

- `Y` is separated with `split_xy` before the pipeline. It is not a feature.
- `customer_id` is unique on every training row. It is dropped (`remainder="drop"`).

## Columns left out of the model matrix

| Column | Reason |
| --- | --- |
| customer_id | Identifier. One row per id in train and in test. |
| toCoupon_GEQ5min | Constant `1` in the official train and test files. |
| direction_opp | Exact complement of `direction_same`. Keeping both would duplicate one bit. `direction_same` is kept. |

## Missing values

Categorical columns, including `car` and the five venue-frequency columns, are filled with the constant label `missing` inside the pipeline. Mode imputation would hide `car`, which is missing on 99.17% of training rows.

`temperature` uses a median imputer, then `StandardScaler`. The official training file has no missing temperatures. The imputer is there so a later empty value cannot crash inference. Its median is learned on the fit rows only.

Binary columns (`has_children`, `toCoupon_GEQ15min`, `toCoupon_GEQ25min`, `direction_same`) use most-frequent imputation and are not scaled.

## Encoding

Nominal columns are one-hot encoded. Unknown categories at transform time are ignored.

`passanger` is encoded under that spelling.

Age, income, and venue frequency stay one-hot in this phase. Ordered numeric versions are a later experiment, not the default, because a numeric code is only justified when the order is semantic and a comparison shows it helps.

## Scaling

`temperature` is scaled because its values are 30, 55, and 80, which would dominate an unscaled linear model relative to 0/1 flags. Binary flags are not scaled.

## What this phase does not do

- It does not fit on `test.csv`.
- It does not train a classifier.
- It does not drop `car`.
- It does not rewrite raw files.
