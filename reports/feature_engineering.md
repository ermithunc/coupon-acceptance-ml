# Feature engineering

Code: `src/feature_engineering.py`. Probe results: `reports/feature_comparison.csv`.

Training rows only. Mappings are fixed from label meaning. `Y` is not an input. The test file is not read. `customer_id` appears once per training row, so there is no repeat-customer history and no customer-coupon rate was built.

## Features

| Feature | Source | Rule |
| --- | --- | --- |
| hour | time | 7AM→7, 10AM→10, 2PM→14, 6PM→18, 10PM→22 |
| daypart | time | morning (7AM, 10AM), afternoon (2PM), evening (6PM), night (10PM) |
| age_ord | age | below21→0, 21→1, 26→2, 31→3, 36→4, 41→5, 46→6, 50plus→7 |
| income_ord | income | Less than $12500→0 through $100000 or More→8, in band order |
| bar_ord, coffeehouse_ord, carryaway_ord, restaurant_lt20_ord, restaurant_20to50_ord | venue frequency | never→0, less1→1, 1~3→2, 4~8→3, gt8→4. Missing stays missing for the training imputer. |
| coupon_venue_freq_ord | coupon plus the matching frequency column | Bar→Bar, Coffee House→CoffeeHouse, Carry out & Take away→CarryAway, Restaurant(<20)→RestaurantLessThan20, Restaurant(20-50)→Restaurant20To50, then the same 0–4 scale |
| distance_band | toCoupon_GEQ15min, toCoupon_GEQ25min | 25_plus if GEQ25 is 1, else 15_to_25 if GEQ15 is 1, else 5_to_15. GEQ5 is always 1, so the short band is 5 to 15 minutes. |
| destination_direction | destination, direction_same | Example: `Work\|same`, `Home\|opposite` |

Education was left as one-hot. Its labels are not a single ladder (some college sits beside an associate degree).

Original columns are kept. The Phase 5 one-hot pipeline still runs, and these columns are added beside it.

## Comparison

Probe: logistic regression, stratified 5-fold, shuffle, `random_state=42`, training data only. Same folds for both sets. This is a feature-set check, not the Phase 8 model record.

| Feature set | Accuracy mean | Accuracy std | ROC-AUC mean | ROC-AUC std |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.6852 | 0.0096 | 0.7360 | 0.0100 |
| engineered | 0.7012 | 0.0138 | 0.7590 | 0.0107 |

Source: `reports/feature_comparison.csv`, computed 2026-10-01.

ROC-AUC rose by 0.0230. That gap is larger than either fold standard deviation (about 0.010–0.011). Accuracy rose by 0.0160. The engineered accuracy spread is wider (0.0138 vs 0.0096).

## Decision

Later model phases will use `make_engineered_pipeline()` as the default matrix. The baseline pipeline stays in the repo so the comparison can be repeated. This does not select the final model.
