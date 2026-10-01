# Exploratory findings (training set only)

Inspection date: 2026-10-01.

Source: `train.csv` via `src/data_loader.py` (10,147 rows). The test file was not used. Charts: `reports/figures/`. Code: `src/eda.py`. Notebook: `notebooks/02_eda.ipynb`.

These are descriptions of the training sample. They are not causal effects.

Overall acceptance rate: **56.84%** (5,768 accepted, 4,379 not accepted). The dashed line on rate charts is this overall rate.

## Coupon

| Coupon | n | Acceptance rate |
| --- | ---: | ---: |
| Carry out & Take away | 1,923 | 73.5% |
| Restaurant(<20) | 2,233 | 70.5% |
| Coffee House | 3,191 | 50.1% |
| Restaurant(20-50) | 1,177 | 43.8% |
| Bar | 1,623 | 40.9% |

Carry-out and cheaper restaurant coupons sit well above the overall rate. Bar coupons sit below it. Coffee House is the largest group and is near 50%.

## Trip context

| Factor | Pattern in training data |
| --- | --- |
| destination | No Urgent Place 63.4% (n=5,045). Home 50.5%. Work 50.2%. |
| passanger | Friend(s) 67.3% (n=2,676). Partner 59.6%. Alone 52.6% (n=5,802). Kid(s) 50.2%. |
| weather | Sunny 59.4% (n=8,015). Snowy 47.8%. Rainy 47.0%. |
| temperature | 80°F 59.7% (n=5,203). 30°F 53.9%. 55°F 53.8%. |
| time | 2PM 65.5%. 10AM 61.6%. 6PM 58.2%. 10PM 50.8%. 7AM 50.2%. |
| expiration | 1d 62.7% (n=5,643). 2h 49.5% (n=4,504). |

`passanger` keeps the dataset spelling.

## Demographics

| Factor | Pattern |
| --- | --- |
| age | below21 62.0% (n=432). 21 and 26 are about 59.6–59.8%. 50plus 51.6% (n=1,431). The middle bands (31, 36) are about 53.5–54.1%. |
| gender | Male 59.0% (n=4,943). Female 54.8% (n=5,204). |
| maritalStatus | Single 60.3% (n=3,806). Married partner 54.4% (n=4,086). Widowed 47.5% (n=101, small). |
| has_children | 0 → 58.5% (n=5,960). 1 → 54.4% (n=4,187). |
| education | Some High School 77.9% but **n=68**. Other levels sit between 52.9% (graduate degree, n=1,477) and 59.8% (some college, n=3,459). |
| income | No steady rise with income. Lowest band in this sample: $75,000–$87,499 at 47.3% (n=695). $100,000 or More is 58.4% (n=1,408). |
| occupation | Highest in this sample: Healthcare Support 69.9% (n=186). Lowest: Retired 46.7% (n=394). Several occupations have n under 200, so the extremes are less stable. |

## Venue frequency

Ordered labels: never, less1, 1~3, 4~8, gt8. Missing rows are kept as their own level.

| Column | Descriptive pattern |
| --- | --- |
| CoffeeHouse | never 45.2% (n=2,375). 1~3 65.4% (n=2,586). 4~8 62.5%. gt8 58.1%. |
| Bar | never 52.9%. Rate rises through 4~8 (64.7%, n=859), then gt8 is 57.9% (n=290). |
| Restaurant20To50 | never 52.6%. gt8 66.7% (n=216). The ordered non-missing levels rise together in this sample. |
| RestaurantLessThan20 | Flatter: never 54.9% (n=173) to gt8 60.9% (n=1,025). |
| CarryAway | Flatter still: less1 50.4% to gt8 58.2%. |

## Distance and direction

| Column | Pattern |
| --- | --- |
| toCoupon_GEQ5min | Constant `1` in training data. No chart. It cannot separate accepted from not accepted. |
| toCoupon_GEQ15min | 0 → 61.4% (n=4,434). 1 → 53.3% (n=5,713). |
| toCoupon_GEQ25min | 0 → 58.6% (n=8,931). 1 → 43.6% (n=1,216). |
| direction_same | 0 → 56.5% (n=7,994). 1 → 58.2% (n=2,153). |
| direction_opp | Exact complement of `direction_same`, so the rates are the mirror image. |

## Car

`car` is missing on 10,063 training rows. Acceptance is 56.9% when missing (n=10,063) and 53.6% when any car value is present (n=84). The observed group is small. Individual car labels were not charted.

## What this phase did not do

- No model training
- No use of test rows for these rates
- No claim that a factor causes acceptance
- No change to the raw CSVs
