# Inference and application testing

Checked the saved file `models/inference_pipeline.joblib`. Nothing was refit. The official test file was scored only to confirm the saved submission.

Command: `pytest tests/test_inference.py` — 7 passed.

| Check | Result |
| --- | --- |
| Model loading | Outer pipeline steps are `prep` and `model`. The classifier is `XGBClassifier`. |
| Feature schema | `feature_names_in_` matches the official test columns, including `passanger`. `Y` is absent. |
| Single-row inference | Default form row (21, Male, friends, Coffee House, sunny, 6PM, no urgent place) scores 0.846452. At threshold 0.5 the class is 1. |
| Prediction type | Acceptance probability is a float. The class from `classify` is an int, 0 or 1. |
| Probability range | Single-row, missing-value, unknown-category, and full test-batch probabilities are finite and inside `[0, 1]`. |
| Batch inference | All 2,537 test rows. Labels at threshold 0.5 match `submission.csv`. |
| Missing values | One official test row with `car`, `Bar`, and `CoffeeHouse` set to missing still returns a probability in `[0, 1]`. |
| Invalid categories | `coupon = Not A Real Coupon` and `destination = Mars` are outside `KNOWN_CATEGORIES`. The pipeline does not raise. One-hot encoding uses `handle_unknown="ignore"`, so those levels add no dummy column. The Streamlit form only lists training-file categories. |
| Streamlit smoke test | AppTest opens `app/app.py`, submits the default form, and shows 84.6% and Accept. That text matches a direct `predict_proba` on the same row. |
