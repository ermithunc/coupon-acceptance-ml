# Submission check

Model: tuned XGBoost inference pipeline, fit on the full training file after selection. Threshold 0.5.

Artifact: `models/inference_pipeline.joblib` (feature engineering, preprocessing, and the classifier).

File: `submission.csv`, columns `customer_id`, `Y`.

| Check | Result |
| --- | --- |
| Row count | 2,537 |
| customer_id order vs test.csv | match |
| customer_id order vs sample_submission.csv | match |
| Missing predictions | 0 |
| Y values | only 0 and 1 |
| Predicted 0 | 1,021 |
| Predicted 1 | 1,516 |

The raw test file was not modified. Test rows were not used to fit the pipeline.

The performance claim for this model is the 5-fold cross-validation ROC-AUC of 0.8381, stored in `models/model_card.json`. The joblib file is a refit on all 10,147 training rows so it can score new customers. Its accuracy on those same training rows is not the performance claim.
