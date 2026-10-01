# xgboost

Training file only. Stratified folds. Engineered feature pipeline.
The official test file was not used. Threshold for class labels is 0.5.

| Metric | CV mean | CV std | Out-of-fold |
| --- | ---: | ---: | ---: |
| accuracy | 0.7526 | 0.0040 | 0.7526 |
| precision | 0.7556 | 0.0045 | 0.7556 |
| recall | 0.8349 | 0.0094 | 0.8350 |
| f1 | 0.7933 | 0.0040 | 0.7933 |
| roc_auc | 0.8214 | 0.0060 | 0.8212 |
| pr_auc | 0.8465 | 0.0076 | 0.8459 |

Out-of-fold Brier score: 0.1692.
Accuracy after a full training refit: 0.8003.
Accuracy gap (full refit minus CV mean): 0.0477.

Out-of-fold confusion matrix, rows actual 0/1, columns predicted 0/1:

| | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Actual 0 | 2821 | 1558 |
| Actual 1 | 952 | 4816 |

This record does not select the final model.
