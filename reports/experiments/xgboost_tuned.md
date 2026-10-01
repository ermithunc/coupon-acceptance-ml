# xgboost_tuned

Training file only. Stratified folds. Engineered feature pipeline.
The official test file was not used. Threshold for class labels is 0.5.

| Metric | CV mean | CV std | Out-of-fold |
| --- | ---: | ---: | ---: |
| accuracy | 0.7654 | 0.0054 | 0.7653 |
| precision | 0.7715 | 0.0073 | 0.7714 |
| recall | 0.8344 | 0.0048 | 0.8344 |
| f1 | 0.8017 | 0.0036 | 0.8017 |
| roc_auc | 0.8381 | 0.0062 | 0.8378 |
| pr_auc | 0.8596 | 0.0090 | 0.8589 |

Out-of-fold Brier score: 0.1606.
Accuracy after a full training refit: 0.8900.
Accuracy gap (full refit minus CV mean): 0.1247.

Out-of-fold confusion matrix, rows actual 0/1, columns predicted 0/1:

| | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Actual 0 | 2953 | 1426 |
| Actual 1 | 955 | 4813 |

This record does not select the final model.
