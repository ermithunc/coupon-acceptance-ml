# decision_tree

Training file only. Stratified folds. Engineered feature pipeline.
The official test file was not used. Threshold for class labels is 0.5.

| Metric | CV mean | CV std | Out-of-fold |
| --- | ---: | ---: | ---: |
| accuracy | 0.6973 | 0.0092 | 0.6973 |
| precision | 0.7004 | 0.0039 | 0.7003 |
| recall | 0.8173 | 0.0264 | 0.8173 |
| f1 | 0.7541 | 0.0113 | 0.7543 |
| roc_auc | 0.7397 | 0.0133 | 0.7406 |
| pr_auc | 0.7501 | 0.0121 | 0.7542 |

Out-of-fold Brier score: 0.2010.
Accuracy after a full training refit: 0.7129.
Accuracy gap (full refit minus CV mean): 0.0156.

Out-of-fold confusion matrix, rows actual 0/1, columns predicted 0/1:

| | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Actual 0 | 2362 | 2017 |
| Actual 1 | 1054 | 4714 |

This record does not select the final model.
