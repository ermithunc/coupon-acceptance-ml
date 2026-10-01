# logistic_regression

Training file only. Stratified folds. Engineered feature pipeline.
The official test file was not used. Threshold for class labels is 0.5.

| Metric | CV mean | CV std | Out-of-fold |
| --- | ---: | ---: | ---: |
| accuracy | 0.7012 | 0.0138 | 0.7012 |
| precision | 0.7168 | 0.0125 | 0.7167 |
| recall | 0.7843 | 0.0103 | 0.7843 |
| f1 | 0.7490 | 0.0109 | 0.7490 |
| roc_auc | 0.7590 | 0.0107 | 0.7588 |
| pr_auc | 0.7913 | 0.0109 | 0.7900 |

Out-of-fold Brier score: 0.1959.
Accuracy after a full training refit: 0.7093.
Accuracy gap (full refit minus CV mean): 0.0081.

Out-of-fold confusion matrix, rows actual 0/1, columns predicted 0/1:

| | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Actual 0 | 2591 | 1788 |
| Actual 1 | 1244 | 4524 |

This record does not select the final model.
