# random_forest_tuned

Training file only. Stratified folds. Engineered feature pipeline.
The official test file was not used. Threshold for class labels is 0.5.

| Metric | CV mean | CV std | Out-of-fold |
| --- | ---: | ---: | ---: |
| accuracy | 0.7513 | 0.0046 | 0.7513 |
| precision | 0.7448 | 0.0058 | 0.7447 |
| recall | 0.8558 | 0.0067 | 0.8558 |
| f1 | 0.7964 | 0.0034 | 0.7964 |
| roc_auc | 0.8201 | 0.0093 | 0.8197 |
| pr_auc | 0.8436 | 0.0067 | 0.8425 |

Out-of-fold Brier score: 0.1703.
Accuracy after a full training refit: 0.8936.
Accuracy gap (full refit minus CV mean): 0.1423.

Out-of-fold confusion matrix, rows actual 0/1, columns predicted 0/1:

| | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Actual 0 | 2687 | 1692 |
| Actual 1 | 832 | 4936 |

This record does not select the final model.
