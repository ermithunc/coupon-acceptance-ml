# random_forest

Training file only. Stratified folds. Engineered feature pipeline.
The official test file was not used. Threshold for class labels is 0.5.

| Metric | CV mean | CV std | Out-of-fold |
| --- | ---: | ---: | ---: |
| accuracy | 0.7289 | 0.0093 | 0.7289 |
| precision | 0.7230 | 0.0081 | 0.7229 |
| recall | 0.8481 | 0.0090 | 0.8481 |
| f1 | 0.7805 | 0.0073 | 0.7805 |
| roc_auc | 0.8026 | 0.0111 | 0.8022 |
| pr_auc | 0.8305 | 0.0081 | 0.8298 |

Out-of-fold Brier score: 0.1833.
Accuracy after a full training refit: 0.7999.
Accuracy gap (full refit minus CV mean): 0.0711.

Out-of-fold confusion matrix, rows actual 0/1, columns predicted 0/1:

| | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Actual 0 | 2504 | 1875 |
| Actual 1 | 876 | 4892 |

This record does not select the final model.
