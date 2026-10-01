# Model comparison

Untuned models. Same training file, engineered features, stratified 5-fold protocol.
The official test file was not used. This table does not by itself select the final model.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train-CV accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.7526 ± 0.0040 | 0.7556 ± 0.0045 | 0.8349 ± 0.0094 | 0.7933 ± 0.0040 | 0.8214 ± 0.0060 | 0.8465 ± 0.0076 | 0.1692 | 0.0477 |
| random_forest | 0.7289 ± 0.0093 | 0.7230 ± 0.0081 | 0.8481 ± 0.0090 | 0.7805 ± 0.0073 | 0.8026 ± 0.0111 | 0.8305 ± 0.0081 | 0.1833 | 0.0711 |
| logistic_regression | 0.7012 ± 0.0138 | 0.7168 ± 0.0125 | 0.7843 ± 0.0103 | 0.7490 ± 0.0109 | 0.7590 ± 0.0107 | 0.7913 ± 0.0109 | 0.1959 | 0.0081 |
| decision_tree | 0.6973 ± 0.0092 | 0.7004 ± 0.0039 | 0.8173 ± 0.0264 | 0.7541 ± 0.0113 | 0.7397 ± 0.0133 | 0.7501 ± 0.0121 | 0.2010 | 0.0156 |

## Measured order

On ROC-AUC, PR-AUC, F1, accuracy, and Brier score, `xgboost` is first in this untuned set.
`random_forest` has the highest recall (0.8481).
Lower Brier is the better calibration score. Lower ROC-AUC standard deviation is the tighter fold spread.

## Other factors

- Stability: `xgboost` has the smallest ROC-AUC fold spread (0.0060). `decision_tree` has the widest recall spread (0.0264).
- Interpretability: logistic regression stays a linear baseline. The decision tree is one depth-capped tree. Random forest and XGBoost are ensembles.
- Inference: each model scores one row through the same preprocessing pipeline on CPU.
- Deployment: logistic regression, the tree, and the forest need scikit-learn. XGBoost also needs the xgboost package.
- Overfit gap, full-refit accuracy minus CV accuracy: `logistic_regression` 0.0081, `decision_tree` 0.0156, `xgboost` 0.0477, `random_forest` 0.0711.

## Tuning shortlist

Rule: keep the ROC-AUC leader, and the next model when it is within 0.02 ROC-AUC. At most 2 models.

Shortlist: `xgboost`, `random_forest`.

Final selection waits until that bounded search is scored on the same 5-fold protocol.
