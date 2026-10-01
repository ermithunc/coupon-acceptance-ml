# Final model decision

Selected model: **tuned XGBoost** (`xgboost_tuned`).

The choice uses the shared training-file protocol: engineered features, stratified 5-fold, threshold 0.5. The official test file was not used. Accuracy is one of the metrics, not the only one.

## Evidence

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy | Brier | ROC-AUC std | Train-CV accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost_tuned | 0.8381 | 0.8596 | 0.8017 | 0.7715 | 0.8344 | 0.7654 | 0.1606 | 0.0062 | 0.1247 |
| xgboost | 0.8214 | 0.8465 | 0.7933 | 0.7556 | 0.8349 | 0.7526 | 0.1692 | 0.0060 | 0.0477 |
| random_forest_tuned | 0.8201 | 0.8436 | 0.7964 | 0.7448 | 0.8558 | 0.7513 | 0.1703 | 0.0093 | 0.1423 |
| random_forest | 0.8026 | 0.8305 | 0.7805 | 0.7230 | 0.8481 | 0.7289 | 0.1833 | 0.0111 | 0.0711 |
| logistic_regression | 0.7590 | 0.7913 | 0.7490 | 0.7168 | 0.7843 | 0.7012 | 0.1959 | 0.0107 | 0.0081 |
| decision_tree | 0.7397 | 0.7501 | 0.7541 | 0.7004 | 0.8173 | 0.6973 | 0.2010 | 0.0133 | 0.0156 |

Lower Brier is better. Sources: `reports/experiments/` and `reports/tuning_results.md`.

Tuned settings: 200 trees, `max_depth=6`, `learning_rate=0.1`, `min_child_weight=1`, `subsample=0.8`.

## Why this model

- Highest ROC-AUC (0.8381) and highest PR-AUC (0.8596). The gap to the next model is about 0.017 ROC-AUC, larger than the fold spread of 0.0062.
- Highest F1 (0.8017) and highest precision (0.7715), with recall 0.8344.
- Best Brier score (0.1606), so the probabilities are the best calibrated of this set.
- Fold ROC-AUC spread stays tight (0.0062), close to the untuned XGBoost spread (0.0060).
- One-row inference stays inside the same sklearn pipeline. The extra deployment need is the `xgboost` package.

The full-refit training accuracy is 0.8900. That number is not the performance claim. The claim is the 5-fold result. The train-versus-CV accuracy gap is 0.1247 because `max_depth=6` fits the training rows more closely than the untuned depth of 4.

## Models not selected

- Untuned XGBoost: same family and a smaller overfit gap (0.0477), but lower ROC-AUC, PR-AUC, F1, precision, accuracy, and a worse Brier score on the same protocol.
- Tuned random forest: highest recall (0.8558). It is behind on ROC-AUC (0.8201), PR-AUC (0.8436), F1 (0.7964), precision (0.7448), accuracy (0.7513), and Brier (0.1703), with a wider fold spread and a larger train-CV gap (0.1423).
- Untuned random forest: below both tuned ensembles on ROC-AUC and PR-AUC.
- Logistic regression: clearest linear baseline and the smallest overfit gap (0.0081). ROC-AUC 0.7590 is well behind the selected model, so it stays the reference rather than the deployed scorer.
- Decision tree: useful as a single-tree check. ROC-AUC 0.7397 is the lowest of the set, and its recall spread is the widest (0.0264).

## Trade-off

The selected model is harder to read as coefficients than logistic regression. Explanations for the demo come from SHAP on this fitted model, using association language rather than causal claims.
