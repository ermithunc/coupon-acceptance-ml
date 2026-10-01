# Tuning results

Training file only. `RandomizedSearchCV`, 8 draws, 3 stratified folds, scoring ROC-AUC. The 5-fold numbers are a fresh run of the best settings on the shared protocol. The official test file was not used.

Shortlist from the untuned comparison: XGBoost and random forest.

## XGBoost

- Search space: `n_estimators` {100, 200, 300}, `max_depth` {3, 4, 6}, `learning_rate` {0.05, 0.1, 0.2}, `min_child_weight` {1, 5}, `subsample` {0.8, 1.0}
- Search ROC-AUC (3-fold): 0.8311
- Runtime: 29.1 seconds
- Best parameters: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `min_child_weight=1`, `subsample=0.8`
- 5-fold accuracy 0.7654 ± 0.0054
- 5-fold precision 0.7715, recall 0.8344, F1 0.8017
- 5-fold ROC-AUC 0.8381 ± 0.0062, PR-AUC 0.8596
- Brier 0.1606
- Train-CV accuracy gap 0.1247 (full-refit accuracy 0.8900)

Untuned XGBoost ROC-AUC was 0.8214 with a gap of 0.0477.

## Random forest

- Search space: `n_estimators` {100, 200, 400}, `max_depth` {8, 12, 16, unlimited}, `min_samples_leaf` {1, 5, 10}, `max_features` {sqrt, 0.5}
- Search ROC-AUC (3-fold): 0.8148
- Runtime: 71.0 seconds
- Best parameters: `n_estimators=200`, `max_depth=12`, `min_samples_leaf=1`, `max_features=0.5`
- 5-fold accuracy 0.7513 ± 0.0046
- 5-fold precision 0.7448, recall 0.8558, F1 0.7964
- 5-fold ROC-AUC 0.8201 ± 0.0093, PR-AUC 0.8436
- Brier 0.1703
- Train-CV accuracy gap 0.1423 (full-refit accuracy 0.8936)

Untuned random forest ROC-AUC was 0.8026 with a gap of 0.0711.

The 3-fold search score is only the search objective. Model comparison uses the 5-fold metrics.
