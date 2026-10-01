# Experiment protocol

Code: `src/experiment.py`.

## What is compared later

Every model uses the same rules:

- Training file only (`train.csv`, 10,147 rows).
- Feature set: engineered pipeline from Phase 6.
- `StratifiedKFold`, 5 folds, shuffle, `random_state=42`.
- Preprocessing is inside the pipeline and is fit on the training side of each fold.
- Decision threshold for precision, recall, F1, and the confusion matrix: 0.5.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC. Also Brier score and a calibration curve from out-of-fold probabilities.
- A full refit on all training rows is scored only to show the accuracy gap versus cross-validation. It is not a selection metric.

`stratified_validation_split` is available for a single 80/20 training-file split. Model records use the 5-fold out-of-fold predictions, not that one split.

## What is not allowed

The official test file is not read for feature selection, tuning, threshold choice, or model choice.

## Where results go

- `reports/experiments/{model}.json`
- `reports/experiments/results.csv`
- `reports/figures/cm_{model}.png`
- `reports/figures/calibration_{model}.png`
- `reports/experiments/{model}.md`
- `reports/{model}_evaluation.md`
