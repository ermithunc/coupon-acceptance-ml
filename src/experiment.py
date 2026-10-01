"""Shared validation protocol for coupon-acceptance models.

Stratified folds are the evidence used for later comparison. The official
test file is never read here. Preprocessing is inside the pipeline, so it
is fit on each training fold only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT, TARGET_COL, split_xy
from src.feature_engineering import make_engineered_pipeline

RANDOM_STATE = 42
N_SPLITS = 5
VALIDATION_SIZE = 0.2
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")

EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def make_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def stratified_validation_split(
    train_df: pd.DataFrame,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
):
    """One stratified split of the training file. This is not the official test set."""
    X, y = split_xy(train_df)
    return train_test_split(
        X,
        y,
        test_size=validation_size,
        stratify=y,
        random_state=random_state,
    )


def make_model_pipeline(estimator) -> Pipeline:
    """Engineered features and preprocessing, then the classifier."""
    return Pipeline(
        steps=[
            ("prep", make_engineered_pipeline()),
            ("model", estimator),
        ]
    )


def metric_bundle(y_true, y_pred, y_proba) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def _fold_arrays(pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold):
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_metrics: list[dict[str, float]] = []
    for train_idx, valid_idx in cv.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        oof_proba[valid_idx] = proba
        oof_pred[valid_idx] = pred
        fold_metrics.append(metric_bundle(y.iloc[valid_idx], pred, proba))
    return oof_pred, oof_proba, fold_metrics


def _summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(fold_metrics)
    summary = {}
    for name in METRIC_NAMES:
        summary[name] = {
            "mean": float(frame[name].mean()),
            "std": float(frame[name].std(ddof=0)),
            "folds": [float(value) for value in frame[name].tolist()],
        }
    return summary


def save_confusion_figure(matrix: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(title)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{int(matrix[row, col]):,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_calibration_figure(y_true, y_proba, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=8, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9b2226", label="perfect")
    ax.plot(mean_predicted, fraction_positive, marker="o", color="#3d5a80", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction accepted in bin")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_experiment(
    name: str,
    estimator,
    train_df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Cross-validate one estimator on the training frame and store the record."""
    if TARGET_COL not in train_df.columns:
        raise ValueError("run_experiment requires the training frame, including Y.")
    X, y = split_xy(train_df)
    pipeline = make_model_pipeline(estimator)
    cv = make_cv(n_splits=n_splits, random_state=random_state)
    oof_pred, oof_proba, fold_metrics = _fold_arrays(pipeline, X, y, cv)
    cv_summary = _summarize_folds(fold_metrics)
    oof_metrics = metric_bundle(y, oof_pred, oof_proba)
    matrix = confusion_matrix(y, oof_pred, labels=[0, 1])
    brier = float(brier_score_loss(y, oof_proba))

    full = clone(pipeline)
    full.fit(X, y)
    train_accuracy = float(accuracy_score(y, full.predict(X)))

    payload = {
        "model": name,
        "n_rows": int(len(train_df)),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "feature_set": "engineered",
        "decision_threshold": 0.5,
        "cv": cv_summary,
        "oof": oof_metrics,
        "oof_brier": brier,
        "confusion_matrix": matrix.astype(int).tolist(),
        "train_accuracy_full_refit": train_accuracy,
        "overfit_gap_accuracy": train_accuracy - cv_summary["accuracy"]["mean"],
        "estimator_params": estimator.get_params(deep=False),
    }

    experiments = Path(output_dir) if output_dir is not None else EXPERIMENTS_DIR
    figures = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    experiments.mkdir(parents=True, exist_ok=True)
    json_path = experiments / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_confusion_figure(
        matrix,
        figures / f"cm_{name}.png",
        f"{name} out-of-fold confusion matrix",
    )
    save_calibration_figure(
        y,
        oof_proba,
        figures / f"calibration_{name}.png",
        f"{name} out-of-fold calibration",
    )
    refresh_results_table(experiments)
    note_path = experiments / f"{name}.md"
    _write_model_note(payload, note_path)
    if output_dir is None:
        _write_model_note(payload, PROJECT_ROOT / "reports" / f"{name}_evaluation.md")
    return payload


def refresh_results_table(experiments_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(experiments_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"model": payload["model"]}
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = payload["cv"][metric]["mean"]
            row[f"{metric}_std"] = payload["cv"][metric]["std"]
        row["oof_brier"] = payload["oof_brier"]
        row["train_accuracy_full_refit"] = payload["train_accuracy_full_refit"]
        row["overfit_gap_accuracy"] = payload["overfit_gap_accuracy"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(experiments_dir / "results.csv", index=False)
    return table


def _write_model_note(payload: dict, path: Path) -> None:
    cv = payload["cv"]
    oof = payload["oof"]
    matrix = payload["confusion_matrix"]
    lines = [
        f"# {payload['model']}",
        "",
        "Training file only. Stratified folds. Engineered feature pipeline.",
        "The official test file was not used. Threshold for class labels is 0.5.",
        "",
        "| Metric | CV mean | CV std | Out-of-fold |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRIC_NAMES:
        lines.append(
            f"| {metric} | {cv[metric]['mean']:.4f} | {cv[metric]['std']:.4f} | {oof[metric]:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Out-of-fold Brier score: {payload['oof_brier']:.4f}.",
            f"Accuracy after a full training refit: {payload['train_accuracy_full_refit']:.4f}.",
            f"Accuracy gap (full refit minus CV mean): {payload['overfit_gap_accuracy']:.4f}.",
            "",
            "Out-of-fold confusion matrix, rows actual 0/1, columns predicted 0/1:",
            "",
            f"| | Pred 0 | Pred 1 |",
            f"| --- | ---: | ---: |",
            f"| Actual 0 | {matrix[0][0]} | {matrix[0][1]} |",
            f"| Actual 1 | {matrix[1][0]} | {matrix[1][1]} |",
            "",
            "This record does not select the final model.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
