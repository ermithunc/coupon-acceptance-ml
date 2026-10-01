"""Build the model-comparison table from saved experiment records.

Reads training-fold results only. Does not refit models and does not read
the official test file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_loader import PROJECT_ROOT

EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"
COMPARISON_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"
COMPARISON_MD = PROJECT_ROOT / "reports" / "model_comparison.md"

UNTUNED_MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
)

# A second model is shortlisted for tuning when its ROC-AUC is within this
# gap of the leader. At most two models are tuned.
SHORTLIST_ROC_GAP = 0.02
MAX_SHORTLIST = 2


def load_untuned_records(experiments_dir: Path | None = None) -> pd.DataFrame:
    directory = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    rows = []
    for name in UNTUNED_MODELS:
        payload = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        row = {"model": name}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            row[f"{metric}_mean"] = payload["cv"][metric]["mean"]
            row[f"{metric}_std"] = payload["cv"][metric]["std"]
        row["oof_brier"] = payload["oof_brier"]
        row["overfit_gap_accuracy"] = payload["overfit_gap_accuracy"]
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def shortlist_models(table: pd.DataFrame) -> list[str]:
    """Leader by ROC-AUC, plus the next model when it is close."""
    ordered = table.sort_values("roc_auc_mean", ascending=False)
    names = ordered["model"].tolist()
    scores = ordered["roc_auc_mean"].tolist()
    chosen = [names[0]]
    if len(names) > 1 and (scores[0] - scores[1]) <= SHORTLIST_ROC_GAP and MAX_SHORTLIST > 1:
        chosen.append(names[1])
    return chosen


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def write_comparison(table: pd.DataFrame, csv_path: Path, md_path: Path) -> list[str]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    shortlist = shortlist_models(table)
    lines = [
        "# Model comparison",
        "",
        "Untuned models. Same training file, engineered features, stratified 5-fold protocol.",
        "The official test file was not used. This table does not by itself select the final model.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train-CV accuracy gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {model} | {acc} ± {acc_s} | {prec} ± {prec_s} | {rec} ± {rec_s} | {f1} ± {f1_s} | {roc} ± {roc_s} | {pr} ± {pr_s} | {brier} | {gap} |".format(
                model=row["model"],
                acc=_fmt(row["accuracy_mean"]),
                acc_s=_fmt(row["accuracy_std"]),
                prec=_fmt(row["precision_mean"]),
                prec_s=_fmt(row["precision_std"]),
                rec=_fmt(row["recall_mean"]),
                rec_s=_fmt(row["recall_std"]),
                f1=_fmt(row["f1_mean"]),
                f1_s=_fmt(row["f1_std"]),
                roc=_fmt(row["roc_auc_mean"]),
                roc_s=_fmt(row["roc_auc_std"]),
                pr=_fmt(row["pr_auc_mean"]),
                pr_s=_fmt(row["pr_auc_std"]),
                brier=_fmt(row["oof_brier"]),
                gap=_fmt(row["overfit_gap_accuracy"]),
            )
        )
    leader = table.iloc[0]
    tightest = table.sort_values("roc_auc_std").iloc[0]
    widest_recall = table.sort_values("recall_std", ascending=False).iloc[0]
    recall_leader = table.sort_values("recall_mean", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## Measured order",
            "",
            f"On ROC-AUC, PR-AUC, F1, accuracy, and Brier score, `{leader['model']}` is first in this untuned set.",
            f"`{recall_leader['model']}` has the highest recall ({recall_leader['recall_mean']:.4f}).",
            "Lower Brier is the better calibration score. Lower ROC-AUC standard deviation is the tighter fold spread.",
            "",
            "## Other factors",
            "",
            f"- Stability: `{tightest['model']}` has the smallest ROC-AUC fold spread ({tightest['roc_auc_std']:.4f}). `{widest_recall['model']}` has the widest recall spread ({widest_recall['recall_std']:.4f}).",
            "- Interpretability: logistic regression stays a linear baseline. The decision tree is one depth-capped tree. Random forest and XGBoost are ensembles.",
            "- Inference: each model scores one row through the same preprocessing pipeline on CPU.",
            "- Deployment: logistic regression, the tree, and the forest need scikit-learn. XGBoost also needs the xgboost package.",
            "- Overfit gap, full-refit accuracy minus CV accuracy: "
            + ", ".join(
                f"`{row.model}` {row.overfit_gap_accuracy:.4f}"
                for row in table.sort_values("overfit_gap_accuracy").itertuples()
            )
            + ".",
            "",
            "## Tuning shortlist",
            "",
            f"Rule: keep the ROC-AUC leader, and the next model when it is within {SHORTLIST_ROC_GAP:.2f} ROC-AUC. At most {MAX_SHORTLIST} models.",
            "",
            "Shortlist: " + ", ".join(f"`{name}`" for name in shortlist) + ".",
            "",
            "Final selection waits until that bounded search is scored on the same 5-fold protocol.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return shortlist
