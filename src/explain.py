"""SHAP explanations for the selected tuned XGBoost model.

Fits on the training file only. Statements describe associations with the
model's predicted probability, not causes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_loader import PROJECT_ROOT, split_xy
from src.experiment import make_model_pipeline
from src.tuning import estimator_from_search

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "shap_explanations.md"
SEARCH_PATH = PROJECT_ROOT / "reports" / "tuning" / "xgboost_search.json"


def fit_selected_model(train_df: pd.DataFrame):
    import json

    best_params = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))["best_params"]
    pipeline = make_model_pipeline(estimator_from_search("xgboost", best_params))
    X, y = split_xy(train_df)
    pipeline.fit(X, y)
    return pipeline, X, y


def _clean_name(name: str) -> str:
    text = str(name)
    if "__" in text:
        text = text.split("__", 1)[1]
    return text


def explain(train_df: pd.DataFrame, sample_size: int = 400, random_state: int = 42) -> dict:
    pipeline, X, y = fit_selected_model(train_df)
    sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    sample_y = y.loc[sample.index]
    matrix = pipeline.named_steps["prep"].transform(sample)
    model = pipeline.named_steps["model"]
    raw_names = pipeline.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    feature_names = [_clean_name(name) for name in raw_names]
    if len(feature_names) != matrix.shape[1]:
        raise RuntimeError(
            f"Feature name count {len(feature_names)} does not match matrix width {matrix.shape[1]}."
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    probabilities = model.predict_proba(matrix)[:, 1]
    high_pos = int(np.argmax(probabilities))
    low_pos = int(np.argmin(probabilities))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    shap.summary_plot(
        shap_values,
        matrix,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        mean_abs[order][::-1],
        color="#3d5a80",
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global association with predicted acceptance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_bar.png", dpi=120)
    plt.close(fig)

    _waterfall(explainer, shap_values, matrix, feature_names, high_pos, FIGURES_DIR / "shap_local_high.png")
    _waterfall(explainer, shap_values, matrix, feature_names, low_pos, FIGURES_DIR / "shap_local_low.png")

    top_global = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in order[:10]
    ]
    report = _report_text(
        sample=sample,
        sample_y=sample_y,
        probabilities=probabilities,
        shap_values=shap_values,
        feature_names=feature_names,
        matrix=matrix,
        high_pos=high_pos,
        low_pos=low_pos,
        top_global=top_global,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    return {"top_global": top_global, "n_sample": int(len(sample))}


def _waterfall(explainer, shap_values, matrix, feature_names, row_pos: int, path: Path) -> None:
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.asarray(expected).reshape(-1)[-1]
    explanation = shap.Explanation(
        values=shap_values[row_pos],
        base_values=float(expected),
        data=matrix[row_pos],
        feature_names=feature_names,
    )
    plt.close("all")
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _local_lines(shap_row, names, values) -> list[str]:
    order = np.argsort(np.abs(shap_row))[::-1][:5]
    lines = []
    for index in order:
        direction = "higher" if shap_row[index] > 0 else "lower"
        shown = values[index]
        if isinstance(shown, float):
            shown = f"{shown:.3f}"
        lines.append(
            f"- `{names[index]}` (value {shown}) was associated with a {direction} predicted acceptance probability (SHAP {shap_row[index]:+.3f})."
        )
    return lines


def _profile(row: pd.Series, label, probability: float) -> str:
    return (
        f"Age {row['age']}, {row['gender']}, destination {row['destination']}, "
        f"passanger {row['passanger']}, coupon {row['coupon']}, time {row['time']}, "
        f"weather {row['weather']}, expiration {row['expiration']}. "
        f"Predicted acceptance probability {probability:.3f}. "
        f"Recorded training label {int(label)}."
    )


def _report_text(
    sample,
    sample_y,
    probabilities,
    shap_values,
    feature_names,
    matrix,
    high_pos,
    low_pos,
    top_global,
) -> str:
    lines = [
        "# SHAP explanations",
        "",
        "Selected model: tuned XGBoost, fit on the training file. SHAP values below use a sample of training rows passed through the same preprocessing pipeline.",
        "",
        "These are associations with the model's predicted acceptance probability. They are not causes of coupon acceptance.",
        "Local values are the preprocessed inputs the model sees, including one-hot flags and scaled numbers.",
        "",
        "## Global features",
        "",
        "Mean absolute SHAP on the sample. Larger values mean the feature moved the prediction more, in either direction.",
        "",
        "| Feature | Mean absolute SHAP |",
        "| --- | ---: |",
    ]
    for item in top_global:
        lines.append(f"| {item['feature']} | {item['mean_abs_shap']:.4f} |")
    lines.extend(
        [
            "",
            "![SHAP summary](figures/shap_summary.png)",
            "",
            "![Mean absolute SHAP](figures/shap_bar.png)",
            "",
            "## Higher predicted probability",
            "",
            _profile(sample.iloc[high_pos], sample_y.iloc[high_pos], float(probabilities[high_pos])),
            "",
        ]
    )
    lines.extend(_local_lines(shap_values[high_pos], feature_names, matrix[high_pos]))
    lines.extend(
        [
            "",
            "![Local SHAP, higher probability](figures/shap_local_high.png)",
            "",
            "## Lower predicted probability",
            "",
            _profile(sample.iloc[low_pos], sample_y.iloc[low_pos], float(probabilities[low_pos])),
            "",
        ]
    )
    lines.extend(_local_lines(shap_values[low_pos], feature_names, matrix[low_pos]))
    lines.extend(
        [
            "",
            "![Local SHAP, lower probability](figures/shap_local_low.png)",
            "",
        ]
    )
    return "\n".join(lines)
