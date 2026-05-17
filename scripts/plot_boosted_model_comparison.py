#!/usr/bin/env python3
"""Plot template-style model comparison figures using merged boosted AutoGluon results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_fingerprint_models import ensure_matplotlib, save_figure_multi


MODEL_ORDER = ["DT", "RF", "ExtraTrees", "KNN", "GB", "XGBoost", "CatBoost", "AutoGluon"]
BAR_MODEL_COLORS = {
    "DT": "#f4d7df",
    "RF": "#f08b78",
    "ExtraTrees": "#f7c3dc",
    "KNN": "#ffd54f",
    "GB": "#c5de8b",
    "XGBoost": "#f3c39b",
    "CatBoost": "#8ec7c5",
    "AutoGluon": "#6fa8dc",
}
MODEL_COLORS = {
    "DT": "#f4d7df",
    "RF": "#f08b78",
    "ExtraTrees": "#f7c3dc",
    "KNN": "#ffd54f",
    "GB": "#c5de8b",
    "XGBoost": "#f3c39b",
    "CatBoost": "#8ec7c5",
    "AutoGluon": "#6fa8dc",
}
METRIC_LABELS = {
    "Sensitivity": "Sn",
    "Specificity": "Sp",
    "Accuracy": "Acc",
    "MCC": "MCC",
    "AUC": "auROC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot template-style metrics bar and ROC curves.")
    parser.add_argument("--base-run", type=Path, required=True, help="Original full-model run directory.")
    parser.add_argument("--boosted-run", type=Path, required=True, help="Boosted AutoGluon run directory.")
    parser.add_argument("--feature-set", required=True, help="Feature set name, e.g. E+F+M.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merged files and figures. Defaults to boosted feature set directory.",
    )
    return parser.parse_args()


def load_table(run_dir: Path, feature_set: str, file_name: str) -> pd.DataFrame:
    path = run_dir / "feature_sets" / feature_set / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def merge_metrics(base_df: pd.DataFrame, boosted_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.copy()
    boosted_auto = boosted_df[boosted_df["Model"] == "AutoGluon"].copy()
    if boosted_auto.empty:
        raise RuntimeError("Boosted metrics do not contain AutoGluon.")
    merged = pd.concat([merged[merged["Model"] != "AutoGluon"], boosted_auto], ignore_index=True)
    merged["Model"] = pd.Categorical(merged["Model"], categories=MODEL_ORDER, ordered=True)
    merged = merged.sort_values("Model").reset_index(drop=True)
    return merged


def merge_roc(base_df: pd.DataFrame, boosted_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat(
        [base_df[base_df["Model"] != "AutoGluon"].copy(), boosted_df[boosted_df["Model"] == "AutoGluon"].copy()],
        ignore_index=True,
    )
    merged["Model"] = pd.Categorical(merged["Model"], categories=MODEL_ORDER, ordered=True)
    merged = merged.sort_values(["Model", "FPR", "TPR"]).reset_index(drop=True)
    return merged


def draw_metrics_bar(ax, metrics_df: pd.DataFrame, feature_set: str, *, title: str | None = None, show_legend: bool = True) -> None:
    metric_cols = ["Sensitivity", "Specificity", "Accuracy", "MCC", "AUC"]
    x_labels = [METRIC_LABELS[col] for col in metric_cols]

    x = range(len(metric_cols))
    width = 0.095
    offsets = [(-3.5 + i) * width for i in range(len(MODEL_ORDER))]

    for idx, model_name in enumerate(MODEL_ORDER):
        row = metrics_df[metrics_df["Model"] == model_name]
        if row.empty:
            continue
        values = row.iloc[0][metric_cols].astype(float).tolist()
        positions = [pos + offsets[idx] for pos in x]
        ax.bar(
            positions,
            values,
            width=width,
            label=model_name,
            color=BAR_MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.95,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title or f"Model Comparison - {feature_set}")
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8, ncol=1)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def plot_metrics_bar(metrics_df: pd.DataFrame, feature_set: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    draw_metrics_bar(ax, metrics_df, feature_set)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def draw_roc_curves(ax, roc_df: pd.DataFrame, feature_set: str, *, title: str | None = None, show_legend: bool = True) -> None:
    for model_name in MODEL_ORDER:
        group = roc_df[roc_df["Model"] == model_name]
        if group.empty:
            continue
        auc = float(group["AUC"].iloc[0])
        ax.plot(
            group["FPR"].astype(float),
            group["TPR"].astype(float),
            color=MODEL_COLORS[model_name],
            linewidth=2.0,
            label=f"{model_name} (AUC={auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.2)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or f"ROC Curves - {feature_set}")
    ax.grid(alpha=0.18, linestyle="--")
    if show_legend:
        ax.legend(loc="lower right", fontsize=7.5, frameon=True)


def plot_roc_curves(roc_df: pd.DataFrame, feature_set: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    draw_roc_curves(ax, roc_df, feature_set)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    base_run = args.base_run.resolve()
    boosted_run = args.boosted_run.resolve()

    metrics_base = load_table(base_run, args.feature_set, "test_metrics.csv")
    metrics_boosted = load_table(boosted_run, args.feature_set, "test_metrics.csv")
    roc_base = load_table(base_run, args.feature_set, "roc_curve_data.csv")
    roc_boosted = load_table(boosted_run, args.feature_set, "roc_curve_data.csv")

    merged_metrics = merge_metrics(metrics_base, metrics_boosted)
    merged_roc = merge_roc(roc_base, roc_boosted)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = boosted_run / "feature_sets" / args.feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_metrics.to_csv(output_dir / "test_metrics_merged.csv", index=False)
    merged_roc.to_csv(output_dir / "roc_curve_data_merged.csv", index=False)

    bar_png = output_dir / "model_metrics_bar_template.png"
    roc_png = output_dir / "roc_curves_template.png"
    plot_metrics_bar(merged_metrics, args.feature_set, bar_png)
    plot_roc_curves(merged_roc, args.feature_set, roc_png)

    print(output_dir / "test_metrics_merged.csv")
    print(output_dir / "roc_curve_data_merged.csv")
    print(bar_png)
    print(bar_png.with_suffix(".svg"))
    print(roc_png)
    print(roc_png.with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
