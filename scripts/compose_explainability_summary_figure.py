#!/usr/bin/env python3
"""Compose one summary figure for E+F+M explainability results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_fingerprint_models import ensure_matplotlib, save_figure_multi


SOURCE_COLORS = {
    "E": "#f57c6e",
    "F": "#71b7ed",
    "M": "#6cb889",
}
SOURCE_ALPHA = {
    "E": 1.0,
    "F": 1.0,
    "M": 1.0,
}
SHARE_COLORS = {
    "SelectedFeatureShare": "#d9e2ef",
    "ModelUsedFeatureShare": "#9fb9d6",
    "PositiveImportanceShare": "#f3b38f",
}
AXIS_LABEL_SCALE = 1.3
TICK_LABEL_SCALE = 1.3
ANNOTATION_SCALE = 1.3
SHARE_LABELS = {
    "SelectedFeatureShare": "Selected Share",
    "ModelUsedFeatureShare": "Model-Used Share",
    "PositiveImportanceShare": "Positive Importance Share",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose explainability summary figure.")
    parser.add_argument(
        "--explainability-dir",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/run_20260406_103701_autogluon_boosted/feature_sets/E+F+M/explainability"),
    )
    return parser.parse_args()


def load_inputs(explainability_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance_df = pd.read_csv(explainability_dir / "global_feature_importance.csv")
    source_df = pd.read_csv(explainability_dir / "source_contribution_summary.csv")
    top_df = pd.read_csv(explainability_dir / "top_features_table.csv")
    meta_df = pd.read_csv(explainability_dir / "explainability_metadata.csv")
    return importance_df, source_df, top_df, meta_df


def make_feature_label(row: pd.Series) -> str:
    return f"{row['Source']}-{int(row['SourceLocalBit'])}"


def draw_importance_bar(ax, importance_df: pd.DataFrame, top_n: int = 15) -> None:
    plot_df = importance_df.head(top_n).copy().iloc[::-1]
    labels = plot_df.apply(make_feature_label, axis=1)
    colors = [SOURCE_COLORS[src] for src in plot_df["Source"]]
    bars = ax.barh(labels, plot_df["Importance"], color=colors, edgecolor="white", linewidth=0.6)
    for bar, src in zip(bars, plot_df["Source"]):
        bar.set_alpha(SOURCE_ALPHA[src])
    for bar, value in zip(bars, plot_df["Importance"]):
        ax.text(
            value + plot_df["Importance"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=11 * ANNOTATION_SCALE,
            fontweight="bold",
        )

    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("Top Features")
    ax.set_title(f"Global Importance (Top {top_n})")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_xlim(0.0, float(plot_df["Importance"].max()) * 1.18)


def draw_share_comparison(ax, source_df: pd.DataFrame) -> None:
    plot_df = source_df.copy()
    x = np.arange(len(plot_df))
    width = 0.22
    metric_cols = ["SelectedFeatureShare", "ModelUsedFeatureShare", "PositiveImportanceShare"]
    offsets = [-width, 0.0, width]

    for offset, col in zip(offsets, metric_cols):
        vals = plot_df[col].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=SHARE_COLORS[col],
            edgecolor="white",
            linewidth=0.6,
            label=SHARE_LABELS[col],
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10 * ANNOTATION_SCALE,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["SourceName"])
    ax.set_ylim(0.0, 0.76)
    ax.set_ylabel("Share")
    ax.set_title("E / F / M Contribution Comparison")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=True)
    ax.set_box_aspect(1 / 1.2)


def style_axes(ax) -> None:
    if ax.get_title():
        ax.title.set_fontsize(18)
    ax.xaxis.label.set_fontsize(15 * AXIS_LABEL_SCALE)
    ax.yaxis.label.set_fontsize(15 * AXIS_LABEL_SCALE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12 * TICK_LABEL_SCALE)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(11 * TICK_LABEL_SCALE)


def main() -> int:
    args = parse_args()
    explainability_dir = args.explainability_dir.resolve()
    importance_df, source_df, top_df, meta_df = load_inputs(explainability_dir)

    plt = ensure_matplotlib()
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.24)

    ax_importance = fig.add_subplot(gs[0, 0])
    ax_share = fig.add_subplot(gs[0, 1])

    draw_importance_bar(ax_importance, importance_df, top_n=15)
    draw_share_comparison(ax_share, source_df)

    style_axes(ax_importance)
    style_axes(ax_share)

    feature_set = str(meta_df.loc[0, "FeatureSet"])
    fig.suptitle(f"Explainability Summary - {feature_set} + AutoGluon", fontsize=21, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.10)

    output_path = explainability_dir / "explainability_summary.png"
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)

    print(output_path)
    print(output_path.with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
