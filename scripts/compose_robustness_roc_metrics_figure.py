#!/usr/bin/env python3
"""Render robustness ROC and metric figures for compound-disjoint and scaffold splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_fingerprint_models import ensure_matplotlib, save_figure_multi


METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F1Score", "AUC", "Sensitivity", "Specificity", "MCC"]
SPLIT_STYLES = {
    "compound_disjoint": {
        "label": "Compound-Disjoint",
        "line_color": "#6fa8dc",
        "bar_color": "#BAD5E7",
    },
    "scaffold": {
        "label": "Scaffold",
        "line_color": "#f08b78",
        "bar_color": "#f7c3b8",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose robustness ROC and metric plots.")
    parser.add_argument(
        "--compound-dir",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/robustness_EFM_AutoGluon_compound_disjoint"),
    )
    parser.add_argument(
        "--scaffold-dir",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/robustness_EFM_AutoGluon_scaffold"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/robustness_EFM_AutoGluon_summary"),
    )
    return parser.parse_args()


def load_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.read_csv(run_dir / "test_metrics.csv")
    roc_df = pd.read_csv(run_dir / "roc_curve_data.csv")
    return metrics_df, roc_df


def draw_roc(
    ax,
    roc_df: pd.DataFrame,
    *,
    title: str | None,
    line_color: str,
    legend_label: str = "AutoGluon",
) -> None:
    auc_val = float(roc_df["AUC"].iloc[0])
    ax.plot(
        roc_df["FPR"],
        roc_df["TPR"],
        color=line_color,
        linewidth=2.0,
        label=f"{legend_label} (AUC={auc_val:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.4)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", frameon=True)


def draw_metrics(ax, metrics_df: pd.DataFrame, *, title: str | None, bar_color: str) -> None:
    row = metrics_df.iloc[0]
    scores = [float(row[m]) for m in METRIC_ORDER]
    ax.bar(METRIC_ORDER, scores, color=bar_color, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def style_axes(ax, *, legend_bold: bool = False) -> None:
    ax.xaxis.label.set_fontsize(22.5)
    ax.yaxis.label.set_fontsize(22.5)
    ax.title.set_fontsize(17)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(18)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(12)
            if legend_bold:
                text.set_fontweight("bold")


def save_single_run_plots(output_dir: Path, prefix: str, metrics_df: pd.DataFrame, roc_df: pd.DataFrame, *, label: str, line_color: str, bar_color: str) -> None:
    plt = ensure_matplotlib()

    fig_roc, ax_roc = plt.subplots(figsize=(8.6, 6.2))
    draw_roc(ax_roc, roc_df, title=f"{label} ROC Curve", line_color=line_color, legend_label=label)
    style_axes(ax_roc, legend_bold=True)
    fig_roc.tight_layout()
    save_figure_multi(fig_roc, output_dir / f"{prefix}_roc_curve.png", dpi=300)
    plt.close(fig_roc)

    fig_bar, ax_bar = plt.subplots(figsize=(10.8, 5.8))
    draw_metrics(ax_bar, metrics_df, title=f"{label} Metrics", bar_color=bar_color)
    style_axes(ax_bar)
    fig_bar.tight_layout()
    save_figure_multi(fig_bar, output_dir / f"{prefix}_metrics_bar.png", dpi=300)
    plt.close(fig_bar)


def save_combined_figure(output_dir: Path, compound_metrics: pd.DataFrame, compound_roc: pd.DataFrame, scaffold_metrics: pd.DataFrame, scaffold_roc: pd.DataFrame) -> None:
    plt = ensure_matplotlib()
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], hspace=0.30, wspace=0.18)

    ax_cd_roc = fig.add_subplot(gs[0, 0])
    ax_cd_bar = fig.add_subplot(gs[0, 1])
    ax_sc_roc = fig.add_subplot(gs[1, 0])
    ax_sc_bar = fig.add_subplot(gs[1, 1])

    draw_roc(
        ax_cd_roc,
        compound_roc,
        title=None,
        line_color=SPLIT_STYLES["compound_disjoint"]["line_color"],
        legend_label="Compound-Disjoint AutoGluon",
    )
    draw_metrics(
        ax_cd_bar,
        compound_metrics,
        title=None,
        bar_color=SPLIT_STYLES["compound_disjoint"]["bar_color"],
    )
    draw_roc(
        ax_sc_roc,
        scaffold_roc,
        title=None,
        line_color=SPLIT_STYLES["scaffold"]["line_color"],
        legend_label="Scaffold AutoGluon",
    )
    draw_metrics(
        ax_sc_bar,
        scaffold_metrics,
        title=None,
        bar_color=SPLIT_STYLES["scaffold"]["bar_color"],
    )

    style_axes(ax_cd_roc, legend_bold=True)
    style_axes(ax_cd_bar)
    style_axes(ax_sc_roc, legend_bold=True)
    style_axes(ax_sc_bar)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.07)
    save_figure_multi(fig, output_dir / "robustness_roc_metrics_combined.png", dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    compound_dir = args.compound_dir.resolve()
    scaffold_dir = args.scaffold_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compound_metrics, compound_roc = load_inputs(compound_dir)
    scaffold_metrics, scaffold_roc = load_inputs(scaffold_dir)

    save_single_run_plots(
        output_dir,
        "compound_disjoint",
        compound_metrics,
        compound_roc,
        label=SPLIT_STYLES["compound_disjoint"]["label"],
        line_color=SPLIT_STYLES["compound_disjoint"]["line_color"],
        bar_color=SPLIT_STYLES["compound_disjoint"]["bar_color"],
    )
    save_single_run_plots(
        output_dir,
        "scaffold",
        scaffold_metrics,
        scaffold_roc,
        label=SPLIT_STYLES["scaffold"]["label"],
        line_color=SPLIT_STYLES["scaffold"]["line_color"],
        bar_color=SPLIT_STYLES["scaffold"]["bar_color"],
    )
    save_combined_figure(output_dir, compound_metrics, compound_roc, scaffold_metrics, scaffold_roc)

    for name in [
        "compound_disjoint_roc_curve.png",
        "compound_disjoint_metrics_bar.png",
        "scaffold_roc_curve.png",
        "scaffold_metrics_bar.png",
        "robustness_roc_metrics_combined.png",
    ]:
        png_path = output_dir / name
        print(png_path)
        print(png_path.with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
