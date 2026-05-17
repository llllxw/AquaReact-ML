#!/usr/bin/env python3
"""Redraw pair-representation comparison plots and compose them into one figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_fingerprint_models import ensure_matplotlib, save_figure_multi


REP_ORDER = ["Concat", "AbsDiff", "Hadamard", "AllCombined"]
REP_COLORS = {
    "Concat": "#BAD5E7",
    "AbsDiff": "#969BC6",
    "Hadamard": "#FCD7B0",
    "AllCombined": "#E4AAA5",
}
METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F1Score", "AUC", "Sensitivity", "Specificity", "MCC"]
AXIS_LABEL_FONT = 21
AXIS_TICK_FONT = 18
TITLE_FONT = 18
LEGEND_FONT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose pair-representation comparison plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/pair_representation_fair_comparison"),
        help="Directory containing pair_representation_metrics.csv and pair_representation_roc_curve_data.csv",
    )
    parser.add_argument("--concat-color", default=None, help="Optional override for Concat color.")
    parser.add_argument("--filename-suffix", default="", help="Optional suffix appended to output filenames.")
    return parser.parse_args()


def load_inputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = output_dir / "pair_representation_metrics.csv"
    roc_path = output_dir / "pair_representation_roc_curve_data.csv"
    metrics_df = pd.read_csv(metrics_path)
    roc_df = pd.read_csv(roc_path)
    return metrics_df, roc_df


def prepare_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df["Representation"] = pd.Categorical(df["Representation"], categories=REP_ORDER, ordered=True)
    df = df.sort_values("Representation").reset_index(drop=True)
    return df


def prepare_roc(roc_df: pd.DataFrame) -> pd.DataFrame:
    df = roc_df.copy()
    df["Model"] = pd.Categorical(df["Model"], categories=REP_ORDER, ordered=True)
    df = df.sort_values(["Model", "FPR", "TPR"]).reset_index(drop=True)
    return df


def draw_metrics_bar(ax, metrics_df: pd.DataFrame, colors: dict[str, str], *, title: str | None = None, show_legend: bool = True) -> None:
    x = np.arange(len(METRIC_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(REP_ORDER))

    for offset, rep in zip(offsets, REP_ORDER):
        row = metrics_df.loc[metrics_df["Representation"] == rep]
        if row.empty:
            continue
        scores = [float(row.iloc[0][metric]) for metric in METRIC_ORDER]
        ax.bar(
            x + offset,
            scores,
            width=width,
            color=colors[rep],
            edgecolor="white",
            linewidth=0.5,
            label=rep,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title or "Pair Representation Comparison")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    if show_legend:
        ax.legend(loc="upper right", ncol=4, frameon=True)


def draw_roc_curves(ax, roc_df: pd.DataFrame, colors: dict[str, str], *, title: str | None = None, show_legend: bool = True) -> None:
    for rep in REP_ORDER:
        group = roc_df.loc[roc_df["Model"] == rep]
        if group.empty:
            continue
        auc_val = float(group["AUC"].iloc[0])
        ax.plot(
            group["FPR"],
            group["TPR"],
            linewidth=2.0,
            color=colors[rep],
            label=f"{rep} (AUC={auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.5)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "Pair Representation ROC Comparison")
    ax.grid(alpha=0.25, linestyle="--")
    if show_legend:
        ax.legend(loc="lower right", frameon=True)


def style_axes(ax) -> None:
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT)
    ax.title.set_fontsize(TITLE_FONT)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(AXIS_TICK_FONT)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_FONT)


def save_standalone_plots(metrics_df: pd.DataFrame, roc_df: pd.DataFrame, output_dir: Path, colors: dict[str, str], suffix: str) -> None:
    plt = ensure_matplotlib()

    fig_bar, ax_bar = plt.subplots(figsize=(14, 7))
    draw_metrics_bar(ax_bar, metrics_df, colors, title="Fair Pair Representation Comparison", show_legend=True)
    style_axes(ax_bar)
    fig_bar.tight_layout()
    save_figure_multi(fig_bar, output_dir / f"pair_representation_metrics_bar{suffix}.png", dpi=300)
    plt.close(fig_bar)

    fig_roc, ax_roc = plt.subplots(figsize=(10.5, 7.5))
    draw_roc_curves(ax_roc, roc_df, colors, title="Fair Pair Representation ROC Comparison", show_legend=True)
    style_axes(ax_roc)
    fig_roc.tight_layout()
    save_figure_multi(fig_roc, output_dir / f"pair_representation_roc_curves{suffix}.png", dpi=300)
    plt.close(fig_roc)


def save_combined_figure(metrics_df: pd.DataFrame, roc_df: pd.DataFrame, output_dir: Path, colors: dict[str, str], suffix: str) -> None:
    plt = ensure_matplotlib()
    fig = plt.figure(figsize=(22, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.18)

    ax_roc = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    draw_roc_curves(ax_roc, roc_df, colors, title="Fair Pair Representation ROC Comparison", show_legend=True)
    draw_metrics_bar(ax_bar, metrics_df, colors, title="Fair Pair Representation Comparison", show_legend=True)

    for ax in (ax_roc, ax_bar):
        style_axes(ax)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10)
    save_figure_multi(fig, output_dir / f"pair_representation_combined{suffix}.png", dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df, roc_df = load_inputs(output_dir)
    metrics_df = prepare_metrics(metrics_df)
    roc_df = prepare_roc(roc_df)
    colors = dict(REP_COLORS)
    if args.concat_color:
        colors["Concat"] = args.concat_color
    suffix = args.filename_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    save_standalone_plots(metrics_df, roc_df, output_dir, colors, suffix)
    save_combined_figure(metrics_df, roc_df, output_dir, colors, suffix)

    print(output_dir / f"pair_representation_metrics_bar{suffix}.png")
    print((output_dir / f"pair_representation_metrics_bar{suffix}.png").with_suffix(".svg"))
    print(output_dir / f"pair_representation_roc_curves{suffix}.png")
    print((output_dir / f"pair_representation_roc_curves{suffix}.png").with_suffix(".svg"))
    print(output_dir / f"pair_representation_combined{suffix}.png")
    print((output_dir / f"pair_representation_combined{suffix}.png").with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
