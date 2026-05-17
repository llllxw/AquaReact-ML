#!/usr/bin/env python3
"""Compose IFS, ROC, and metrics bar charts into one multi-panel figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_fingerprint_models import ensure_matplotlib, save_figure_multi
from plot_boosted_ifs_curves import draw_ifs_all_models, load_ifs, merge_ifs
from plot_boosted_model_comparison import (
    draw_metrics_bar,
    draw_roc_curves,
    load_table,
    merge_metrics,
    merge_roc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose three-panel figure for E+F+M.")
    parser.add_argument("--base-run", type=Path, required=True, help="Original full-model run directory.")
    parser.add_argument("--boosted-run", type=Path, required=True, help="Boosted AutoGluon run directory.")
    parser.add_argument("--feature-set", default="E+F+M", help="Feature set name.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path. SVG will be written with the same stem. Defaults to feature set directory.",
    )
    return parser.parse_args()


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=32,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def scale_axis_fonts(ax, *, scale: float = 2.0, bold_legend: bool = False) -> None:
    title = ax.title
    title.set_fontsize(title.get_fontsize() * scale)

    ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * scale)
    ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * scale)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * scale)

    for text in ax.texts:
        if text.get_text() in {"a", "b", "c"}:
            continue
        text.set_fontsize(text.get_fontsize() * scale)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(text.get_fontsize() * scale)
            if bold_legend:
                text.set_fontweight("bold")


def match_title_to_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is None or not legend.get_texts():
        return
    legend_size = legend.get_texts()[0].get_fontsize()
    ax.title.set_fontsize(legend_size)


def main() -> int:
    args = parse_args()
    base_run = args.base_run.resolve()
    boosted_run = args.boosted_run.resolve()
    feature_set = args.feature_set

    base_ifs = load_ifs(base_run, feature_set)
    boosted_ifs = load_ifs(boosted_run, feature_set)
    merged_ifs = merge_ifs(base_ifs, boosted_ifs)

    metrics_base = load_table(base_run, feature_set, "test_metrics.csv")
    metrics_boosted = load_table(boosted_run, feature_set, "test_metrics.csv")
    roc_base = load_table(base_run, feature_set, "roc_curve_data.csv")
    roc_boosted = load_table(boosted_run, feature_set, "roc_curve_data.csv")
    merged_metrics = merge_metrics(metrics_base, metrics_boosted)
    merged_roc = merge_roc(roc_base, roc_boosted)

    output_path = args.output_path
    if output_path is None:
        output_path = boosted_run / "feature_sets" / feature_set / "combined_ifs_roc_metrics.png"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt = ensure_matplotlib()
    fig = plt.figure(figsize=(17.0, 14.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.36, wspace=0.28)

    ax_ifs = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    draw_ifs_all_models(ax_ifs, merged_ifs, feature_set, title="IFS Curve", show_legend=True)
    draw_roc_curves(ax_roc, merged_roc, feature_set, title="ROC Curves", show_legend=True)
    draw_metrics_bar(ax_bar, merged_metrics, feature_set, title="Model Comparison", show_legend=True)

    add_panel_label(ax_ifs, "a")
    add_panel_label(ax_roc, "b")
    add_panel_label(ax_bar, "c")

    scale_axis_fonts(ax_ifs, scale=2.0)
    scale_axis_fonts(ax_roc, scale=2.0, bold_legend=True)
    scale_axis_fonts(ax_bar, scale=2.0)

    ax_ifs.legend(loc="lower right", ncol=1, frameon=True, fontsize=14.4)
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=8, frameon=True, fontsize=18)

    match_title_to_legend(ax_ifs)
    match_title_to_legend(ax_roc)

    ax_ifs.set_title(ax_ifs.get_title(), pad=10)
    ax_roc.set_title(ax_roc.get_title(), pad=10)
    ax_bar.set_title(ax_bar.get_title(), pad=34)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08)
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)

    print(output_path)
    print(output_path.with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
