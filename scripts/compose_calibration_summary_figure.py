#!/usr/bin/env python3
"""Compose one summary figure for AutoGluon calibration results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_fingerprint_models import ensure_matplotlib, save_figure_multi


METHOD_ORDER = ["Uncalibrated", "Platt", "Isotonic"]
METHOD_COLORS = {
    "Uncalibrated": "#f2cbb7",
    "Platt": "#d65244",
    "Isotonic": "#5977e3",
}
DISPLAY_NAME = {
    "Uncalibrated": "Uncalibrated",
    "Platt": "Platt",
    "Isotonic": "Isotonic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a calibration summary figure.")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/run_20260406_103701_autogluon_boosted/feature_sets/E+F+M/calibration"),
    )
    return parser.parse_args()


def load_inputs(calibration_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.read_csv(calibration_dir / "calibration_metrics_test.csv")
    prob_df = pd.read_csv(calibration_dir / "calibration_probabilities.csv")
    meta_df = pd.read_csv(calibration_dir / "calibration_run_metadata.csv")
    metrics_df["Method"] = pd.Categorical(metrics_df["Method"], categories=METHOD_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values("Method").reset_index(drop=True)
    return metrics_df, prob_df, meta_df


def compute_reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)
    mean_pred = []
    frac_pos = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        mean_pred.append(float(np.mean(y_prob[mask])))
        frac_pos.append(float(np.mean(y_true[mask])))
    return np.asarray(mean_pred), np.asarray(frac_pos)


def draw_calibration_curve(ax, prob_df: pd.DataFrame) -> None:
    test_df = prob_df.loc[prob_df["split"] == "test"].copy()
    y_true = test_df["y_true"].to_numpy(dtype=int)
    prob_cols = {
        "Uncalibrated": "prob_uncalibrated",
        "Platt": "prob_platt",
        "Isotonic": "prob_isotonic",
    }

    for method in METHOD_ORDER:
        mean_pred, frac_pos = compute_reliability_points(y_true, test_df[prob_cols[method]].to_numpy(dtype=float))
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=METHOD_COLORS[method],
            label=DISPLAY_NAME[method],
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.5, label="Ideal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.set_title("Calibration Curve")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=True)


def draw_score_bars(ax, metrics_df: pd.DataFrame, metric_name: str, title: str) -> None:
    values = metrics_df[metric_name].to_numpy(dtype=float)
    labels = [DISPLAY_NAME[m] for m in metrics_df["Method"]]
    colors = [METHOD_COLORS[m] for m in metrics_df["Method"]]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    upper = max(values) * 1.18 if max(values) > 0 else 1.0
    ax.set_ylim(0.0, upper)
    ax.tick_params(axis="x", rotation=12)


def draw_metric_table(ax, metrics_df: pd.DataFrame) -> None:
    ax.axis("off")
    display_df = metrics_df[["Method", "AUC", "BrierScore", "ECE", "LogLoss", "Accuracy", "MCC"]].copy()
    display_df["Method"] = display_df["Method"].map(DISPLAY_NAME)

    value_cols = ["AUC", "BrierScore", "ECE", "LogLoss", "Accuracy", "MCC"]
    table_vals = display_df[value_cols].to_numpy(dtype=float)

    # Normalize by column; for Brier/ECE/LogLoss smaller is better.
    norm_vals = np.zeros_like(table_vals)
    for idx, col in enumerate(value_cols):
        col_vals = table_vals[:, idx]
        if col in {"BrierScore", "ECE", "LogLoss"}:
            best = col_vals.min()
            worst = col_vals.max()
            if worst > best:
                norm_vals[:, idx] = (worst - col_vals) / (worst - best)
            else:
                norm_vals[:, idx] = 1.0
        else:
            best = col_vals.max()
            worst = col_vals.min()
            if best > worst:
                norm_vals[:, idx] = (col_vals - worst) / (best - worst)
            else:
                norm_vals[:, idx] = 1.0

    plt = ensure_matplotlib()
    cmap = plt.get_cmap("YlGnBu")
    cell_colours = []
    for row in norm_vals:
        color_row = []
        for val in row:
            rgba = cmap(0.15 + 0.65 * float(val))
            color_row.append(rgba)
        cell_colours.append(color_row)

    cell_text = [[f"{v:.3f}" for v in row] for row in table_vals]
    table = ax.table(
        cellText=cell_text,
        rowLabels=display_df["Method"].tolist(),
        colLabels=value_cols,
        cellColours=cell_colours,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.32)

    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f3f4f6")
        else:
            cell.set_text_props(color="#111827", fontweight="bold")

    ax.set_title("Key Metrics Summary", fontsize=16, pad=10)


def style_axes(ax, *, tick_scale: float = 1.0) -> None:
    if ax.get_title():
        ax.title.set_fontsize(18)
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12 * tick_scale)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(12)
            if "Ideal" not in text.get_text():
                text.set_fontweight("bold")


def main() -> int:
    args = parse_args()
    calibration_dir = args.calibration_dir.resolve()
    metrics_df, prob_df, meta_df = load_inputs(calibration_dir)

    plt = ensure_matplotlib()
    fig = plt.figure(figsize=(16.5, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.28)

    ax_curve = fig.add_subplot(gs[:, 0])
    right_top = gs[0, 1].subgridspec(1, 2, wspace=0.24)
    ax_brier = fig.add_subplot(right_top[0, 0])
    ax_ece = fig.add_subplot(right_top[0, 1])
    ax_table = fig.add_subplot(gs[1, 1])

    draw_calibration_curve(ax_curve, prob_df)
    draw_score_bars(ax_brier, metrics_df, "BrierScore", "Brier Score")
    draw_score_bars(ax_ece, metrics_df, "ECE", "ECE")
    draw_metric_table(ax_table, metrics_df)

    style_axes(ax_curve, tick_scale=1.5)
    style_axes(ax_brier)
    style_axes(ax_ece)

    feature_set = str(meta_df.loc[0, "FeatureSet"])
    model_name = str(meta_df.loc[0, "Model"])
    selected_k = int(meta_df.loc[0, "SelectedFeatureCount"])
    fig.suptitle(f"Calibration Summary - {feature_set} ({model_name}, k={selected_k})", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.06)

    output_path = calibration_dir / "calibration_summary.png"
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)

    print(output_path)
    print(output_path.with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
