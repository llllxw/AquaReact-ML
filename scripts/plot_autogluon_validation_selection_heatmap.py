#!/usr/bin/env python3
"""Plot validation-based AutoGluon internal model ranking heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_fingerprint_models import ensure_matplotlib, save_figure_multi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation-based AutoGluon internal selection heatmap.")
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/run_20260406_103701_autogluon_boosted/best_feature_combination/autogluon_internal_validation_leaderboard.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/run_20260406_103701_autogluon_boosted/best_feature_combination/autogluon_internal_validation_heatmap.png"),
    )
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.leaderboard_csv).sort_values("score_val", ascending=False).head(args.top_n).copy()
    df = df.rename(
        columns={
            "model": "Model",
            "score_val": "ValidationAUC",
            "pred_time_val": "PredTimeVal",
            "fit_time": "FitTime",
            "stack_level": "StackLevel",
        }
    )

    plot_df = df.set_index("Model")[["ValidationAUC", "StackLevel"]]

    plt = ensure_matplotlib()
    fig_h = max(4.5, 0.62 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    im = ax.imshow(plot_df.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(1.0, float(plot_df.values.max())))

    ax.set_xticks(range(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns, rotation=20, ha="right")
    ax.set_yticks(range(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("AutoGluon Internal Model Selection (Validation)")

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = float(plot_df.iloc[i, j])
            txt = f"{value:.3f}" if j == 0 else f"{int(value)}"
            color = "white" if value > (plot_df.values.max() * 0.55) else "#1f2937"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=15)
    fig.tight_layout()
    save_figure_multi(fig, args.output_path, dpi=300)
    plt.close(fig)

    print(args.output_path)
    print(args.output_path.with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
