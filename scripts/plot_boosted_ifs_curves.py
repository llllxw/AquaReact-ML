#!/usr/bin/env python3
"""Plot merged IFS curves using a full run plus boosted AutoGluon results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from train_fingerprint_models import ensure_matplotlib, save_figure_multi
from plot_boosted_model_comparison import MODEL_COLORS


MODEL_ORDER = ["DT", "RF", "ExtraTrees", "KNN", "GB", "XGBoost", "CatBoost", "AutoGluon"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot merged IFS curves for a feature set.")
    parser.add_argument("--base-run", type=Path, required=True, help="Original full-model run directory.")
    parser.add_argument("--boosted-run", type=Path, required=True, help="Boosted AutoGluon run directory.")
    parser.add_argument("--feature-set", required=True, help="Feature set name, e.g. E+F+M.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merged CSV and plots. Defaults to boosted feature set directory.",
    )
    return parser.parse_args()


def load_ifs(run_dir: Path, feature_set: str) -> pd.DataFrame:
    path = run_dir / "feature_sets" / feature_set / "ifs_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing IFS results: {path}")
    return pd.read_csv(path)


def merge_ifs(base_df: pd.DataFrame, boosted_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.copy()
    if "AutoGluon" not in boosted_df.columns:
        raise RuntimeError("Boosted IFS results do not contain AutoGluon.")
    auto = boosted_df[["FeatureCount", "AutoGluon"]].copy()
    merged = merged.drop(columns=["AutoGluon"], errors="ignore").merge(auto, on="FeatureCount", how="outer")
    columns = ["FeatureCount"] + [col for col in MODEL_ORDER if col in merged.columns]
    merged = merged[columns].sort_values("FeatureCount").reset_index(drop=True)
    return merged


def best_point(df: pd.DataFrame, model_name: str) -> tuple[int, float]:
    valid = df[["FeatureCount", model_name]].dropna()
    best_idx = valid[model_name].astype(float).idxmax()
    return int(valid.loc[best_idx, "FeatureCount"]), float(valid.loc[best_idx, model_name])


def draw_ifs_all_models(ax, df: pd.DataFrame, feature_set: str, *, title: str | None = None, show_legend: bool = True) -> None:
    for model_name in MODEL_ORDER:
        if model_name not in df.columns:
            continue
        valid = df[["FeatureCount", model_name]].dropna()
        if valid.empty:
            continue
        ax.plot(
            valid["FeatureCount"],
            valid[model_name],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=MODEL_COLORS[model_name],
            label=model_name,
        )

    best_x, best_y = best_point(df, "AutoGluon")
    x_max = int(df["FeatureCount"].max())
    ax.axvline(best_x, color="0.5", linestyle=(0, (3, 3)), linewidth=1.0, alpha=0.9)
    ax.scatter([best_x], [best_y], color="black", s=28, zorder=6)
    ax.annotate(
        f"[{best_x} {best_y:.3f}]",
        (best_x, best_y),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=9,
    )

    ax.set_title(title or f"IFS Curve - {feature_set} (8 Models)")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Validation AUC")
    ax.set_xlim(0, x_max + 350)
    ax.grid(alpha=0.25, linestyle="--")

    if show_legend:
        ax.legend(loc="lower right", ncol=2, fontsize=9, frameon=True)


def plot_all_models(df: pd.DataFrame, feature_set: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    draw_ifs_all_models(ax, df, feature_set)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def plot_autogluon_only(df: pd.DataFrame, feature_set: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    valid = df[["FeatureCount", "AutoGluon"]].dropna()
    best_x, best_y = best_point(df, "AutoGluon")
    x_max = int(valid["FeatureCount"].max())

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(valid["FeatureCount"], valid["AutoGluon"], color="skyblue", linewidth=2.0)
    ax.axvline(best_x, color="0.5", linestyle=(0, (3, 3)), linewidth=1.0, alpha=0.9)
    ax.scatter([best_x], [best_y], color="black", s=28, zorder=6)
    ax.annotate(
        f"[{best_x} {best_y:.3f}]",
        (best_x, best_y),
        textcoords="offset points",
        xytext=(-42, 10),
        fontsize=9,
    )

    margin = 0.0012
    ax.set_xlim(valid["FeatureCount"].min() - 300, x_max + 300)
    ax.set_ylim(valid["AutoGluon"].min() - margin, valid["AutoGluon"].max() + margin)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_title(f"IFS Curve - {feature_set} (AutoGluon)")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Validation AUC")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    base_df = load_ifs(args.base_run.resolve(), args.feature_set)
    boosted_df = load_ifs(args.boosted_run.resolve(), args.feature_set)
    merged_df = merge_ifs(base_df, boosted_df)

    if "AutoGluon" not in merged_df.columns or merged_df["AutoGluon"].dropna().empty:
        raise RuntimeError("Merged IFS results do not contain AutoGluon values.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.boosted_run.resolve() / "feature_sets" / args.feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(output_dir / "ifs_results_merged.csv", index=False)
    plot_all_models(merged_df, args.feature_set, output_dir / "ifs_curve_all_models.png")
    plot_autogluon_only(merged_df, args.feature_set, output_dir / "ifs_curve_autogluon_only.png")

    print(output_dir / "ifs_results_merged.csv")
    print(output_dir / "ifs_curve_all_models.png")
    print((output_dir / "ifs_curve_all_models.png").with_suffix(".svg"))
    print(output_dir / "ifs_curve_autogluon_only.png")
    print((output_dir / "ifs_curve_autogluon_only.png").with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
