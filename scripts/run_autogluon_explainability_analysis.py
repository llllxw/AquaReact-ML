#!/usr/bin/env python3
"""Run explainability analysis for an AutoGluon fingerprint model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from train_fingerprint_models import (
    default_config,
    discover_feature_sets,
    ensure_matplotlib,
    load_full_feature_matrix,
    load_json,
    save_figure_multi,
    select_feature_columns,
)


SOURCE_META = {
    "E": {"name": "ECFP4", "start": 0, "end": 2047, "color": "#f08b78"},
    "F": {"name": "FCFP4", "start": 2048, "end": 6143, "color": "#6fa8dc"},
    "M": {"name": "MACCS", "start": 6144, "end": 6477, "color": "#c5de8b"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explainability analysis for AutoGluon.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory.")
    parser.add_argument("--feature-set", default="E+F+M", help="Feature set name.")
    parser.add_argument("--candidate-count", type=int, default=30, help="Top ranked used features to score.")
    parser.add_argument("--top-table-n", type=int, default=20, help="Number of top features in visualization table.")
    parser.add_argument("--subsample-size", type=int, default=200, help="Subsample size for permutation importance.")
    parser.add_argument("--num-shuffle-sets", type=int, default=1, help="Shuffle repeats for feature importance.")
    parser.add_argument("--time-limit", type=float, default=120.0, help="Time limit for feature importance.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to feature_sets/<feature_set>/explainability.",
    )
    return parser.parse_args()


def map_feature_source(feature_name: str) -> Dict[str, object]:
    idx = int(feature_name)
    for source_key, meta in SOURCE_META.items():
        if meta["start"] <= idx <= meta["end"]:
            return {
                "Feature": feature_name,
                "FeatureIndex": idx,
                "Source": source_key,
                "SourceName": meta["name"],
                "SourceLocalBit": idx - meta["start"],
                "SourceRange": f"{meta['start']}-{meta['end']}",
            }
    return {
        "Feature": feature_name,
        "FeatureIndex": idx,
        "Source": "Unknown",
        "SourceName": "Unknown",
        "SourceLocalBit": idx,
        "SourceRange": "",
    }


def load_context(run_dir: Path, feature_set: str):
    config = default_config()
    config_path = run_dir / "resolved_config.json"
    if config_path.exists():
        config.update(load_json(config_path))

    discovered = discover_feature_sets(Path(config["data_root"]))
    full_df = load_full_feature_matrix(discovered[feature_set])
    X_full, y_full = select_feature_columns(full_df)
    manifest = pd.read_csv(run_dir / "split_manifest.csv")
    test_idx = manifest.loc[manifest["split"] == "test", "row_id"].to_numpy(dtype=int)
    return X_full, y_full, test_idx


def load_predictor(feature_dir: Path):
    module = __import__("autogluon.tabular", fromlist=["TabularPredictor"])
    TabularPredictor = module.TabularPredictor
    return TabularPredictor.load(str(feature_dir / "autogluon_model"))


def prepare_candidate_features(feature_dir: Path, predictor, candidate_count: int) -> Dict[str, List[str]]:
    ranked = pd.read_csv(feature_dir / "ranked_features.csv")["feature"].astype(str).tolist()
    selected = pd.read_csv(feature_dir / "selected_features_by_model.csv")
    selected = selected[selected["Model"] == "AutoGluon"]["Feature"].astype(str).tolist()
    used = predictor.feature_metadata_in.get_features()

    used_set = set(used)
    selected_set = set(selected)
    candidate_features = [f for f in ranked if f in used_set and f in selected_set][:candidate_count]
    return {"ranked": ranked, "selected": selected, "used": used, "candidates": candidate_features}


def compute_feature_importance(
    predictor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidate_features: List[str],
    *,
    subsample_size: int,
    num_shuffle_sets: int,
    time_limit: float,
) -> pd.DataFrame:
    test_data = X_test.copy()
    test_data["target"] = y_test.to_numpy(dtype=int)
    fi_df = predictor.feature_importance(
        data=test_data,
        features=candidate_features,
        subsample_size=subsample_size,
        num_shuffle_sets=num_shuffle_sets,
        time_limit=time_limit,
        silent=True,
    ).reset_index().rename(columns={"index": "Feature"})
    return fi_df.sort_values("importance", ascending=False).reset_index(drop=True)


def enrich_importance_table(fi_df: pd.DataFrame, ranked: List[str]) -> pd.DataFrame:
    rank_map = {feat: idx + 1 for idx, feat in enumerate(ranked)}
    rows = []
    for _, row in fi_df.iterrows():
        mapped = map_feature_source(str(row["Feature"]))
        rows.append(
            {
                **mapped,
                "RankedFeatureOrder": rank_map.get(str(row["Feature"]), np.nan),
                "Importance": float(row["importance"]),
                "StdDev": float(row["stddev"]) if pd.notna(row["stddev"]) else np.nan,
                "PValue": float(row["p_value"]) if pd.notna(row["p_value"]) else np.nan,
                "ShuffleRepeats": int(row["n"]) if pd.notna(row["n"]) else np.nan,
            }
        )
    enriched = pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)
    return enriched


def build_contribution_summary(selected: List[str], used: List[str], importance_df: pd.DataFrame) -> pd.DataFrame:
    selected_meta = pd.DataFrame([map_feature_source(f) for f in selected])
    used_meta = pd.DataFrame([map_feature_source(f) for f in used])
    positive_importance = importance_df.copy()
    positive_importance["ImportanceClipped"] = positive_importance["Importance"].clip(lower=0.0)

    rows = []
    total_selected = len(selected_meta)
    total_used = len(used_meta)
    total_importance = float(positive_importance["ImportanceClipped"].sum()) or np.nan

    for source_key, meta in SOURCE_META.items():
        selected_count = int((selected_meta["Source"] == source_key).sum())
        used_count = int((used_meta["Source"] == source_key).sum())
        importance_sum = float(
            positive_importance.loc[positive_importance["Source"] == source_key, "ImportanceClipped"].sum()
        )
        rows.append(
            {
                "Source": source_key,
                "SourceName": meta["name"],
                "SelectedFeatureCount": selected_count,
                "SelectedFeatureShare": selected_count / total_selected if total_selected else np.nan,
                "ModelUsedFeatureCount": used_count,
                "ModelUsedFeatureShare": used_count / total_used if total_used else np.nan,
                "PositiveImportanceSum": importance_sum,
                "PositiveImportanceShare": importance_sum / total_importance if total_importance and not np.isnan(total_importance) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_global_importance_top(importance_df: pd.DataFrame, output_path: Path, *, top_n: int) -> None:
    plt = ensure_matplotlib()
    plot_df = importance_df.head(top_n).iloc[::-1].copy()
    colors = [SOURCE_META.get(src, {}).get("color", "#999999") for src in plot_df["Source"]]
    labels = [f"{src}-{bit}" for src, bit in zip(plot_df["Source"], plot_df["SourceLocalBit"])]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    bars = ax.barh(labels, plot_df["Importance"], color=colors, alpha=0.95)
    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Global Feature Importance (Top Features)")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    for bar, value in zip(bars, plot_df["Importance"]):
        ax.text(
            bar.get_width() + max(plot_df["Importance"].max(), 1e-6) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def plot_source_contribution(contrib_df: pd.DataFrame, output_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8))
    x = np.arange(len(contrib_df))
    colors = [SOURCE_META[row["Source"]]["color"] for _, row in contrib_df.iterrows()]

    for ax, col, title in [
        (axes[0], "ModelUsedFeatureShare", "Model-Used Feature Share"),
        (axes[1], "PositiveImportanceShare", "Positive Importance Share"),
    ]:
        vals = contrib_df[col].astype(float).fillna(0.0).to_numpy()
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(contrib_df["SourceName"])
        ax.set_ylim(0.0, max(vals.max() * 1.18, 0.4))
        ax.set_ylabel("Share")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ax.get_ylim()[1] * 0.03,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def plot_top_features_table(top_df: pd.DataFrame, output_path: Path) -> None:
    plt = ensure_matplotlib()
    table_df = top_df[
        ["Feature", "SourceName", "SourceLocalBit", "RankedFeatureOrder", "Importance"]
    ].copy()
    table_df["Importance"] = table_df["Importance"].map(lambda x: f"{x:.4f}")
    table_df["RankedFeatureOrder"] = table_df["RankedFeatureOrder"].astype(int)
    table_df.columns = ["Feature", "Source", "LocalBit", "RankOrder", "Importance"]

    fig_h = max(4.6, 0.42 * len(table_df) + 1.3)
    fig, ax = plt.subplots(figsize=(9.4, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values.tolist(),
        colLabels=table_df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#f3f4f6")
        else:
            source = table_df.iloc[row - 1]["Source"]
            if col == 1:
                bg = {
                    "ECFP4": "#fde2d8",
                    "FCFP4": "#dbeafe",
                    "MACCS": "#e8f5d0",
                }.get(source, "#ffffff")
                cell.set_facecolor(bg)
    ax.set_title("Top Feature Visualization Table", pad=12)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    feature_dir = run_dir / "feature_sets" / args.feature_set
    output_dir = args.output_dir.resolve() if args.output_dir else feature_dir / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(feature_dir)
    X_full, y_full, test_idx = load_context(run_dir, args.feature_set)
    context = prepare_candidate_features(feature_dir, predictor, args.candidate_count)
    used = context["used"]
    selected = context["selected"]
    candidates = context["candidates"]

    X_test = X_full.iloc[test_idx][used].reset_index(drop=True)
    y_test = y_full.iloc[test_idx].reset_index(drop=True)

    fi_raw = compute_feature_importance(
        predictor,
        X_test,
        y_test,
        candidates,
        subsample_size=args.subsample_size,
        num_shuffle_sets=args.num_shuffle_sets,
        time_limit=args.time_limit,
    )
    fi_enriched = enrich_importance_table(fi_raw, context["ranked"])
    fi_enriched.to_csv(output_dir / "global_feature_importance.csv", index=False)

    contrib_df = build_contribution_summary(selected, used, fi_enriched)
    contrib_df.to_csv(output_dir / "source_contribution_summary.csv", index=False)

    top_df = fi_enriched.head(args.top_table_n).copy()
    top_df.to_csv(output_dir / "top_features_table.csv", index=False)

    plot_global_importance_top(fi_enriched, output_dir / "global_feature_importance_top.png", top_n=min(20, len(fi_enriched)))
    plot_source_contribution(contrib_df, output_dir / "source_contribution.png")
    plot_top_features_table(top_df, output_dir / "top_features_table.png")

    metadata = pd.DataFrame(
        [
            {
                "FeatureSet": args.feature_set,
                "PredictorBestModel": predictor.model_best,
                "SelectedFeatureCount": len(selected),
                "ModelUsedFeatureCount": len(used),
                "CandidateCountScored": len(candidates),
                "TopTableN": min(args.top_table_n, len(top_df)),
                "SubsampleSize": args.subsample_size,
                "NumShuffleSets": args.num_shuffle_sets,
                "TimeLimit": args.time_limit,
            }
        ]
    )
    metadata.to_csv(output_dir / "explainability_metadata.csv", index=False)

    print(output_dir / "global_feature_importance.csv")
    print(output_dir / "source_contribution_summary.csv")
    print(output_dir / "top_features_table.csv")
    print(output_dir / "global_feature_importance_top.png")
    print((output_dir / "global_feature_importance_top.png").with_suffix(".svg"))
    print(output_dir / "source_contribution.png")
    print((output_dir / "source_contribution.png").with_suffix(".svg"))
    print(output_dir / "top_features_table.png")
    print((output_dir / "top_features_table.png").with_suffix(".svg"))
    print(output_dir / "explainability_metadata.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
