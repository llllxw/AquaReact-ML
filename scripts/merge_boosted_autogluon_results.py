#!/usr/bin/env python3
"""Merge boosted AutoGluon results into an existing full-model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_fingerprint_models import (
    plot_autogluon_internal_heatmap,
    plot_model_bar,
    plot_roc_curves,
    save_summary_tables,
    select_best_feature_set_for_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge boosted AutoGluon outputs into a full run.")
    parser.add_argument("--base-run", type=Path, required=True, help="Original run directory with all models.")
    parser.add_argument("--boosted-run", type=Path, required=True, help="Boosted AutoGluon run directory.")
    return parser.parse_args()


def load_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "tables" / "all_metrics_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics table: {path}")
    return pd.read_csv(path)


def load_roc(run_dir: Path, feature_set: str) -> pd.DataFrame:
    path = run_dir / "feature_sets" / feature_set / "roc_curve_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ROC data: {path}")
    return pd.read_csv(path)


def feature_order(df: pd.DataFrame) -> dict:
    return {name: idx for idx, name in enumerate(df["FeatureSet"].drop_duplicates().tolist())}


def model_order(df: pd.DataFrame) -> dict:
    return {name: idx for idx, name in enumerate(df["Model"].drop_duplicates().tolist())}


def sort_metrics(df: pd.DataFrame, feature_map: dict, model_map: dict) -> pd.DataFrame:
    ordered = df.copy()
    ordered["_feature_order"] = ordered["FeatureSet"].map(feature_map).fillna(10**9)
    ordered["_model_order"] = ordered["Model"].map(model_map).fillna(10**9)
    ordered = ordered.sort_values(["_feature_order", "_model_order", "FeatureSet", "Model"]).drop(
        columns=["_feature_order", "_model_order"]
    )
    return ordered.reset_index(drop=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    base_run = args.base_run.resolve()
    boosted_run = args.boosted_run.resolve()

    base_metrics = load_metrics(base_run)
    boosted_metrics = load_metrics(boosted_run)

    boosted_auto = boosted_metrics[boosted_metrics["Model"] == "AutoGluon"].copy()
    if boosted_auto.empty:
        raise RuntimeError("Boosted run does not contain AutoGluon results.")

    merged = pd.concat(
        [
            base_metrics[base_metrics["Model"] != "AutoGluon"].copy(),
            boosted_auto,
        ],
        ignore_index=True,
    )
    merged = sort_metrics(merged, feature_order(base_metrics), model_order(base_metrics))

    save_summary_tables(merged, base_run)

    best_feature_set, best_model, selection_basis = select_best_feature_set_for_plots(merged)
    best_dir = base_run / "best_feature_combination"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_metrics = merged[merged["FeatureSet"] == best_feature_set].copy().sort_values("AUC", ascending=False)
    best_metrics.to_csv(best_dir / "model_metrics.csv", index=False)
    write_json(
        best_dir / "best_selection.json",
        {
            "best_feature_set": best_feature_set,
            "best_model": best_model,
            "selection_basis": selection_basis,
        },
    )

    base_roc = load_roc(base_run, best_feature_set)
    boosted_roc = load_roc(boosted_run, best_feature_set)
    merged_roc = pd.concat(
        [
            base_roc[base_roc["Model"] != "AutoGluon"].copy(),
            boosted_roc[boosted_roc["Model"] == "AutoGluon"].copy(),
        ],
        ignore_index=True,
    )
    model_map = model_order(base_metrics)
    merged_roc["_model_order"] = merged_roc["Model"].map(model_map).fillna(10**9)
    merged_roc = merged_roc.sort_values(["_model_order", "FPR", "TPR"]).drop(columns=["_model_order"])
    merged_roc.to_csv(best_dir / "roc_curve_data.csv", index=False)

    plot_model_bar(best_metrics, best_dir / "model_metrics_bar.png", title=f"Model Comparison - {best_feature_set}")
    plot_roc_curves(merged_roc, best_dir / "roc_curves.png", title=f"ROC Curves - {best_feature_set}")

    internal_csv = best_dir / "autogluon_internal_metrics.csv"
    internal_png = best_dir / "autogluon_internal_heatmap.png"
    if best_model == "AutoGluon":
        boosted_selection_path = boosted_run / "best_feature_combination" / "best_selection.json"
        boosted_selection = json.loads(boosted_selection_path.read_text(encoding="utf-8"))
        if boosted_selection.get("best_feature_set") == best_feature_set:
            boosted_internal = boosted_run / "best_feature_combination" / "autogluon_internal_metrics.csv"
            if boosted_internal.exists():
                internal_df = pd.read_csv(boosted_internal)
                internal_df.to_csv(internal_csv, index=False)
                plot_autogluon_internal_heatmap(
                    internal_df,
                    internal_png,
                    title="AutoGluon Internal Models Metrics",
                )
    else:
        if internal_csv.exists():
            internal_csv.unlink()
        if internal_png.exists():
            internal_png.unlink()

    print(f"Updated tables in: {base_run / 'tables'}")
    print(f"Updated best feature combination in: {best_dir}")
    print(f"Best feature set/model after merge: {best_feature_set} / {best_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
