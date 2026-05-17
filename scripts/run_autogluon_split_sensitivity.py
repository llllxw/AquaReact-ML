#!/usr/bin/env python3
"""Robustness experiments for the best E+F+M + AutoGluon setting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from train_fingerprint_models import (
    compute_metrics,
    discover_feature_sets,
    evaluate_autogluon_internal_models,
    fit_predict_autogluon_with_retry,
    load_full_feature_matrix,
    plot_autogluon_internal_heatmap,
    plot_roc_curves,
    select_feature_columns,
)


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class GroupSelection:
    groups: Set[str]
    total: int
    positive: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaffold/compound-disjoint sensitivity for E+F+M + AutoGluon.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/xwl/药物禁忌/元数据"))
    parser.add_argument("--feature-set", default="E+F+M")
    parser.add_argument(
        "--split-strategy",
        required=True,
        choices=["scaffold", "compound-disjoint"],
        help="Robustness split strategy.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6478, help="Number of top features to keep.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autogluon-config", type=Path, default=Path("/home/xwl/药物禁忌/configs/autogluon_boosted.json"))
    parser.add_argument("--autogluon-presets", default=None, help="Override AutoGluon presets.")
    parser.add_argument("--autogluon-time-limit", type=int, default=None, help="Override AutoGluon time limit in seconds.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build the split and save manifests without training.")
    parser.add_argument("--subset-trials", type=int, default=1200, help="Trials for approximate group subset search.")
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_pair_metadata(data_root: Path) -> pd.DataFrame:
    combined_path = data_root / "combined-data.csv"
    meta = pd.read_csv(combined_path)
    if meta.shape[1] < 2:
        raise RuntimeError(f"Expected at least two columns in {combined_path}")
    meta = meta.iloc[:, :2].copy()
    meta.columns = ["smiles_a", "smiles_b"]
    meta["smiles_a"] = meta["smiles_a"].astype(str).str.strip()
    meta["smiles_b"] = meta["smiles_b"].astype(str).str.strip()
    return meta


def import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        conda_prefix = Path(os.environ.get("CONDA_PREFIX", "/home/xwl/miniconda3"))
        candidate_paths = [
            conda_prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
            Path("/home/xwl/miniconda3/lib/python3.12/site-packages"),
        ]
        for candidate in candidate_paths:
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.append(candidate_str)
                try:
                    from rdkit import Chem
                    from rdkit.Chem.Scaffolds import MurckoScaffold
                    return Chem, MurckoScaffold
                except Exception:
                    continue
        raise ModuleNotFoundError(
            "RDKit is required for scaffold split. This machine already has RDKit under "
            "/home/xwl/miniconda3/lib/python3.12/site-packages, so you can run with "
            "`PYTHONPATH=/home/xwl/miniconda3/lib/python3.12/site-packages` if needed."
        )
    return Chem, MurckoScaffold


def murcko_key(smiles: str, chem_module, murcko_module) -> str:
    mol = chem_module.MolFromSmiles(smiles)
    if mol is None:
        return f"INVALID::{smiles}"
    scaffold = murcko_module.MurckoScaffoldSmiles(mol=mol)
    if scaffold:
        return scaffold
    # For acyclic compounds, fall back to canonical SMILES so they do not collapse into one giant scaffold bucket.
    return f"ACYCLIC::{chem_module.MolToSmiles(mol, canonical=True)}"


def build_scaffold_groups(meta: pd.DataFrame) -> np.ndarray:
    chem_module, murcko_module = import_rdkit()
    unique_smiles = pd.Index(pd.unique(pd.concat([meta["smiles_a"], meta["smiles_b"]], ignore_index=True)))
    scaffold_map = {smiles: murcko_key(smiles, chem_module, murcko_module) for smiles in unique_smiles}
    scaffold_pairs = []
    for smi_a, smi_b in zip(meta["smiles_a"], meta["smiles_b"]):
        sa = scaffold_map[smi_a]
        sb = scaffold_map[smi_b]
        scaffold_pairs.append("||".join(sorted((sa, sb))))
    return np.asarray(scaffold_pairs, dtype=object)


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1
            return item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def build_component_groups(meta: pd.DataFrame, key_cols: Sequence[str]) -> np.ndarray:
    uf = UnionFind()
    for left, right in zip(meta[key_cols[0]], meta[key_cols[1]]):
        uf.union(str(left), str(right))
    groups = np.asarray([uf.find(str(item)) for item in meta[key_cols[0]]], dtype=object)
    return groups


def summarize_groups(labels: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"group": groups, "label": labels, "row_index": np.arange(len(labels))})
    summary = (
        df.groupby("group", sort=False)
        .agg(total=("label", "size"), positive=("label", "sum"), row_index=("row_index", list))
        .reset_index()
    )
    summary["negative"] = summary["total"] - summary["positive"]
    return summary


def partition_score(total: int, positive: int, target_total: float, target_positive: float) -> float:
    negative = total - positive
    target_negative = target_total - target_positive
    size_err = abs(total - target_total) / max(target_total, 1.0)
    pos_err = abs(positive - target_positive) / max(target_positive, 1.0)
    neg_err = abs(negative - target_negative) / max(target_negative, 1.0)
    return float(size_err + 0.8 * pos_err + 0.8 * neg_err)


def choose_group_subset(
    group_df: pd.DataFrame,
    target_total: float,
    target_positive: float,
    seed: int,
    n_trials: int,
) -> GroupSelection:
    rng = np.random.default_rng(seed)
    group_df = group_df.copy().reset_index(drop=True)
    stats = {str(row["group"]): row for _, row in group_df.iterrows()}
    groups = list(stats.keys())
    total_all = int(group_df["total"].sum())
    positive_all = int(group_df["positive"].sum())
    best: Optional[GroupSelection] = None

    for trial in range(max(n_trials, 50)):
        if trial % 3 == 0:
            ordered = sorted(groups, key=lambda g: (-int(stats[g]["total"]), rng.random()))
        elif trial % 3 == 1:
            ordered = sorted(groups, key=lambda g: (-int(stats[g]["positive"]), rng.random()))
        else:
            ordered = groups[:]
            rng.shuffle(ordered)

        selected: Set[str] = set()
        total = 0
        positive = 0
        current_score = partition_score(total, positive, target_total, target_positive)

        for group in ordered:
            row = stats[group]
            next_total = total + int(row["total"])
            next_positive = positive + int(row["positive"])
            next_score = partition_score(next_total, next_positive, target_total, target_positive)
            if next_score < current_score:
                selected.add(group)
                total = next_total
                positive = next_positive
                current_score = next_score
                continue
            if total < target_total * 0.75 and next_total <= max(target_total * 1.35, target_total + 8):
                if rng.random() < 0.06:
                    selected.add(group)
                    total = next_total
                    positive = next_positive
                    current_score = next_score

        improved = True
        while improved:
            improved = False
            for group in list(selected):
                row = stats[group]
                next_total = total - int(row["total"])
                next_positive = positive - int(row["positive"])
                next_score = partition_score(next_total, next_positive, target_total, target_positive)
                if next_score < current_score:
                    selected.remove(group)
                    total = next_total
                    positive = next_positive
                    current_score = next_score
                    improved = True

        for group in ordered:
            if group in selected:
                continue
            row = stats[group]
            next_total = total + int(row["total"])
            next_positive = positive + int(row["positive"])
            next_score = partition_score(next_total, next_positive, target_total, target_positive)
            if next_score < current_score:
                selected.add(group)
                total = next_total
                positive = next_positive
                current_score = next_score

        negative = total - positive
        remaining_total = total_all - total
        remaining_positive = positive_all - positive
        remaining_negative = remaining_total - remaining_positive
        if min(total, positive, negative, remaining_total, remaining_positive, remaining_negative) <= 0:
            continue

        candidate = GroupSelection(groups=selected, total=total, positive=positive, score=current_score)
        if best is None or candidate.score < best.score:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to construct a valid grouped split with both classes in each partition.")
    return best


def grouped_train_val_test_split(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    validation_size: float,
    seed: int,
    subset_trials: int,
) -> Tuple[SplitData, dict]:
    summary = summarize_groups(labels, groups)
    total_n = len(labels)
    total_pos = int(labels.sum())

    target_test_total = total_n * test_size
    target_test_pos = total_pos * test_size
    test_selection = choose_group_subset(summary, target_test_total, target_test_pos, seed, subset_trials)

    remaining = summary[~summary["group"].isin(test_selection.groups)].copy().reset_index(drop=True)
    target_val_total = remaining["total"].sum() * validation_size
    target_val_pos = remaining["positive"].sum() * validation_size
    val_selection = choose_group_subset(remaining, target_val_total, target_val_pos, seed + 1, subset_trials)

    test_mask = pd.Series(groups).isin(test_selection.groups).to_numpy()
    val_mask = pd.Series(groups).isin(val_selection.groups).to_numpy() & ~test_mask
    train_mask = ~(test_mask | val_mask)

    split = SplitData(
        train_idx=np.flatnonzero(train_mask),
        val_idx=np.flatnonzero(val_mask),
        test_idx=np.flatnonzero(test_mask),
    )
    split_summary = {
        "n_total": int(total_n),
        "n_positive": int(total_pos),
        "train_n": int(train_mask.sum()),
        "train_positive": int(labels[train_mask].sum()),
        "val_n": int(val_mask.sum()),
        "val_positive": int(labels[val_mask].sum()),
        "test_n": int(test_mask.sum()),
        "test_positive": int(labels[test_mask].sum()),
        "n_groups": int(summary.shape[0]),
        "test_groups": len(test_selection.groups),
        "val_groups": len(val_selection.groups),
    }
    return split, split_summary


def build_split_manifest(meta: pd.DataFrame, labels: np.ndarray, groups: np.ndarray, split: SplitData) -> pd.DataFrame:
    assignments = np.full(len(labels), "train", dtype=object)
    assignments[split.val_idx] = "validation"
    assignments[split.test_idx] = "test"
    manifest = meta.copy()
    manifest["label"] = labels
    manifest["group_id"] = groups
    manifest["split"] = assignments
    manifest["row_index"] = np.arange(len(labels))
    cols = ["row_index", "split", "label", "group_id", "smiles_a", "smiles_b"]
    return manifest[cols]


def overlap_summary(manifest: pd.DataFrame, strategy: str) -> dict:
    train = manifest[manifest["split"] == "train"]
    val = manifest[manifest["split"] == "validation"]
    test = manifest[manifest["split"] == "test"]

    if strategy == "compound-disjoint":
        train_items = set(train["smiles_a"]) | set(train["smiles_b"])
        val_items = set(val["smiles_a"]) | set(val["smiles_b"])
        test_items = set(test["smiles_a"]) | set(test["smiles_b"])
        return {
            "train_val_overlap": len(train_items & val_items),
            "train_test_overlap": len(train_items & test_items),
            "val_test_overlap": len(val_items & test_items),
        }

    chem_module, murcko_module = import_rdkit()

    def scaffold_set(frame: pd.DataFrame) -> Set[str]:
        items = set(frame["smiles_a"]) | set(frame["smiles_b"])
        return {murcko_key(smi, chem_module, murcko_module) for smi in items}

    train_items = scaffold_set(train)
    val_items = scaffold_set(val)
    test_items = scaffold_set(test)
    return {
        "train_val_overlap": len(train_items & val_items),
        "train_test_overlap": len(train_items & test_items),
        "val_test_overlap": len(val_items & test_items),
    }


def load_autogluon_settings(config_path: Path, *, presets_override: Optional[str], time_limit_override: Optional[int]) -> dict:
    config = load_json(config_path)
    return {
        "presets": presets_override or config.get("autogluon_presets", "best_quality"),
        "time_limit": time_limit_override if time_limit_override is not None else config.get("autogluon_time_limit"),
        "fit_kwargs": config.get("autogluon_fit_kwargs", {}),
        "retry_without_xgb": config.get("autogluon_retry_without_xgb", True),
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    feature_map = discover_feature_sets(args.data_root)
    if args.feature_set not in feature_map:
        raise KeyError(f"Feature set not found: {args.feature_set}")

    full_df = load_full_feature_matrix(feature_map[args.feature_set])
    X_full, y_full = select_feature_columns(full_df)
    labels = y_full.to_numpy(dtype=int)
    meta = load_pair_metadata(args.data_root)
    if len(meta) != len(X_full):
        raise RuntimeError(f"Metadata rows ({len(meta)}) do not match feature rows ({len(X_full)}).")

    if args.split_strategy == "scaffold":
        groups = build_scaffold_groups(meta)
    else:
        groups = build_component_groups(meta, ["smiles_a", "smiles_b"])

    split, split_summary = grouped_train_val_test_split(
        labels,
        groups,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        subset_trials=args.subset_trials,
    )

    manifest = build_split_manifest(meta, labels, groups, split)
    manifest.to_csv(output_root / "split_manifest.csv", index=False)

    overlap = overlap_summary(manifest, args.split_strategy)
    summary_payload = {
        "feature_set": args.feature_set,
        "split_strategy": args.split_strategy,
        "top_k": int(min(args.top_k, X_full.shape[1])),
        "n_features_total": int(X_full.shape[1]),
        "split_summary": split_summary,
        "overlap_summary": overlap,
    }
    dump_json(output_root / "split_summary.json", summary_payload)

    log(f"[split] strategy={args.split_strategy} train={split_summary['train_n']} val={split_summary['val_n']} test={split_summary['test_n']}")
    log(f"[split] overlap summary: {overlap}")

    if args.prepare_only:
        log(f"[done] prepared split only at: {output_root}")
        return 0

    selected_k = int(min(args.top_k, X_full.shape[1]))
    if selected_k < X_full.shape[1]:
        raise ValueError(
            "This script currently assumes the best E+F+M setting uses all features. "
            "If you want top-k feature selection under a new split, please add ranking on the training fold first."
        )
    feature_names = list(X_full.columns[:selected_k])
    pd.DataFrame({"feature": feature_names}).to_csv(output_root / "selected_features.csv", index=False)

    train_val_idx = np.concatenate([split.train_idx, split.val_idx])
    X_train = X_full.iloc[train_val_idx, :selected_k].reset_index(drop=True)
    y_train = y_full.iloc[train_val_idx].reset_index(drop=True)
    X_test = X_full.iloc[split.test_idx, :selected_k].reset_index(drop=True)
    y_test = y_full.iloc[split.test_idx].to_numpy(dtype=int)

    ag = load_autogluon_settings(
        args.autogluon_config,
        presets_override=args.autogluon_presets,
        time_limit_override=args.autogluon_time_limit,
    )
    predictor_path = output_root / "autogluon_model"
    predictor, y_pred, y_score = fit_predict_autogluon_with_retry(
        X_train,
        y_train,
        X_test,
        predictor_path=predictor_path,
        presets=ag["presets"],
        time_limit=ag["time_limit"],
        retry_without_xgb=ag["retry_without_xgb"],
        extra_fit_kwargs=ag["fit_kwargs"],
    )
    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics_df = pd.DataFrame(
        [
            {
                "FeatureSet": args.feature_set,
                "Model": "AutoGluon",
                "SplitStrategy": args.split_strategy,
                "SelectedFeatureCount": selected_k,
                **metrics,
            }
        ]
    )
    metrics_df.to_csv(output_root / "test_metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_df = pd.DataFrame(
        {
            "FeatureSet": args.feature_set,
            "Model": "AutoGluon",
            "SplitStrategy": args.split_strategy,
            "FPR": fpr,
            "TPR": tpr,
            "AUC": metrics["AUC"],
        }
    )
    roc_df.to_csv(output_root / "roc_curve_data.csv", index=False)
    plot_roc_curves(roc_df[["Model", "FPR", "TPR", "AUC"]], output_root / "roc_curves.png", title=f"ROC - {args.feature_set} ({args.split_strategy})")

    internal_df = evaluate_autogluon_internal_models(predictor_path, X_test, y_test, output_root)
    plot_autogluon_internal_heatmap(
        internal_df,
        output_root / "autogluon_internal_heatmap.png",
        title=f"AutoGluon Internal Metrics - {args.split_strategy}",
    )
    del predictor

    log(f"[done] outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
