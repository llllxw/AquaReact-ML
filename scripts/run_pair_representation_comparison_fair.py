#!/usr/bin/env python3
"""Fair pair-representation comparison built around the best E+F+M + AutoGluon setup.

Goals:
1. Reproduce the original E+F+M feature matrix as the authoritative `Concat` baseline.
2. Rebuild single-drug E+F+M fingerprints with the same per-drug logic.
3. Compare Concat / AbsDiff / Hadamard / AllCombined under the same split and
   the same feature-selection pipeline (mRMR + IFS with AutoGluon).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def _sanitize_runtime_sys_path() -> None:
    if ".venv" not in sys.executable:
        return
    blocked_markers = [
        "/home/xwl/miniconda3/lib/python3.12/site-packages",
        "/home/xwl/miniconda3/lib/python3.11/site-packages",
    ]
    sys.path[:] = [p for p in sys.path if not any(marker in p for marker in blocked_markers)]


_sanitize_runtime_sys_path()
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from train_fingerprint_models import (
    adaptive_ifs_counts,
    choose_best_feature_count,
    compute_metrics,
    discover_feature_sets,
    evaluate_autogluon_internal_models,
    fit_predict_autogluon_with_retry,
    load_full_feature_matrix,
    plot_autogluon_internal_heatmap,
    plot_ifs_curve,
    plot_model_bar,
    plot_roc_curves,
    rank_features,
    score_models_for_ifs,
    select_feature_columns,
)


MACCS_LEN = 167
ECFP_LEN = 1024
FCFP_LEN = 2048
SINGLE_DRUG_DIM = MACCS_LEN + ECFP_LEN + FCFP_LEN


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair pair representation comparison for E+F+M + AutoGluon.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/xwl/药物禁忌/元数据"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/run_20260403_215511_full_final/split_manifest.csv"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--representations",
        nargs="*",
        default=["Concat", "AbsDiff", "Hadamard", "AllCombined"],
        help="Subset to run. Concat uses the original E+F+M feature files as the authoritative baseline.",
    )
    parser.add_argument("--selector", default="mrmr")
    parser.add_argument("--selector-fallback", default="mutual_info")
    parser.add_argument("--ifs-step", type=int, default=400)
    parser.add_argument("--ifs-max-points", type=int, default=40)
    parser.add_argument("--autogluon-config", type=Path, default=Path("/home/xwl/药物禁忌/configs/autogluon_boosted.json"))
    parser.add_argument("--autogluon-presets", default=None)
    parser.add_argument("--autogluon-time-limit", type=int, default=None)
    parser.add_argument("--autogluon-ifs-presets", default=None)
    parser.add_argument("--autogluon-ifs-time-limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_split_manifest(path: Path, n_rows: int) -> pd.DataFrame:
    manifest = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
    required = {"row_id", "label", "split"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"Split manifest missing required columns: {required}")
    if len(manifest) != n_rows:
        raise RuntimeError(f"Split manifest rows ({len(manifest)}) do not match data rows ({n_rows})")
    return manifest


def import_rdkit():
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem, MACCSkeys
        RDLogger.DisableLog("rdApp.warning")
    except Exception:
        candidate_paths = [
            Path("/home/xwl/miniconda3/lib/python3.12/site-packages"),
            Path("/home/xwl/miniconda3/lib/python3.11/site-packages"),
        ]
        for candidate in candidate_paths:
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.append(candidate_str)
                try:
                    from rdkit import Chem, DataStructs, RDLogger
                    from rdkit.Chem import AllChem, MACCSkeys
                    RDLogger.DisableLog("rdApp.warning")
                    return Chem, DataStructs, AllChem, MACCSkeys
                except Exception:
                    continue
        raise ModuleNotFoundError(
            "RDKit is required. A system copy exists under /home/xwl/miniconda3, but it could not be loaded."
        )
    return Chem, DataStructs, AllChem, MACCSkeys


def bitvect_to_array(fp, n_bits: int, data_structs) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    data_structs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_mol(smiles: str, chem_module):
    smi = str(smiles).strip()
    if not smi:
        return None
    return chem_module.MolFromSmiles(smi)


def load_pair_metadata(data_root: Path) -> pd.DataFrame:
    path = data_root / "combined-data.csv"
    meta = pd.read_csv(path)
    if meta.shape[1] < 2:
        raise RuntimeError(f"Expected at least two columns in {path}")
    meta = meta.iloc[:, :2].copy()
    meta.columns = ["smiles_a", "smiles_b"]
    meta["smiles_a"] = meta["smiles_a"].astype(str).str.strip()
    meta["smiles_b"] = meta["smiles_b"].astype(str).str.strip()
    return meta


def build_single_drug_cache(meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    chem_module, data_structs, all_chem, maccskeys = import_rdkit()
    cache: Dict[str, np.ndarray] = {}
    unique_smiles = pd.Index(pd.unique(pd.concat([meta["smiles_a"], meta["smiles_b"]], ignore_index=True)))
    for idx, smiles in enumerate(unique_smiles, start=1):
        mol = smiles_to_mol(smiles, chem_module)
        if mol is None:
            vec = np.zeros((SINGLE_DRUG_DIM,), dtype=np.int8)
        else:
            fp_maccs = maccskeys.GenMACCSKeys(mol)
            arr_maccs = bitvect_to_array(fp_maccs, MACCS_LEN, data_structs)

            fp_ecfp = all_chem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=ECFP_LEN)
            arr_ecfp = bitvect_to_array(fp_ecfp, ECFP_LEN, data_structs)

            fp_fcfp = all_chem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FCFP_LEN, useFeatures=True)
            arr_fcfp = bitvect_to_array(fp_fcfp, FCFP_LEN, data_structs)

            vec = np.concatenate([arr_maccs, arr_ecfp, arr_fcfp]).astype(np.int8)
        cache[smiles] = vec
        if idx % 2000 == 0 or idx == len(unique_smiles):
            log(f"[prep] cached {idx}/{len(unique_smiles)} unique SMILES")
    return cache


def build_representation_features(meta: pd.DataFrame, representation: str, cache: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[np.ndarray] = []
    total = len(meta)
    for idx, (smiles_a, smiles_b) in enumerate(zip(meta["smiles_a"], meta["smiles_b"]), start=1):
        vec_a = cache[smiles_a]
        vec_b = cache[smiles_b]
        if representation == "Concat":
            row = np.concatenate([vec_a, vec_b]).astype(np.int8)
        elif representation == "AbsDiff":
            row = np.abs(vec_a.astype(np.int16) - vec_b.astype(np.int16)).astype(np.int8)
        elif representation == "Hadamard":
            row = (vec_a.astype(np.int8) * vec_b.astype(np.int8)).astype(np.int8)
        elif representation == "AllCombined":
            concat = np.concatenate([vec_a, vec_b]).astype(np.int8)
            absdiff = np.abs(vec_a.astype(np.int16) - vec_b.astype(np.int16)).astype(np.int8)
            hadamard = (vec_a.astype(np.int8) * vec_b.astype(np.int8)).astype(np.int8)
            row = np.concatenate([concat, absdiff, hadamard]).astype(np.int8)
        else:
            raise ValueError(f"Unsupported built representation: {representation}")
        rows.append(row)
        if idx % 4000 == 0 or idx == total:
            log(f"[build] {representation}: {idx}/{total} pairs")
    matrix = np.vstack(rows)
    return pd.DataFrame(matrix, columns=[str(i) for i in range(matrix.shape[1])])


def load_original_concat_full_df(data_root: Path) -> pd.DataFrame:
    feature_map = discover_feature_sets(data_root)
    return load_full_feature_matrix(feature_map["E+F+M"])


def validate_rebuilt_concat(original_X: pd.DataFrame, rebuilt_X: pd.DataFrame, output_dir: Path) -> None:
    same_shape = tuple(original_X.shape) == tuple(rebuilt_X.shape)
    validation = {"same_shape": same_shape}
    if same_shape:
        matches = (original_X.to_numpy(dtype=np.int8) == rebuilt_X.to_numpy(dtype=np.int8))
        validation["elementwise_match_ratio"] = float(matches.mean())
        validation["all_equal"] = bool(matches.all())
    dump_json(output_dir / "concat_validation.json", validation)


def load_ag_config(args: argparse.Namespace) -> dict:
    config = load_json(args.autogluon_config)
    return {
        "selector_method": args.selector,
        "selector_fallback": args.selector_fallback,
        "ifs_metric": "AUC",
        "ifs_step": args.ifs_step,
        "ifs_max_points": args.ifs_max_points,
        "autogluon_presets": args.autogluon_presets or config.get("autogluon_presets", "best_quality"),
        "autogluon_time_limit": args.autogluon_time_limit if args.autogluon_time_limit is not None else config.get("autogluon_time_limit"),
        "autogluon_fit_kwargs": config.get("autogluon_fit_kwargs", {}),
        "autogluon_ifs_presets": args.autogluon_ifs_presets or config.get("autogluon_ifs_presets") or config.get("autogluon_presets", "high_quality"),
        "autogluon_ifs_time_limit": args.autogluon_ifs_time_limit if args.autogluon_ifs_time_limit is not None else config.get("autogluon_ifs_time_limit"),
        "autogluon_ifs_fit_kwargs": config.get("autogluon_ifs_fit_kwargs", {}),
        "autogluon_retry_without_xgb": config.get("autogluon_retry_without_xgb", True),
        "random_seed": 42,
    }


def run_one_representation(
    name: str,
    full_df: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> dict:
    X_full, y_full = select_feature_columns(full_df)
    train_idx = manifest.index[manifest["split"] == "train"].to_numpy()
    val_idx = manifest.index[manifest["split"] == "validation"].to_numpy()
    test_idx = manifest.index[manifest["split"] == "test"].to_numpy()

    X_train = X_full.iloc[train_idx].reset_index(drop=True)
    y_train = y_full.iloc[train_idx].reset_index(drop=True)
    X_val = X_full.iloc[val_idx].reset_index(drop=True)
    y_val = y_full.iloc[val_idx].reset_index(drop=True)
    X_test = X_full.iloc[test_idx].reset_index(drop=True)
    y_test = y_full.iloc[test_idx].to_numpy(dtype=int)

    ranked_features = rank_features(
        X_train,
        y_train,
        method=config["selector_method"],
        fallback=config["selector_fallback"],
    )
    pd.DataFrame({"rank": np.arange(1, len(ranked_features) + 1), "feature": ranked_features}).to_csv(
        output_dir / "ranked_features.csv", index=False
    )

    counts = adaptive_ifs_counts(X_train.shape[1], fixed_step=config["ifs_step"], max_points=int(config["ifs_max_points"]))
    ifs_df = score_models_for_ifs(
        ["AutoGluon"],
        X_train,
        y_train,
        X_val,
        y_val,
        counts,
        ranked_features,
        config,
        output_dir,
    )
    ifs_df.to_csv(output_dir / "ifs_results.csv", index=False)
    plot_ifs_curve(ifs_df, output_dir / "ifs_curve.png", reference_model="AutoGluon", feature_set=name)

    best_k = choose_best_feature_count(ifs_df, reference_model="AutoGluon")
    pd.DataFrame([{"Model": "AutoGluon", "SelectedFeatureCount": best_k}]).to_csv(
        output_dir / "selected_feature_counts_by_model.csv", index=False
    )
    pd.DataFrame({"Model": "AutoGluon", "Feature": ranked_features[:best_k]}).to_csv(
        output_dir / "selected_features_by_model.csv", index=False
    )
    log(f"[run] {name} selected feature count = {best_k}")

    train_val_mask = manifest["split"].isin(["train", "validation"]).to_numpy()
    X_train_pool = X_full.loc[train_val_mask, ranked_features[:best_k]].reset_index(drop=True)
    y_train_pool = y_full.loc[train_val_mask].reset_index(drop=True)
    X_test_final = X_test[ranked_features[:best_k]]
    predictor_path = output_dir / "autogluon_model"

    predictor, y_pred, y_score = fit_predict_autogluon_with_retry(
        X_train_pool,
        y_train_pool,
        X_test_final,
        predictor_path=predictor_path,
        presets=config["autogluon_presets"],
        time_limit=config["autogluon_time_limit"],
        retry_without_xgb=config["autogluon_retry_without_xgb"],
        extra_fit_kwargs=config["autogluon_fit_kwargs"],
    )

    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics_row = {
        "Representation": name,
        "SelectedFeatureCount": best_k,
        "FeatureDim": int(X_full.shape[1]),
        **metrics,
    }
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_df = pd.DataFrame({"Model": [name] * len(fpr), "FPR": fpr, "TPR": tpr, "AUC": metrics["AUC"]})
    roc_df.to_csv(output_dir / "roc_curve_data.csv", index=False)

    internal_df = evaluate_autogluon_internal_models(predictor_path, X_test_final, y_test, output_dir)
    plot_autogluon_internal_heatmap(
        internal_df,
        output_dir / "autogluon_internal_heatmap.png",
        title=f"AutoGluon Internal Metrics - {name}",
    )
    del predictor

    return {"metrics": metrics_row, "roc": roc_df}


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_ag_config(args)
    meta = load_pair_metadata(args.data_root)
    original_concat_df = load_original_concat_full_df(args.data_root)
    original_X, labels = select_feature_columns(original_concat_df)
    manifest = load_split_manifest(args.split_manifest, n_rows=len(original_concat_df))

    if not np.array_equal(labels.to_numpy(dtype=int), manifest["label"].to_numpy(dtype=int)):
        raise RuntimeError("Labels in split_manifest do not match the original E+F+M feature matrix.")

    dump_json(
        output_root / "run_config.json",
        {
            "representations": args.representations,
            "split_manifest": str(args.split_manifest),
            "selector": config["selector_method"],
            "selector_fallback": config["selector_fallback"],
            "ifs_step": config["ifs_step"],
            "ifs_max_points": config["ifs_max_points"],
            "autogluon_presets": config["autogluon_presets"],
            "autogluon_time_limit": config["autogluon_time_limit"],
            "autogluon_ifs_presets": config["autogluon_ifs_presets"],
            "autogluon_ifs_time_limit": config["autogluon_ifs_time_limit"],
        },
    )

    cache = build_single_drug_cache(meta)
    rebuilt_concat_X = build_representation_features(meta, "Concat", cache)
    validate_rebuilt_concat(original_X, rebuilt_concat_X, output_root)
    del rebuilt_concat_X

    results = []
    roc_frames = []

    for representation in args.representations:
        rep_dir = output_root / representation
        rep_dir.mkdir(parents=True, exist_ok=True)

        if representation == "Concat":
            full_df = original_concat_df.copy()
        else:
            if representation not in {"AbsDiff", "Hadamard", "AllCombined"}:
                raise ValueError(f"Unsupported representation: {representation}")
            X_built = build_representation_features(meta, representation, cache)
            full_df = pd.concat([labels.reset_index(drop=True), X_built.reset_index(drop=True)], axis=1)
            full_df.columns = ["type"] + [str(i) for i in range(X_built.shape[1])]

        artifact = run_one_representation(representation, full_df, manifest, config, rep_dir)
        results.append(artifact["metrics"])
        roc_frames.append(artifact["roc"])

    metrics_df = pd.DataFrame(results).sort_values("AUC", ascending=False).reset_index(drop=True)
    roc_df = pd.concat(roc_frames, ignore_index=True)
    metrics_df.to_csv(output_root / "pair_representation_metrics.csv", index=False)
    roc_df.to_csv(output_root / "pair_representation_roc_curve_data.csv", index=False)

    plot_bar_df = metrics_df.rename(columns={"Representation": "Model"})
    plot_model_bar(plot_bar_df, output_root / "pair_representation_metrics_bar.png", title="Fair Pair Representation Comparison")
    plot_roc_curves(roc_df, output_root / "pair_representation_roc_curves.png", title="Fair Pair Representation ROC Comparison")

    log(f"[done] outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
