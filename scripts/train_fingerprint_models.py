#!/usr/bin/env python3
"""Command-line pipeline for fingerprint modeling experiments.

This script reorganizes the notebook workflow into a reproducible pipeline:
1. Rebuild a full dataset for each fingerprint set from the existing train/test CSV files.
2. Create a new internal train / validation / test split.
3. Run feature ranking (mRMR by default) and IFS on the training portion only.
4. Train classical ML models and optional AutoGluon on the selected features.
5. Save summary tables and publication-ready plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


CLASSICAL_MODELS = ["DT", "RF", "ExtraTrees", "KNN", "GB", "XGBoost", "CatBoost"]
OPTIONAL_MODELS = ["XGBoost", "CatBoost", "AutoGluon"]
DEFAULT_MODELS = ["DT", "RF", "ExtraTrees", "KNN", "GB", "XGBoost", "CatBoost", "AutoGluon"]


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def import_optional(module_name: str):
    try:
        return __import__(module_name, fromlist=["dummy"])
    except Exception:
        return None


def check_runtime_dependencies() -> Dict[str, bool]:
    checks = {
        "matplotlib": import_optional("matplotlib") is not None,
        "mrmr": import_optional("mrmr") is not None,
        "xgboost": import_optional("xgboost") is not None,
        "catboost": import_optional("catboost") is not None,
        "autogluon.tabular": import_optional("autogluon.tabular") is not None,
    }
    for name, ok in checks.items():
        log(f"[env] {name:18s} {'OK' if ok else 'MISSING'}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fingerprint reaction models.")
    parser.add_argument("--config", type=Path, default=None, help="Path to experiment json config.")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root containing 元数据/训练集 and 元数据/测试集.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory for experiment outputs.")
    parser.add_argument("--feature-sets", nargs="*", default=None, help="Override feature sets from config.")
    parser.add_argument("--models", nargs="*", default=None, help="Override models from config.")
    parser.add_argument("--ifs-models", nargs="*", default=None, help="Models used for IFS scoring.")
    parser.add_argument("--selector", default=None, help="Feature ranking method: mrmr or mutual_info.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--test-size", type=float, default=None, help="Hold-out test size.")
    parser.add_argument("--validation-size", type=float, default=None, help="Validation size taken from train split.")
    parser.add_argument("--ifs-step", type=int, default=None, help="Fixed IFS step.")
    parser.add_argument("--ifs-max-points", type=int, default=None, help="Maximum points in IFS curve.")
    parser.add_argument("--skip-missing-models", action="store_true", help="Skip optional models that are not installed.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip all plot generation.")
    parser.add_argument("--check-env", action="store_true", help="Only print dependency availability and exit.")
    return parser.parse_args()


def merge_cli_into_config(args: argparse.Namespace, config: dict) -> dict:
    merged = dict(config)
    if args.data_root is not None:
        merged["data_root"] = str(args.data_root)
    if args.output_root is not None:
        merged["output_root"] = str(args.output_root)
    if args.feature_sets:
        merged["feature_sets"] = args.feature_sets
    if args.models:
        merged["models"] = args.models
    if args.ifs_models:
        merged["ifs_models"] = args.ifs_models
    if args.selector is not None:
        merged["selector_method"] = args.selector
    if args.seed is not None:
        merged["random_seed"] = args.seed
    if args.test_size is not None:
        merged["test_size"] = args.test_size
    if args.validation_size is not None:
        merged["validation_size"] = args.validation_size
    if args.ifs_step is not None:
        merged["ifs_step"] = args.ifs_step
    if args.ifs_max_points is not None:
        merged["ifs_max_points"] = args.ifs_max_points
    if args.skip_missing_models:
        merged["skip_missing_models"] = True
    if args.skip_plots:
        merged["skip_plots"] = True
    return merged


def default_config() -> dict:
    return {
        "data_root": "/home/xwl/药物禁忌/元数据",
        "output_root": "/home/xwl/药物禁忌/outputs/default_run",
        "feature_sets": "all",
        "models": DEFAULT_MODELS,
        "ifs_models": ["AutoGluon"],
        "per_model_feature_selection": True,
        "selector_method": "mrmr",
        "selector_fallback": "mutual_info",
        "ifs_metric": "AUC",
        "ifs_reference_model": "AutoGluon",
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_seed": 42,
        "ifs_step": None,
        "ifs_max_points": 60,
        "skip_missing_models": True,
        "skip_plots": False,
        "autogluon_presets": "medium_quality",
        "autogluon_time_limit": None,
        "autogluon_fit_kwargs": {},
        "autogluon_ifs_presets": None,
        "autogluon_ifs_time_limit": None,
        "autogluon_ifs_fit_kwargs": {},
        "autogluon_retry_without_xgb": True,
    }


def normalize_model_name(name: str) -> str:
    aliases = {
        "dt": "DT",
        "rf": "RF",
        "extratrees": "ExtraTrees",
        "extra_trees": "ExtraTrees",
        "knn": "KNN",
        "gb": "GB",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "autogluon": "AutoGluon",
    }
    key = name.strip()
    return aliases.get(key.lower(), key)


def discover_feature_sets(data_root: Path) -> Dict[str, Dict[str, Path]]:
    train_dir = data_root / "训练集"
    test_dir = data_root / "测试集"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Cannot find train/test directories under: {data_root}")

    train_map = {
        p.stem.replace("train_", ""): p
        for p in sorted(train_dir.glob("train_*.csv"))
    }
    test_map = {
        p.stem.replace("test_", ""): p
        for p in sorted(test_dir.glob("test_*.csv"))
    }
    shared = sorted(set(train_map) & set(test_map))
    if not shared:
        raise RuntimeError("No shared feature sets were found.")
    return {name: {"train": train_map[name], "test": test_map[name]} for name in shared}


def resolve_feature_sets(config: dict, discovered: Sequence[str]) -> List[str]:
    configured = config.get("feature_sets", "all")
    if configured == "all":
        return list(discovered)
    selected = [name for name in configured if name in discovered]
    missing = sorted(set(configured) - set(selected))
    if missing:
        raise ValueError(f"Feature sets not found: {missing}")
    return selected


def load_full_feature_matrix(feature_paths: Dict[str, Path]) -> pd.DataFrame:
    train_df = pd.read_csv(feature_paths["train"])
    test_df = pd.read_csv(feature_paths["test"])
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df.columns = [str(c) for c in full_df.columns]
    full_df.iloc[:, 0] = full_df.iloc[:, 0].astype(int)
    return full_df


def create_split(labels: np.ndarray, test_size: float, validation_size: float, seed: int) -> SplitData:
    all_idx = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=validation_size,
        stratify=labels[train_idx],
        random_state=seed,
    )
    return SplitData(
        train_idx=np.sort(train_idx),
        val_idx=np.sort(val_idx),
        test_idx=np.sort(test_idx),
    )


def build_split_manifest(labels: np.ndarray, split_data: SplitData) -> pd.DataFrame:
    split_col = np.full(labels.shape[0], "unused", dtype=object)
    split_col[split_data.train_idx] = "train"
    split_col[split_data.val_idx] = "validation"
    split_col[split_data.test_idx] = "test"
    return pd.DataFrame(
        {
            "row_id": np.arange(labels.shape[0]),
            "label": labels.astype(int),
            "split": split_col,
        }
    )


def select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df.iloc[:, 0].astype(int)
    X = df.iloc[:, 1:].copy()
    X.columns = [str(c) for c in X.columns]
    return X, y


def rank_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    fallback: str = "mutual_info",
) -> List[str]:
    method = method.lower()
    if method == "mrmr":
        module = import_optional("mrmr")
        if module is not None:
            ranked = module.mrmr_classif(X=X, y=y, K=min(X.shape[1], X.shape[1]))
            return [str(col) for col in ranked]
        if fallback:
            log("[warn] mrmr is not installed, fallback to mutual_info ranking.")
            return rank_features(X, y, fallback, fallback="")
        raise ModuleNotFoundError("mrmr is required but not installed.")

    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif

        scores = mutual_info_classif(X, y, random_state=42, discrete_features=True)
        order = np.argsort(scores)[::-1]
        return [str(X.columns[i]) for i in order]

    raise ValueError(f"Unsupported selector method: {method}")


def adaptive_ifs_counts(n_features: int, fixed_step: Optional[int], max_points: int) -> List[int]:
    if n_features <= 0:
        return []
    if fixed_step and fixed_step > 0:
        if fixed_step >= n_features:
            return [n_features]
        counts = list(range(fixed_step, n_features + 1, fixed_step))
        if not counts or counts[-1] != n_features:
            counts.append(n_features)
        return counts

    if n_features <= 334:
        step = 10
    elif n_features <= 1024:
        step = 25
    elif n_features <= 4096:
        step = 100
    elif n_features <= 8192:
        step = 200
    else:
        step = 250

    counts = list(range(step, n_features + 1, step))
    if not counts or counts[-1] != n_features:
        counts.append(n_features)

    if len(counts) > max_points:
        sampled = np.linspace(1, len(counts) - 1, num=max_points - 1, dtype=int)
        counts = [counts[0]] + [counts[i] for i in sampled]
        counts = sorted(set(counts + [n_features]))
    return counts


def load_existing_feature_artifacts(
    feature_dir: Path,
    full_df: pd.DataFrame,
    split_data: SplitData,
) -> Optional[dict]:
    metrics_path = feature_dir / "test_metrics.csv"
    if not metrics_path.exists():
        return None

    try:
        metrics_df = pd.read_csv(metrics_path)
    except Exception as exc:
        log(f"[warn] unable to load existing metrics from {metrics_path}: {exc}")
        return None

    if metrics_df.empty:
        return None

    roc_path = feature_dir / "roc_curve_data.csv"
    roc_df = pd.read_csv(roc_path) if roc_path.exists() else pd.DataFrame()

    selected_counts_path = feature_dir / "selected_feature_counts_by_model.csv"
    if selected_counts_path.exists():
        counts_df = pd.read_csv(selected_counts_path)
        best_k_by_model = {
            str(row["Model"]): int(row["SelectedFeatureCount"])
            for _, row in counts_df.iterrows()
        }
    else:
        best_k_by_model = {
            str(row["Model"]): int(row["SelectedFeatureCount"])
            for _, row in metrics_df.iterrows()
        }

    X_full, y_full = select_feature_columns(full_df)
    X_test = X_full.iloc[split_data.test_idx].reset_index(drop=True)
    y_test = y_full.iloc[split_data.test_idx].to_numpy(dtype=int)

    X_test_by_model: Dict[str, pd.DataFrame] = {}
    selected_features_path = feature_dir / "selected_features_by_model.csv"
    if selected_features_path.exists():
        selected_df = pd.read_csv(selected_features_path)
        for model_name, group in selected_df.groupby("Model"):
            selected_features = [str(x) for x in group["Feature"].tolist()]
            if selected_features:
                X_test_by_model[str(model_name)] = X_test[selected_features]

    autogluon_path = feature_dir / "autogluon_model"
    return {
        "feature_dir": feature_dir,
        "metrics": metrics_df,
        "roc": roc_df,
        "selected_k_by_model": best_k_by_model,
        "X_test_by_model": X_test_by_model,
        "y_test": y_test,
        "autogluon_path": autogluon_path if autogluon_path.exists() else None,
    }


def instantiate_model(model_name: str, seed: int):
    model_name = normalize_model_name(model_name)
    if model_name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if model_name == "GB":
        return GradientBoostingClassifier(random_state=seed)
    if model_name == "XGBoost":
        module = import_optional("xgboost")
        if module is None:
            raise ModuleNotFoundError("xgboost is not installed.")
        return module.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "CatBoost":
        module = import_optional("catboost")
        if module is None:
            raise ModuleNotFoundError("catboost is not installed.")
        return module.CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=seed,
            verbose=False,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def positive_scores(pred_proba) -> np.ndarray:
    if hasattr(pred_proba, "columns"):
        cols = list(pred_proba.columns)
        if 1 in cols:
            return pred_proba[1].to_numpy(dtype=float)
        if "1" in cols:
            return pred_proba["1"].to_numpy(dtype=float)
        return pred_proba[cols[-1]].to_numpy(dtype=float)

    arr = np.asarray(pred_proba, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.ravel()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1Score": f1,
        "AUC": auc,
        "Sensitivity": sens,
        "Specificity": spec,
        "MCC": mcc,
    }


def fit_predict_classical(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model = instantiate_model(model_name, seed)
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_eval)).astype(int)
    if hasattr(model, "predict_proba"):
        y_score = positive_scores(model.predict_proba(X_eval))
    elif hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X_eval), dtype=float).ravel()
    else:
        y_score = y_pred.astype(float)
    return y_pred, y_score


def fit_predict_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    *,
    predictor_path: Path,
    presets: str,
    time_limit: Optional[int],
    hyperparameters: Optional[dict] = None,
    extra_fit_kwargs: Optional[dict] = None,
) -> Tuple[object, np.ndarray, np.ndarray]:
    module = import_optional("autogluon.tabular")
    if module is None:
        raise ModuleNotFoundError("autogluon.tabular is not installed.")

    TabularPredictor = module.TabularPredictor
    train_data = X_train.copy()
    train_data["target"] = y_train.to_numpy(dtype=int)
    shutil.rmtree(predictor_path, ignore_errors=True)

    fit_kwargs = {
        "presets": presets,
        "time_limit": time_limit,
        "verbosity": 0,
    }
    if extra_fit_kwargs:
        fit_kwargs.update(extra_fit_kwargs)
    if hyperparameters is not None:
        fit_kwargs["hyperparameters"] = hyperparameters

    predictor = TabularPredictor(
        label="target",
        eval_metric="roc_auc",
        path=str(predictor_path),
    ).fit(train_data, **fit_kwargs)
    y_pred = predictor.predict(X_eval).to_numpy(dtype=int)
    y_score = positive_scores(predictor.predict_proba(X_eval))
    return predictor, y_pred, y_score


def fit_predict_autogluon_with_retry(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    *,
    predictor_path: Path,
    presets: str,
    time_limit: Optional[int],
    retry_without_xgb: bool,
    extra_fit_kwargs: Optional[dict] = None,
) -> Tuple[object, np.ndarray, np.ndarray]:
    try:
        return fit_predict_autogluon(
            X_train,
            y_train,
            X_eval,
            predictor_path=predictor_path,
            presets=presets,
            time_limit=time_limit,
            extra_fit_kwargs=extra_fit_kwargs,
        )
    except Exception as exc:
        if not retry_without_xgb or "n_classes_" not in str(exc):
            raise
        log("[warn] AutoGluon hit XGBoost compatibility issue; retrying without internal XGBoost models.")
        stable_hyperparameters = {
            "GBM": {},
            "CAT": {},
            "RF": {},
            "XT": {},
            "KNN": {},
            "NN_TORCH": {},
        }
        return fit_predict_autogluon(
            X_train,
            y_train,
            X_eval,
            predictor_path=predictor_path,
            presets=presets,
            time_limit=time_limit,
            hyperparameters=stable_hyperparameters,
            extra_fit_kwargs=extra_fit_kwargs,
        )


def score_models_for_ifs(
    model_names: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    counts: Sequence[int],
    ranked_features: Sequence[str],
    config: dict,
    feature_dir: Path,
) -> pd.DataFrame:
    rows = []
    for k in counts:
        selected = list(ranked_features[:k])
        row = {"FeatureCount": k}
        X_train_sel = X_train[selected]
        X_val_sel = X_val[selected]
        for model_name in model_names:
            try:
                if normalize_model_name(model_name) == "AutoGluon":
                    predictor_path = feature_dir / "ifs_models" / f"ag_k{k}"
                    ifs_presets = config.get("autogluon_ifs_presets") or config["autogluon_presets"]
                    ifs_time_limit = config.get("autogluon_ifs_time_limit")
                    if ifs_time_limit is None:
                        ifs_time_limit = config.get("autogluon_time_limit")
                    predictor, y_pred, y_score = fit_predict_autogluon_with_retry(
                        X_train_sel,
                        y_train,
                        X_val_sel,
                        predictor_path=predictor_path,
                        presets=ifs_presets,
                        time_limit=ifs_time_limit,
                        retry_without_xgb=config.get("autogluon_retry_without_xgb", True),
                        extra_fit_kwargs=config.get("autogluon_ifs_fit_kwargs", {}),
                    )
                    del predictor
                else:
                    y_pred, y_score = fit_predict_classical(
                        model_name,
                        X_train_sel,
                        y_train,
                        X_val_sel,
                        config["random_seed"],
                    )
                metrics = compute_metrics(y_val.to_numpy(dtype=int), y_pred, y_score)
                row[normalize_model_name(model_name)] = metrics[config["ifs_metric"]]
            except Exception as exc:
                row[normalize_model_name(model_name)] = np.nan
                log(f"[warn] IFS scoring failed for {model_name} @ k={k}: {exc}")
        rows.append(row)
        log(f"[IFS] finished k={k} for {feature_dir.name}")
    return pd.DataFrame(rows)


def choose_best_feature_count(
    ifs_df: pd.DataFrame,
    reference_model: str,
) -> int:
    reference_model = normalize_model_name(reference_model)
    if reference_model in ifs_df.columns and ifs_df[reference_model].notna().any():
        best_idx = ifs_df[reference_model].astype(float).idxmax()
        return int(ifs_df.loc[best_idx, "FeatureCount"])

    metric_cols = [c for c in ifs_df.columns if c != "FeatureCount"]
    if not metric_cols:
        raise RuntimeError("IFS result has no metric columns.")
    temp = ifs_df[metric_cols].mean(axis=1, skipna=True)
    best_idx = temp.idxmax()
    return int(ifs_df.loc[best_idx, "FeatureCount"])


def choose_best_feature_counts_by_model(
    ifs_df: pd.DataFrame,
    model_names: Sequence[str],
    reference_model: str,
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    fallback_k = choose_best_feature_count(ifs_df, reference_model=reference_model)
    for model_name in model_names:
        norm = normalize_model_name(model_name)
        if norm in ifs_df.columns and ifs_df[norm].notna().any():
            best_idx = ifs_df[norm].astype(float).idxmax()
            result[norm] = int(ifs_df.loc[best_idx, "FeatureCount"])
        else:
            result[norm] = fallback_k
    return result


def ensure_matplotlib():
    module = import_optional("matplotlib.pyplot")
    if module is None:
        raise ModuleNotFoundError("matplotlib is not installed.")
    module.rcParams["font.family"] = "sans-serif"
    module.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    module.rcParams["pdf.fonttype"] = 42
    module.rcParams["ps.fonttype"] = 42
    module.rcParams["axes.unicode_minus"] = False
    return module


def save_figure_multi(fig, output_path: Path, dpi: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path)


def plot_ifs_curve(ifs_df: pd.DataFrame, output_path: Path, reference_model: str, feature_set: str) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_cols = [c for c in ifs_df.columns if c != "FeatureCount"]
    for col in metric_cols:
        valid = ifs_df[["FeatureCount", col]].dropna()
        if valid.empty:
            continue
        ax.plot(valid["FeatureCount"], valid[col], marker="o", linewidth=1.3, alpha=0.9, label=col)
        best_idx = valid[col].astype(float).idxmax()
        best_x = valid.loc[best_idx, "FeatureCount"]
        best_y = valid.loc[best_idx, col]
        ax.scatter([best_x], [best_y], zorder=5)
        ax.annotate(f"{col}:{int(best_x)}", (best_x, best_y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_title(f"IFS Curve - {feature_set}")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Validation AUC")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def plot_model_bar(metrics_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt = ensure_matplotlib()
    metrics = ["Accuracy", "Precision", "Recall", "F1Score", "AUC", "Sensitivity", "Specificity", "MCC"]
    plot_df = metrics_df.set_index("Model")[metrics]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(plot_df), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (model_name, row) in enumerate(plot_df.iterrows()):
        ax.bar(x + idx * width, row.values, width=width, label=model_name)
    ax.set_xticks(x + width * (len(plot_df) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(roc_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, group in roc_df.groupby("Model"):
        ax.plot(group["FPR"], group["TPR"], linewidth=1.8, label=f"{model_name} (AUC={group['AUC'].iloc[0]:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def plot_autogluon_internal_heatmap(
    metrics_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: Optional[int] = None,
    prioritize_models: Optional[Sequence[str]] = None,
    annotation_fontsize: int = 8,
    cmap: str = "YlGnBu",
    annotation_fontweight: str = "normal",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt = ensure_matplotlib()
    value_cols = ["Accuracy", "Precision", "Recall", "F1Score", "AUC", "Sensitivity", "Specificity", "MCC"]
    plot_source = metrics_df.copy()
    if top_n is not None and len(plot_source) > top_n:
        plot_source = plot_source.sort_values("AUC", ascending=False).head(top_n).copy()
    if prioritize_models:
        priority_order = {name: idx for idx, name in enumerate(prioritize_models)}
        plot_source["_priority"] = plot_source["Model"].map(priority_order).fillna(len(priority_order))
        plot_source = plot_source.sort_values(["_priority", "AUC"], ascending=[True, False]).drop(columns=["_priority"])
    plot_df = plot_source.set_index("Model")[value_cols]
    fig_h = max(4.0, 0.55 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    resolved_vmin = max(0.0, float(plot_df.min().min()) - 0.03) if vmin is None else vmin
    resolved_vmax = min(1.0, float(plot_df.max().max()) + 0.03) if vmax is None else vmax
    im = ax.imshow(plot_df.values, aspect="auto", cmap=cmap, vmin=resolved_vmin, vmax=resolved_vmax)
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            cell_value = float(plot_df.iloc[i, j])
            text_color = "white" if cell_value >= (resolved_vmin + resolved_vmax) / 2 else "#1f2937"
            ax.text(
                j,
                i,
                f"{cell_value:.3f}",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color=text_color,
                fontweight=annotation_fontweight,
            )
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=270, labelpad=15)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def save_summary_tables(
    all_results: pd.DataFrame,
    output_root: Path,
) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(tables_dir / "all_metrics_long.csv", index=False)

    accuracy_wide = all_results.pivot_table(index="FeatureSet", columns="Model", values="Accuracy")
    auc_wide = all_results.pivot_table(index="FeatureSet", columns="Model", values="AUC")
    selected_k = all_results.pivot_table(index="FeatureSet", columns="Model", values="SelectedFeatureCount")

    accuracy_wide.to_csv(tables_dir / "summary_accuracy_wide.csv")
    auc_wide.to_csv(tables_dir / "summary_auc_wide.csv")
    selected_k.to_csv(tables_dir / "selected_feature_counts.csv")

    report_df = pd.concat(
        {
            "SelectedFeatureCount": selected_k,
            "Accuracy": accuracy_wide,
            "AUC": auc_wide,
        },
        axis=1,
    )
    report_df.to_csv(tables_dir / "paper_ready_summary.csv")


def available_models(model_names: Sequence[str], skip_missing: bool, allow_empty: bool = False) -> List[str]:
    resolved = []
    for name in model_names:
        norm = normalize_model_name(name)
        if norm == "AutoGluon" and import_optional("autogluon.tabular") is None:
            if skip_missing:
                log("[warn] AutoGluon is missing and will be skipped.")
                continue
            raise ModuleNotFoundError("autogluon.tabular is not installed.")
        if norm == "XGBoost" and import_optional("xgboost") is None:
            if skip_missing:
                log("[warn] XGBoost is missing and will be skipped.")
                continue
            raise ModuleNotFoundError("xgboost is not installed.")
        if norm == "CatBoost" and import_optional("catboost") is None:
            if skip_missing:
                log("[warn] CatBoost is missing and will be skipped.")
                continue
            raise ModuleNotFoundError("catboost is not installed.")
        resolved.append(norm)
    if not resolved and not allow_empty:
        raise RuntimeError("No trainable models are available.")
    return resolved


def select_best_feature_set_for_plots(results_df: pd.DataFrame) -> Tuple[str, str, str]:
    combo_df = results_df[results_df["FeatureSet"].str.contains(r"\+", regex=True)].copy()
    if combo_df.empty:
        best_row = results_df.sort_values("AUC", ascending=False).iloc[0]
        return str(best_row["FeatureSet"]), str(best_row["Model"]), "Highest AUC among all selected feature sets."
    best_row = combo_df.sort_values("AUC", ascending=False).iloc[0]
    return str(best_row["FeatureSet"]), str(best_row["Model"]), "Highest AUC among fingerprint combinations on internal test set."


def evaluate_autogluon_internal_models(
    predictor_path: Path,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    module = import_optional("autogluon.tabular")
    if module is None:
        raise ModuleNotFoundError("autogluon.tabular is not installed.")
    predictor = module.TabularPredictor.load(str(predictor_path))
    leaderboard = predictor.leaderboard(silent=True)
    rows = []
    for model_name in leaderboard["model"].tolist():
        try:
            y_pred = predictor.predict(X_test, model=model_name).to_numpy(dtype=int)
            y_score = positive_scores(predictor.predict_proba(X_test, model=model_name))
            metrics = compute_metrics(y_test, y_pred, y_score)
            rows.append({"Model": model_name, **metrics})
        except Exception as exc:
            log(f"[warn] Failed to score AutoGluon internal model {model_name}: {exc}")
    metrics_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    metrics_df.to_csv(output_dir / "autogluon_internal_metrics.csv", index=False)
    return metrics_df


def main() -> int:
    args = parse_args()
    if args.check_env:
        check_runtime_dependencies()
        return 0

    config = default_config()
    if args.config is not None:
        config.update(load_json(args.config))
    config = merge_cli_into_config(args, config)

    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(output_root / "resolved_config.json", config)

    discovered = discover_feature_sets(data_root)
    feature_sets = resolve_feature_sets(config, sorted(discovered))
    if not config.get("skip_plots", False) and import_optional("matplotlib.pyplot") is None:
        log("[warn] matplotlib is missing, plots will be skipped in this run.")
        config["skip_plots"] = True

    final_models = available_models(config["models"], skip_missing=config.get("skip_missing_models", True))
    ifs_models = available_models(
        config.get("ifs_models", []),
        skip_missing=config.get("skip_missing_models", True),
        allow_empty=True,
    )
    if not ifs_models:
        fallback = next((m for m in final_models if m != "AutoGluon"), final_models[0])
        ifs_models = [fallback]
        log(f"[warn] No IFS models available. Fallback to {fallback}.")
    if config.get("per_model_feature_selection", True):
        ifs_models = list(dict.fromkeys(final_models + ifs_models))

    log(f"[run] feature sets: {feature_sets}")
    log(f"[run] final models: {final_models}")
    log(f"[run] ifs models: {ifs_models}")

    # Use the first selected feature set to generate the split manifest.
    reference_df = load_full_feature_matrix(discovered[feature_sets[0]])
    _, reference_y = select_feature_columns(reference_df)
    split_data = create_split(
        reference_y.to_numpy(dtype=int),
        test_size=float(config["test_size"]),
        validation_size=float(config["validation_size"]),
        seed=int(config["random_seed"]),
    )
    manifest = build_split_manifest(reference_y.to_numpy(dtype=int), split_data)
    manifest.to_csv(output_root / "split_manifest.csv", index=False)

    all_results = []
    best_artifacts: Dict[str, dict] = {}

    for feature_set in feature_sets:
        log(f"[run] processing feature set: {feature_set}")
        feature_dir = output_root / "feature_sets" / feature_set
        feature_dir.mkdir(parents=True, exist_ok=True)

        full_df = load_full_feature_matrix(discovered[feature_set])
        existing_artifacts = load_existing_feature_artifacts(feature_dir, full_df, split_data)
        if existing_artifacts is not None:
            log(f"[resume] found existing completed outputs for {feature_set}, skipping recomputation.")
            all_results.append(existing_artifacts["metrics"])
            best_artifacts[feature_set] = existing_artifacts
            continue

        X_full, y_full = select_feature_columns(full_df)

        X_train = X_full.iloc[split_data.train_idx].reset_index(drop=True)
        y_train = y_full.iloc[split_data.train_idx].reset_index(drop=True)
        X_val = X_full.iloc[split_data.val_idx].reset_index(drop=True)
        y_val = y_full.iloc[split_data.val_idx].reset_index(drop=True)
        X_test = X_full.iloc[split_data.test_idx].reset_index(drop=True)
        y_test = y_full.iloc[split_data.test_idx].reset_index(drop=True)

        ranked_features = rank_features(
            X_train,
            y_train,
            method=config["selector_method"],
            fallback=config.get("selector_fallback", "mutual_info"),
        )
        pd.DataFrame(
            {"rank": np.arange(1, len(ranked_features) + 1), "feature": ranked_features}
        ).to_csv(feature_dir / "ranked_features.csv", index=False)

        counts = adaptive_ifs_counts(
            X_train.shape[1],
            fixed_step=config.get("ifs_step"),
            max_points=int(config.get("ifs_max_points", 60)),
        )
        ifs_df = score_models_for_ifs(
            ifs_models,
            X_train,
            y_train,
            X_val,
            y_val,
            counts,
            ranked_features,
            config,
            feature_dir,
        )
        ifs_df.to_csv(feature_dir / "ifs_results.csv", index=False)

        best_k_by_model = choose_best_feature_counts_by_model(
            ifs_df,
            final_models,
            reference_model=config["ifs_reference_model"],
        )
        pd.DataFrame(
            [{"Model": model_name, "SelectedFeatureCount": best_k_by_model[model_name]} for model_name in final_models]
        ).to_csv(feature_dir / "selected_feature_counts_by_model.csv", index=False)
        feature_rows = []
        for model_name in final_models:
            selected_features = ranked_features[: best_k_by_model[model_name]]
            feature_rows.extend({"Model": model_name, "Feature": feat} for feat in selected_features)
        pd.DataFrame(feature_rows).to_csv(feature_dir / "selected_features_by_model.csv", index=False)
        log(f"[run] {feature_set} selected feature counts = {best_k_by_model}")

        if not config.get("skip_plots", False):
            plot_ifs_curve(
                ifs_df,
                feature_dir / "ifs_curve.png",
                reference_model=config["ifs_reference_model"],
                feature_set=feature_set,
            )

        train_val_idx = np.concatenate([split_data.train_idx, split_data.val_idx])
        X_train_pool = X_full.iloc[train_val_idx].reset_index(drop=True)
        y_train_final = y_full.iloc[train_val_idx].reset_index(drop=True)
        y_test_np = y_test.to_numpy(dtype=int)

        rows = []
        roc_rows = []
        autogluon_path = None
        X_test_by_model: Dict[str, pd.DataFrame] = {}
        for model_name in final_models:
            log(f"[model] {feature_set} -> {model_name}")
            try:
                selected_features = ranked_features[: best_k_by_model[model_name]]
                X_train_final = X_train_pool[selected_features]
                X_test_final = X_test[selected_features]
                X_test_by_model[model_name] = X_test_final
                if model_name == "AutoGluon":
                    autogluon_path = feature_dir / "autogluon_model"
                    predictor, y_pred, y_score = fit_predict_autogluon_with_retry(
                        X_train_final,
                        y_train_final,
                        X_test_final,
                        predictor_path=autogluon_path,
                        presets=config["autogluon_presets"],
                        time_limit=config.get("autogluon_time_limit"),
                        retry_without_xgb=config.get("autogluon_retry_without_xgb", True),
                        extra_fit_kwargs=config.get("autogluon_fit_kwargs", {}),
                    )
                    del predictor
                else:
                    y_pred, y_score = fit_predict_classical(
                        model_name,
                        X_train_final,
                        y_train_final,
                        X_test_final,
                        int(config["random_seed"]),
                    )
                metrics = compute_metrics(y_test_np, y_pred, y_score)
                row = {
                    "FeatureSet": feature_set,
                    "Model": model_name,
                    "SelectedFeatureCount": best_k_by_model[model_name],
                    **metrics,
                }
                rows.append(row)

                if len(np.unique(y_test_np)) == 2:
                    fpr, tpr, _ = roc_curve(y_test_np, y_score)
                    roc_rows.extend(
                        {
                            "FeatureSet": feature_set,
                            "Model": model_name,
                            "FPR": float(x),
                            "TPR": float(y),
                            "AUC": metrics["AUC"],
                        }
                        for x, y in zip(fpr, tpr)
                    )
            except Exception as exc:
                log(f"[warn] final model failed for {feature_set}/{model_name}: {exc}")

        metrics_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        roc_df = pd.DataFrame(roc_rows)
        metrics_df.to_csv(feature_dir / "test_metrics.csv", index=False)
        roc_df.to_csv(feature_dir / "roc_curve_data.csv", index=False)

        all_results.append(metrics_df)
        best_artifacts[feature_set] = {
            "feature_dir": feature_dir,
            "metrics": metrics_df,
            "roc": roc_df,
            "selected_k_by_model": best_k_by_model,
            "X_test_by_model": X_test_by_model,
            "y_test": y_test_np,
            "autogluon_path": autogluon_path,
        }

    all_results_df = pd.concat(all_results, ignore_index=True)
    save_summary_tables(all_results_df, output_root)

    best_feature_set, best_model, selection_basis = select_best_feature_set_for_plots(all_results_df)
    best_dir = output_root / "best_feature_combination"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_info = {
        "best_feature_set": best_feature_set,
        "best_model": best_model,
        "selection_basis": selection_basis,
    }
    dump_json(best_dir / "best_selection.json", best_info)

    best_metrics_df = best_artifacts[best_feature_set]["metrics"]
    best_metrics_df.to_csv(best_dir / "model_metrics.csv", index=False)
    best_roc_df = best_artifacts[best_feature_set]["roc"]
    best_roc_df.to_csv(best_dir / "roc_curve_data.csv", index=False)

    if not config.get("skip_plots", False):
        plot_model_bar(
            best_metrics_df,
            best_dir / "model_metrics_bar.png",
            title=f"Model Comparison - {best_feature_set}",
        )
        plot_roc_curves(
            best_roc_df,
            best_dir / "roc_curves.png",
            title=f"ROC Curves - {best_feature_set}",
        )

    if best_model == "AutoGluon":
        ag_path = best_artifacts[best_feature_set]["autogluon_path"]
        if ag_path is not None:
            internal_df = evaluate_autogluon_internal_models(
                ag_path,
                best_artifacts[best_feature_set]["X_test_by_model"]["AutoGluon"],
                best_artifacts[best_feature_set]["y_test"],
                best_dir,
            )
            if not config.get("skip_plots", False):
                plot_autogluon_internal_heatmap(
                    internal_df,
                    best_dir / "autogluon_internal_heatmap.png",
                    title="AutoGluon Internal Models Metrics",
                )

    log(f"[done] outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("[abort] interrupted by user.")
        raise SystemExit(130)
