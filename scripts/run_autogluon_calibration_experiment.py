#!/usr/bin/env python3
"""Run probability calibration experiment for an AutoGluon fingerprint model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from train_fingerprint_models import (
    compute_metrics,
    create_split,
    default_config,
    discover_feature_sets,
    ensure_matplotlib,
    fit_predict_autogluon_with_retry,
    import_optional,
    load_full_feature_matrix,
    load_json,
    positive_scores,
    save_figure_multi,
    select_feature_columns,
)


METHOD_ORDER = ["Uncalibrated", "Platt", "Isotonic"]
METHOD_COLORS = {
    "Uncalibrated": "#6f7682",
    "Platt": "#f08b78",
    "Isotonic": "#6fa8dc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoGluon calibration experiment.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing outputs and config.")
    parser.add_argument("--feature-set", default="E+F+M", help="Feature set to calibrate.")
    parser.add_argument("--model", default="AutoGluon", help="Model name. Only AutoGluon is currently supported.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for calibration outputs. Defaults to feature_sets/<feature_set>/calibration.",
    )
    parser.add_argument("--n-bins", type=int, default=10, help="Number of bins for ECE and calibration curve.")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached train-only IFS model and retrain AutoGluon on the train split.",
    )
    return parser.parse_args()


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += abs(acc - conf) * float(np.sum(mask)) / total
    return float(ece)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1 - 1e-8)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["BrierScore"] = float(brier_score_loss(y_true, y_prob))
    metrics["LogLoss"] = float(log_loss(y_true, y_prob))
    metrics["ECE"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    metrics["MeanPredProb"] = float(np.mean(y_prob))
    return metrics


def load_split_indices(run_dir: Path) -> Dict[str, np.ndarray]:
    manifest_path = run_dir / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    return {
        "train": manifest.loc[manifest["split"] == "train", "row_id"].to_numpy(dtype=int),
        "validation": manifest.loc[manifest["split"] == "validation", "row_id"].to_numpy(dtype=int),
        "test": manifest.loc[manifest["split"] == "test", "row_id"].to_numpy(dtype=int),
    }


def load_selected_features(feature_dir: Path, model_name: str) -> list[str]:
    selected_path = feature_dir / "selected_features_by_model.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected features table: {selected_path}")
    selected_df = pd.read_csv(selected_path)
    subset = selected_df[selected_df["Model"] == model_name]
    if subset.empty:
        raise RuntimeError(f"No selected features found for model {model_name}.")
    return [str(x) for x in subset["Feature"].tolist()]


def load_selected_feature_count(feature_dir: Path, model_name: str) -> int:
    counts_path = feature_dir / "selected_feature_counts_by_model.csv"
    if not counts_path.exists():
        raise FileNotFoundError(f"Missing selected feature counts table: {counts_path}")
    counts_df = pd.read_csv(counts_path)
    subset = counts_df[counts_df["Model"] == model_name]
    if subset.empty:
        raise RuntimeError(f"No selected feature count found for model {model_name}.")
    return int(subset.iloc[0]["SelectedFeatureCount"])


def load_feature_matrices(run_dir: Path, feature_set: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, np.ndarray]]:
    config_path = run_dir / "resolved_config.json"
    config = default_config()
    if config_path.exists():
        config.update(load_json(config_path))
    discovered = discover_feature_sets(Path(config["data_root"]))
    full_df = load_full_feature_matrix(discovered[feature_set])
    X_full, y_full = select_feature_columns(full_df)
    split_idx = load_split_indices(run_dir)
    return X_full, y_full, split_idx


def load_autogluon_predictor(predictor_path: Path):
    module = import_optional("autogluon.tabular")
    if module is None:
        raise ModuleNotFoundError("autogluon.tabular is not installed.")
    return module.TabularPredictor.load(str(predictor_path))


def resolve_train_only_predictor(
    run_dir: Path,
    feature_dir: Path,
    feature_set: str,
    selected_features: list[str],
    selected_k: int,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    split_idx: Dict[str, np.ndarray],
    *,
    force_retrain: bool,
) -> Tuple[Path, str]:
    cached_path = feature_dir / "ifs_models" / f"ag_k{selected_k}"
    if cached_path.exists() and not force_retrain:
        return cached_path, "cached_train_only_ifs_model"

    config = default_config()
    config_path = run_dir / "resolved_config.json"
    if config_path.exists():
        config.update(load_json(config_path))

    output_dir = feature_dir / "calibration" / "train_only_autogluon_model"
    X_train = X_full.iloc[split_idx["train"]][selected_features].reset_index(drop=True)
    y_train = y_full.iloc[split_idx["train"]].reset_index(drop=True)
    X_val = X_full.iloc[split_idx["validation"]][selected_features].reset_index(drop=True)

    fit_predict_autogluon_with_retry(
        X_train,
        y_train,
        X_val,
        predictor_path=output_dir,
        presets=config["autogluon_presets"],
        time_limit=config.get("autogluon_time_limit"),
        retry_without_xgb=config.get("autogluon_retry_without_xgb", True),
        extra_fit_kwargs=config.get("autogluon_fit_kwargs", {}),
    )
    return output_dir, "retrained_train_only_model"


def fit_calibrators(y_val: np.ndarray, prob_val: np.ndarray) -> Dict[str, object]:
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(prob_val.reshape(-1, 1), y_val)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(prob_val, y_val)
    return {"Platt": platt, "Isotonic": isotonic}


def transform_probabilities(calibrators: Dict[str, object], prob: np.ndarray) -> Dict[str, np.ndarray]:
    transformed = {
        "Uncalibrated": np.asarray(prob, dtype=float),
        "Platt": calibrators["Platt"].predict_proba(prob.reshape(-1, 1))[:, 1],
        "Isotonic": calibrators["Isotonic"].predict(prob),
    }
    return {k: np.clip(v, 1e-8, 1 - 1e-8) for k, v in transformed.items()}


def build_probability_table(
    row_ids: np.ndarray,
    y_true: np.ndarray,
    prob_map: Dict[str, np.ndarray],
    split_name: str,
) -> pd.DataFrame:
    data = {"row_id": row_ids, "split": split_name, "y_true": y_true.astype(int)}
    for method_name in METHOD_ORDER:
        data[f"prob_{method_name.lower()}"] = prob_map[method_name]
    return pd.DataFrame(data)


def plot_calibration_curves(
    y_true: np.ndarray,
    prob_map: Dict[str, np.ndarray],
    output_path: Path,
    *,
    n_bins: int,
    title: str,
) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    for method_name in METHOD_ORDER:
        frac_pos, mean_pred = calibration_curve(y_true, prob_map[method_name], n_bins=n_bins, strategy="quantile")
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=METHOD_COLORS[method_name],
            label=method_name,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.2, label="Ideal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def plot_brier_ece_comparison(metrics_df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    plt = ensure_matplotlib()
    plot_df = metrics_df.set_index("Method").loc[METHOD_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    for ax, metric_name in zip(axes, ["BrierScore", "ECE"]):
        values = plot_df[metric_name].astype(float).tolist()
        colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
        bars = ax.bar(METHOD_ORDER, values, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_title(metric_name)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(axis="x", rotation=15)
        y_max = max(values) * 1.18 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, y_max)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle(title)
    fig.tight_layout()
    save_figure_multi(fig, output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.model != "AutoGluon":
        raise ValueError("This script currently supports AutoGluon only.")

    run_dir = args.run_dir.resolve()
    feature_dir = run_dir / "feature_sets" / args.feature_set
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature set directory not found: {feature_dir}")

    output_dir = args.output_dir.resolve() if args.output_dir else feature_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_features = load_selected_features(feature_dir, args.model)
    selected_k = load_selected_feature_count(feature_dir, args.model)
    X_full, y_full, split_idx = load_feature_matrices(run_dir, args.feature_set)

    X_val = X_full.iloc[split_idx["validation"]][selected_features].reset_index(drop=True)
    y_val = y_full.iloc[split_idx["validation"]].to_numpy(dtype=int)
    X_test = X_full.iloc[split_idx["test"]][selected_features].reset_index(drop=True)
    y_test = y_full.iloc[split_idx["test"]].to_numpy(dtype=int)

    predictor_path, predictor_source = resolve_train_only_predictor(
        run_dir,
        feature_dir,
        args.feature_set,
        selected_features,
        selected_k,
        X_full,
        y_full,
        split_idx,
        force_retrain=args.force_retrain,
    )
    predictor = load_autogluon_predictor(predictor_path)

    prob_val_uncal = positive_scores(predictor.predict_proba(X_val))
    prob_test_uncal = positive_scores(predictor.predict_proba(X_test))

    calibrators = fit_calibrators(y_val, prob_val_uncal)
    prob_val_map = transform_probabilities(calibrators, prob_val_uncal)
    prob_test_map = transform_probabilities(calibrators, prob_test_uncal)

    metrics_rows = []
    for method_name in METHOD_ORDER:
        metrics_rows.append(
            {
                "Method": method_name,
                "Split": "test",
                **compute_calibration_metrics(y_test, prob_test_map[method_name], n_bins=args.n_bins),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "calibration_metrics_test.csv", index=False)

    val_prob_df = build_probability_table(split_idx["validation"], y_val, prob_val_map, "validation")
    test_prob_df = build_probability_table(split_idx["test"], y_test, prob_test_map, "test")
    pd.concat([val_prob_df, test_prob_df], ignore_index=True).to_csv(
        output_dir / "calibration_probabilities.csv",
        index=False,
    )

    plot_calibration_curves(
        y_test,
        prob_test_map,
        output_dir / "calibration_curve_test.png",
        n_bins=args.n_bins,
        title=f"Calibration Curves - {args.feature_set} ({args.model})",
    )
    plot_brier_ece_comparison(
        metrics_df,
        output_dir / "brier_ece_comparison_test.png",
        title=f"Brier / ECE Comparison - {args.feature_set} ({args.model})",
    )

    meta_df = pd.DataFrame(
        [
            {
                "FeatureSet": args.feature_set,
                "Model": args.model,
                "SelectedFeatureCount": selected_k,
                "PredictorPath": str(predictor_path),
                "PredictorSource": predictor_source,
                "CalibrationBins": args.n_bins,
            }
        ]
    )
    meta_df.to_csv(output_dir / "calibration_run_metadata.csv", index=False)

    print(output_dir / "calibration_metrics_test.csv")
    print(output_dir / "calibration_curve_test.png")
    print((output_dir / "calibration_curve_test.png").with_suffix(".svg"))
    print(output_dir / "brier_ece_comparison_test.png")
    print((output_dir / "brier_ece_comparison_test.png").with_suffix(".svg"))
    print(output_dir / "calibration_probabilities.csv")
    print(output_dir / "calibration_run_metadata.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
