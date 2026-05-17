# AquaReact-ML

## Introduction

AquaReact-ML is an AutoGluon-based framework for prioritizing mild aqueous
small-molecule reactant pairs. It identifies reactant pairs associated with mild
aqueous reaction conditions and supports earlier-stage reaction planning through
molecular fingerprint screening, pairwise representation comparison, mRMR-based
feature selection, incremental feature selection, algorithm benchmarking,
probability calibration, robustness evaluation, and model interpretation.


## Environment Requirement

The code has been checked under Python 3.12.7. The required packages are listed
in `requirements.txt` and include the following:

```text
numpy
pandas
scikit-learn
matplotlib
mrmr-selection
xgboost
catboost
autogluon.tabular
```

Install the environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

Some structure-visualization scripts require RDKit. If RDKit is not available
in the active Python environment, install it through conda or another
environment manager before running the structure overview script.

## Source codes

`data/combined-data.csv`: curated reactant-pair dataset used by the
AquaReact-ML workflow.

`configs/default_experiment.json`: baseline molecular-fingerprint experiment
settings.

`configs/autogluon_boosted.json`: final boosted AutoGluon settings used for the
manuscript result.

`scripts/train_fingerprint_models.py`: main pipeline for fingerprint loading,
feature ranking, incremental feature selection, model training, test-set
evaluation, ROC curves, model metrics, and AutoGluon internal diagnostics.

`scripts/merge_boosted_autogluon_results.py`: merges updated boosted AutoGluon
results into the final model-comparison summary.

`scripts/plot_boosted_ifs_curves.py`: plots incremental feature selection
curves for the selected fingerprint combination.

`scripts/plot_boosted_model_comparison.py`: plots model metric comparisons and
ROC curves.

`scripts/compose_efm_three_panel_figure.py`: combines the IFS, ROC, and metric
plots into the main ECFP4+FCFP4+MACCS performance figure.

`scripts/plot_autogluon_validation_selection_heatmap.py`: plots AutoGluon
internal validation-selection heatmaps.

`scripts/run_autogluon_calibration_experiment.py`: fits Platt scaling and
isotonic calibration on validation probabilities and evaluates calibration on
the independent test set.

`scripts/compose_calibration_summary_figure.py`: creates the calibration
summary figure.

`scripts/run_autogluon_explainability_analysis.py`: computes permutation
feature importance and fingerprint-source contribution statistics.

`scripts/compose_explainability_summary_figure.py`: creates the final
interpretability summary figure.

`scripts/run_autogluon_split_sensitivity.py`: evaluates the final
ECFP4+FCFP4+MACCS AutoGluon model under compound-disjoint and scaffold split
settings.

`scripts/compose_robustness_roc_metrics_figure.py`: creates robustness ROC and
metric summary figures.

`scripts/run_pair_representation_comparison_fair.py`: compares pair
representation strategies under a fair evaluation setup.

`scripts/compose_pair_representation_comparison_figure.py`: creates the pair
representation comparison figure.

`scripts/generate_synthesis_planning_cases.py`: selects high-confidence
predicted positive reactant pairs and annotates medicinal-chemistry motifs.

`scripts/plot_synthesis_planning_cases_overview.py`: draws the 2D structure
overview figure for representative high-confidence cases.

`CODE_MANIFEST.md`: concise manifest of the included code package.

`docs/execution_checklist.md`: local execution checklist.

`docs/install_commands.md`: local installation commands used during
development.
