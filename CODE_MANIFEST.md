# Code Manifest

This folder contains only the code needed for the final molecular-fingerprint
`E+F+M + AutoGluon` manuscript workflow.

## Core Modeling

- `scripts/train_fingerprint_models.py`: main pipeline for fingerprint loading,
  feature ranking, incremental feature selection, model training, test-set
  evaluation, ROC curves, model metrics, and AutoGluon internal diagnostics.
- `configs/default_experiment.json`: baseline fingerprint experiment settings.
- `configs/autogluon_boosted.json`: final boosted AutoGluon settings used for
  the manuscript result.
- `scripts/merge_boosted_autogluon_results.py`: combines updated AutoGluon
  results with the final model-comparison summary.

## IFS, ROC, and Performance Figures

- `scripts/plot_boosted_ifs_curves.py`: plots IFS curves for the selected
  fingerprint combination.
- `scripts/plot_boosted_model_comparison.py`: plots model metric comparisons
  and ROC curves.
- `scripts/compose_efm_three_panel_figure.py`: combines IFS, ROC, and metric
  plots into the main E+F+M performance figure.
- `scripts/plot_autogluon_validation_selection_heatmap.py`: plots AutoGluon
  internal validation-selection heatmaps.

## Probability Calibration

- `scripts/run_autogluon_calibration_experiment.py`: fits Platt scaling and
  isotonic calibration on validation probabilities and evaluates calibration on
  the test set.
- `scripts/compose_calibration_summary_figure.py`: creates the calibration
  summary figure.

## Interpretability

- `scripts/run_autogluon_explainability_analysis.py`: computes permutation
  feature importance and fingerprint-source contribution statistics.
- `scripts/compose_explainability_summary_figure.py`: creates the final
  interpretability summary figure.

## Robustness and Pair Representation

- `scripts/run_autogluon_split_sensitivity.py`: evaluates the final E+F+M
  AutoGluon model under compound-disjoint and scaffold split settings.
- `scripts/compose_robustness_roc_metrics_figure.py`: creates robustness ROC
  and metric summary figures.
- `scripts/run_pair_representation_comparison_fair.py`: compares pair
  representation strategies under a fair evaluation setup.
- `scripts/compose_pair_representation_comparison_figure.py`: creates the pair
  representation comparison figure.

## Drug-Discovery-Oriented Case Analysis

- `scripts/generate_synthesis_planning_cases.py`: selects high-confidence
  predicted positive reactant pairs and annotates medicinal-chemistry motifs.
- `scripts/plot_synthesis_planning_cases_overview.py`: draws the 2D structure
  overview figure for representative high-confidence cases.

## Documentation

- `docs/execution_checklist.md`: local execution checklist.
- `docs/install_commands.md`: local install commands used during development.
