AquaReact-ML: an AutoGluon-based framework for prioritizing mild aqueous small-molecule reactant pairs

This repository contains the code used for the final molecular-fingerprint
modeling workflow in the drug-pair contraindication prediction study.

The final paper result is based on:

`ECFP4 + FCFP4 + MACCS` (`E+F+M`) with `AutoGluon`

The corresponding final result directory in the original workstation is:

`/home/xwl/药物禁忌/outputs/run_20260406_103701_autogluon_boosted`

Large outputs, trained models, raw data, temporary logs, and virtual
environments are intentionally excluded from this GitHub-ready folder.

## Repository Structure

```text
drug_contraindication_code/
├── README.md
├── CODE_MANIFEST.md
├── requirements.txt
├── configs/
│   ├── default_experiment.json
│   └── autogluon_boosted.json
├── docs/
└── scripts/
```

## Main Workflow

1. Fingerprint-based model training and feature selection

```bash
python scripts/train_fingerprint_models.py --config configs/default_experiment.json
```

2. Boosted AutoGluon training for the final E+F+M model

```bash
python scripts/train_fingerprint_models.py --config configs/autogluon_boosted.json
```

3. Merge boosted AutoGluon results into the final summary table

```bash
python scripts/merge_boosted_autogluon_results.py
```

4. Generate IFS, ROC, and model-comparison figures

```bash
python scripts/plot_boosted_ifs_curves.py
python scripts/plot_boosted_model_comparison.py
python scripts/compose_efm_three_panel_figure.py
```

5. Run probability calibration

```bash
python scripts/run_autogluon_calibration_experiment.py
python scripts/compose_calibration_summary_figure.py
```

6. Run interpretability analysis

```bash
python scripts/run_autogluon_explainability_analysis.py
python scripts/compose_explainability_summary_figure.py
```

7. Run robustness and pair-representation analyses

```bash
python scripts/run_autogluon_split_sensitivity.py
python scripts/compose_robustness_roc_metrics_figure.py
python scripts/run_pair_representation_comparison_fair.py
python scripts/compose_pair_representation_comparison_figure.py
```

8. Generate drug-discovery-oriented synthesis-planning examples

```bash
python scripts/generate_synthesis_planning_cases.py
/home/xwl/miniconda3/bin/python scripts/plot_synthesis_planning_cases_overview.py
```

## Environment

For the fingerprint and AutoGluon pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

Some structure-visualization scripts require RDKit. In the original workstation,
RDKit was available from the conda Python environment:

```bash
/home/xwl/miniconda3/bin/python scripts/plot_synthesis_planning_cases_overview.py
```

## Notes

This code package focuses only on the molecular-fingerprint `E+F+M + AutoGluon`
workflow used for the final manuscript results. Other exploratory or unrelated
training pipelines are not included here.
