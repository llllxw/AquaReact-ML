# Revised F+M / AutoGluon reproduction

This entry point reproduces the **revised FCFP4 + MACCS workflow**. It does not
change the legacy plotting, calibration, interpretation or benchmark scripts.
Those scripts and `data/combined-data.csv` describe the original submission,
not this revised dataset. Run all commands from the repository root.

## Data release

Files are in `data/dataset/`. No external workstation files are needed.

| File | Rows | Purpose |
| --- | ---: | --- |
| `all_samples.csv` | 19,456 | All eligible unordered reactant pairs |
| `positive_samples.csv` | 4,964 | Mild aqueous class (`label=1`) |
| `negative_samples.csv` | 14,492 | Eligible high-temperature control pool (`label=0`) |
| `balanced_samples.csv` | 9,928 | All positives and 4,964 randomly selected controls |
| `reserved_negative_samples.csv` | 9,528 | Remaining controls; never used for fitting or selection |
| `cohort_membership.csv` | 19,456 | Balanced/reserved membership by sample ID |
| `split_manifest.csv` | 49,640 | Fixed train/validation/test membership in each of five folds |
| `reaction_records.csv.gz` | 50,473 | All supporting sample-to-ORD-record links and reaction structures |

The ORD snapshot contained 2,376,120 records, of which 555,132 had explicit
numeric temperatures. Positive records required water during the reaction
stage and either a numeric temperature of 20-40 degrees C or an uncontradicted
qualitative ambient-temperature annotation. Qualitative room temperature was
not converted into an invented numeric temperature. Mixed aqueous solvents
were permitted. Of the 4,964 positive pairs, 745 had qualifying numeric evidence
and 4,219 were retained on qualitative ambient evidence.

The initial >=100 degrees C screen contained 126,888 records. Both classes
required two distinct parseable reactants and usable product structures.
Structures were canonicalized, atom maps removed, and unordered reactant pairs
deduplicated; the two stored reactant columns use the frozen canonical order.
High-temperature pairs conflicting with the mild aqueous class were excluded,
leaving 14,492 controls. These are **condition-associated controls**, not proven
failures under mild aqueous conditions; water is not required in this class.

The sample tables supply labels and reactant SMILES for training. Join the
record table on `sample_id` for ORD reaction IDs, source-file hashes and relative
ORD paths, record identifiers, temperature evidence, solvents and products.
All supporting records/product sets are retained, rather than selecting one
product arbitrarily. Missing numeric temperatures are intentional where the
evidence is qualitative. Products and conditions are provenance only: **they
are not model inputs**. Gzip CSV can be read directly using `pandas.read_csv`.

`configs/fm_release_manifest.json` contains SHA-256 hashes of the released
data, configuration and archival reference results. The `verify` command also
reconstructs the undersampling and split memberships independently.

## Training protocol

1. Select 4,964 controls without replacement using NumPy `default_rng(42)`;
   retain all positives. Keep the remaining 9,528 controls separate.
2. Use `KFold(5, shuffle=True, random_state=42)`, without stratification. Every
   balanced sample is tested exactly once. Split each development fold again
   with `train_test_split(test_size=0.25, random_state=42+fold)` (folds 1-5),
   giving approximately 60:20:20 train/validation/test proportions. Folds 1-3
   have 5,956/1,986/1,986 samples; folds 4-5 have 5,957/1,986/1,985.
3. Compute FCFP4 (Morgan feature invariants, radius 2, 2,048 bits, no chirality)
   and MACCS (167 bits including the unused zero bit) using RDKit. Concatenate
   as **[F_A, F_B, M_A, M_B]**, giving 4,430 raw pair features. No ECFP4 or
   RDKit-path fingerprints are used in this entry point.
4. Within each training fold, remove constant bits and rank the rest by
   empirical binary mutual-information mRMR: relevance minus mean redundancy
   (MID). Zero-relevance nonconstant bits remain eligible. Neither validation
   nor outer test labels enter ranking. Matrix and ranking checks compare the
   rebuilt inputs with the archived paper inputs.
5. Perform incremental feature selection (IFS) using held-out **validation
   AUROC**, not test AUROC. Screen k=1, 400, 800, ... and all eligible features
   using 120-second AutoGluon fits (3-fold internal bagging, no stacking,
   `good_quality`, light hyperparameters). Refine the three best k values plus
   the full eligible set using 900-second fits (5-fold internal bagging, one
   stack level, `best_quality`, the explicitly stored enhanced portfolio).
   Select the largest validation AUROC, breaking ties with smaller k.
6. Refit the selected subset on training plus validation data using the stored
   1,800-second enhanced configuration. Evaluate once on that outer test fold.
   AutoGluon's internal bagging is separate from external five-fold evaluation.
   Dynamic stacking and `refit_full` are disabled; bagged model files are kept.
   Classification uses the predictor's decision threshold, which is recorded.
7. Aggregate five fold metrics as **mean plus/minus sample SD** (`ddof=1`),
   displayed to three decimal places. AP is average precision, not trapezoidal
   PR area. Sensitivity is recall. These SDs are not confidence intervals.

The full concrete hyperparameter dictionaries, not just preset names, are in
`configs/fm_revision.json`. Primary splits are pair-level random splits, not
compound-disjoint splits. No reserved controls enter any training stage.

## Commands

Use Python 3.12 and the separate revised dependencies:

```bash
python3.12 -m venv .venv-fm
source .venv-fm/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-fm.txt
python scripts/fm_autogluon.py verify
python -m unittest discover -s tests -v
python scripts/fm_autogluon.py prepare --workers 2
python scripts/fm_autogluon.py rank --threads 12
python scripts/fm_autogluon.py smoke-test --out outputs/fm_smoke --threads 2
```

Full selection and training for all five folds:

```bash
python scripts/fm_autogluon.py train --out outputs/fm_revision --workers 2 --threads 12 --selection rerun
```

To refit using the **frozen paper-selected subsets** rather than rerun IFS:

```bash
python scripts/fm_autogluon.py train --out outputs/fm_paper --workers 2 --threads 12 --selection paper
```

`paper` still rebuilds/verifies fold-local ranking and trains new models. It
does not copy archived test predictions into new results. Published selected
feature counts for folds 1-5 are 4,242, 4,249, 4,000, 4,231 and 4,239.

Use `--fold 1` to run one fold, then run `summarize` after all five complete:

```bash
python scripts/fm_autogluon.py summarize --out outputs/fm_revision
python scripts/fm_autogluon.py reference-summary --out outputs/paper_reference
```

`reference-summary` explicitly reports archived results and performs no
training. Its output names start with `reference_` to avoid confusion.

## Outputs and resources

Each fold saves training IDs, ranking, IFS scores and validation predictions,
selected features, selection criteria, final predictor files, test predictions,
ROC data, metrics and role provenance. Root-level `summary.csv` contains raw
mean/SD and three-decimal display strings. Time budgets are per fit, not per
fold. `--workers 2 --threads 12` permits two folds concurrently with up to 12
CPUs each; use fewer workers on a smaller machine. GPUs are not required.
Allow several hours and substantial disk space for full IFS: all candidate
predictors are retained. Keep enough RAM for concurrent AutoGluon fits; reduce
workers to one if memory is limited.

Completed fits can resume with the same command. A changed training script,
configuration, selection mode or thread count requires a new output directory.
Interrupted model directories are reported, not silently reused or deleted;
inspect and move an incomplete directory before retrying. Do not run two jobs
against the same output directory. Do not use Python's `-O` option, which would
disable verification assertions.

## Reference results and limitations

`reference/fm/` stores five-fold test predictions/metrics, selected features,
screening/refinement scores and original selections. Local columns are remapped
to F+M; `original_column` preserves the original all-fingerprint column index.
Archived mean AUROC is 0.962 with SD 0.004 and AP is 0.961 with SD 0.007.
No fitted model weights are distributed, and a refit on all 9,928 samples is
not substituted for the five held-out evaluations.

The later `supplementary_full_ifs_*` files document the expanded enhanced-IFS
plot experiment. It found different validation-optimal k values for some folds
but did not replace the frozen predictors underlying the published test
results. It is not silently used to change the selections in `paper` mode.

This release reproduces the fixed inputs and training procedure. Time-limited
AutoGluon runs can train different portfolios on different hardware, even with
the pinned versions, so newly fitted scores/subsets need not be bit-identical.
The enhanced configuration was developed after earlier exploratory benchmark
results were available; these folds are not a new prospective confirmation of
all design choices. Installing dependencies and a small integration test are
not equivalent to rerunning the full manuscript experiment.
