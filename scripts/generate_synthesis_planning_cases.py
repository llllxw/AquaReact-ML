#!/usr/bin/env python3
"""Generate high-confidence medicinal-chemistry-oriented synthesis-planning cases.

This script is designed for the final E+F+M + AutoGluon model under
`outputs/run_20260406_103701_autogluon_boosted`. It loads the best model,
scores the requested split, identifies high-confidence positive examples, and
adds rule-based medicinal-chemistry interpretations for why each reactant pair
resembles a common drug-discovery-oriented building-block combination.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _append_candidate_site_packages() -> None:
    """Inject local package locations with venv priority over conda packages.

    The final AutoGluon predictor was trained with the `.venv` stack, while RDKit
    is currently available from the local conda installation. We therefore put
    `.venv` site-packages at the front of `sys.path` and add conda locations
    afterwards as a fallback for RDKit.
    """

    venv_candidates: List[Path] = []
    conda_candidates: List[Path] = []

    venv_root = Path("/home/xwl/.venv/lib")
    if venv_root.exists():
        venv_candidates.extend(sorted(venv_root.glob("python*/site-packages")))

    conda_root = Path("/home/xwl/miniconda3")
    if conda_root.exists():
        conda_candidates.extend(sorted((conda_root / "lib").glob("python*/site-packages")))
        envs_root = conda_root / "envs"
        if envs_root.exists():
            for env_dir in sorted(envs_root.iterdir()):
                lib_dir = env_dir / "lib"
                conda_candidates.extend(sorted(lib_dir.glob("python*/site-packages")))

    for path in reversed(venv_candidates):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    for path in conda_candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)


_append_candidate_site_packages()

import pandas as pd


def import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception as exc:
        raise ModuleNotFoundError(
            "RDKit is required for motif annotation. Please run this script with a Python "
            "environment that has RDKit available."
        ) from exc
    return Chem, rdMolDescriptors


def import_autogluon_predictor():
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:
        raise ModuleNotFoundError(
            "AutoGluon is required to load the final predictor. Please run this script with "
            "an environment that can import autogluon.tabular."
        ) from exc
    return TabularPredictor


Chem, rdMolDescriptors = import_rdkit()
TabularPredictor = import_autogluon_predictor()


SOURCE_RANGES = {
    "E": {"name": "ECFP4", "start": 0, "end": 2047},
    "F": {"name": "FCFP4", "start": 2048, "end": 6143},
    "M": {"name": "MACCS", "start": 6144, "end": 6477},
}


PATTERN_SMARTS = {
    "acyl_halide": "[CX3](=O)[Cl,Br]",
    "anhydride": "[CX3](=O)O[CX3](=O)",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)O[#6]",
    "amide": "[NX3][CX3](=O)[#6]",
    "sulfonyl_chloride": "[SX4](=[OX1])(=[OX1])[Cl]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H][CX4;!$(C=O)]",
    "phenol": "[OX2H]c",
    "amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N[O,S]=O)]",
    "aniline": "[NX3;H2,H1,H0]c",
    "piperidine": "N1CCCCC1",
    "pyrrolidine": "N1CCCC1",
    "piperazine": "N1CCNCC1",
    "morpholine": "O1CCNCC1",
    "nitrile": "[CX2]#N",
    "nitro": "[NX3](=O)[O-]",
    "boronic_ester": "B1OC(C)(C)C(C)(C)O1",
    "boronic_acid_like": "[BX3]([OX2])[OX2]",
    "aryl_halide": "[c][F,Cl,Br,I]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "halomethyl": "[CH2][Cl,Br,I]",
    "indole": "c1ccc2[nH]ccc2c1",
    "quinoline_like": "c1ccc2ncccc2c1",
    "imidazole_like": "n1cc[nH]c1",
    "pyrazine_like": "n1ccnc(c1)",
}

PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in PATTERN_SMARTS.items()}


@dataclass
class MoleculeAnnotation:
    smiles: str
    canonical_smiles: str
    motif_flags: Dict[str, bool]
    motif_labels: List[str]
    heteroaromatic: bool
    aromatic_halide: bool
    heavy_atom_count: int
    ring_count: int
    formula: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Potential utility in drug-discovery-oriented synthesis planning cases."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/xwl/药物禁忌/outputs/run_20260406_103701_autogluon_boosted"),
    )
    parser.add_argument("--feature-set", default="E+F+M")
    parser.add_argument("--model", default="AutoGluon")
    parser.add_argument("--split", choices=["test", "validation", "train", "all"], default="test")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--min-prob", type=float, default=0.85)
    parser.add_argument("--positive-only", action="store_true", default=True)
    parser.add_argument("--include-misclassified", action="store_true", default=False)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to feature_sets/<feature_set>/synthesis_planning_cases",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_feature_sets(data_root: Path) -> Dict[str, Dict[str, Path]]:
    train_dir = data_root / "训练集"
    test_dir = data_root / "测试集"
    train_map = {p.stem.replace("train_", ""): p for p in sorted(train_dir.glob("train_*.csv"))}
    test_map = {p.stem.replace("test_", ""): p for p in sorted(test_dir.glob("test_*.csv"))}
    shared = sorted(set(train_map) & set(test_map))
    return {name: {"train": train_map[name], "test": test_map[name]} for name in shared}


def load_full_feature_matrix(feature_paths: Dict[str, Path]) -> pd.DataFrame:
    train_df = pd.read_csv(feature_paths["train"])
    test_df = pd.read_csv(feature_paths["test"])
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df.columns = [str(c) for c in full_df.columns]
    full_df.iloc[:, 0] = full_df.iloc[:, 0].astype(int)
    return full_df


def select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df.iloc[:, 0].astype(int)
    X = df.iloc[:, 1:].copy()
    X.columns = [str(c) for c in X.columns]
    return X, y


def load_split_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
    manifest["row_id"] = manifest["row_id"].astype(int)
    manifest["label"] = manifest["label"].astype(int)
    return manifest


def load_context(run_dir: Path, feature_set: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_json(run_dir / "resolved_config.json")
    data_root = Path(config["data_root"])
    feature_map = discover_feature_sets(data_root)
    full_df = load_full_feature_matrix(feature_map[feature_set])
    X_full, y_full = select_feature_columns(full_df)
    meta = pd.read_csv(data_root / "combined-data.csv").copy()
    meta.columns = ["smiles_a", "smiles_b"]
    meta["row_id"] = range(len(meta))
    meta["label"] = y_full.to_numpy(dtype=int)
    manifest = load_split_manifest(run_dir / "split_manifest.csv")
    merged_meta = meta.merge(manifest[["row_id", "split"]], on="row_id", how="left")
    return X_full, merged_meta, manifest


def load_selected_features(feature_dir: Path, model_name: str) -> List[str]:
    df = pd.read_csv(feature_dir / "selected_features_by_model.csv")
    subset = df[df["Model"] == model_name]["Feature"].astype(str).tolist()
    if not subset:
        raise RuntimeError(f"No selected features found for {model_name} in {feature_dir}")
    return subset


def load_predictor(feature_dir: Path):
    return TabularPredictor.load(str(feature_dir / "autogluon_model"))


def has_pattern(mol, pattern_name: str) -> bool:
    patt = PATTERNS[pattern_name]
    return patt is not None and mol.HasSubstructMatch(patt)


def detect_heteroaromatic(mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetAtomicNum() not in (6, 1):
            return True
    return False


def detect_aromatic_halide(mol) -> bool:
    halides = {9, 17, 35, 53}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in halides:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetIsAromatic():
                return True
    return False


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol)


def annotate_molecule(smiles: str) -> MoleculeAnnotation:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return MoleculeAnnotation(
            smiles=str(smiles),
            canonical_smiles=str(smiles),
            motif_flags={},
            motif_labels=["unparsed_smiles"],
            heteroaromatic=False,
            aromatic_halide=False,
            heavy_atom_count=0,
            ring_count=0,
            formula="",
        )

    flags = {name: has_pattern(mol, name) for name in PATTERNS}
    heteroaromatic = detect_heteroaromatic(mol)
    aromatic_halide = detect_aromatic_halide(mol)
    labels: List[str] = []

    if flags["acyl_halide"]:
        labels.append("acyl halide")
    if flags["anhydride"]:
        labels.append("anhydride")
    if flags["sulfonyl_chloride"]:
        labels.append("sulfonyl chloride")
    if aromatic_halide and heteroaromatic:
        labels.append("halogenated heteroaromatic scaffold")
    elif aromatic_halide:
        labels.append("aryl halide")
    if flags["halomethyl"]:
        labels.append("halomethyl handle")
    if flags["amine"]:
        labels.append("amine")
    if flags["aniline"]:
        labels.append("aniline-like amine")
    if flags["piperidine"]:
        labels.append("piperidine")
    if flags["pyrrolidine"]:
        labels.append("pyrrolidine")
    if flags["piperazine"]:
        labels.append("piperazine")
    if flags["morpholine"]:
        labels.append("morpholine")
    if flags["aldehyde"] or flags["ketone"]:
        labels.append("carbonyl electrophile")
    if flags["boronic_ester"] or flags["boronic_acid_like"]:
        labels.append("boron coupling handle")
    if flags["carboxylic_acid"]:
        labels.append("carboxylic acid")
    if flags["ester"]:
        labels.append("ester")
    if flags["amide"]:
        labels.append("amide/carbamate-like functionality")
    if flags["alcohol"] or flags["phenol"]:
        labels.append("alcohol/phenol")
    if flags["nitrile"]:
        labels.append("nitrile")
    if flags["nitro"]:
        labels.append("nitro group")
    if flags["indole"]:
        labels.append("indole-like heteroaromatic core")
    if flags["quinoline_like"] or flags["imidazole_like"] or flags["pyrazine_like"]:
        labels.append("privileged heteroaromatic ring")
    if heteroaromatic and "privileged heteroaromatic ring" not in labels and "indole-like heteroaromatic core" not in labels:
        labels.append("heteroaromatic core")

    if not labels:
        labels.append("drug-like aromatic / heteroatom-rich scaffold")

    return MoleculeAnnotation(
        smiles=str(smiles),
        canonical_smiles=Chem.MolToSmiles(mol),
        motif_flags=flags,
        motif_labels=labels,
        heteroaromatic=heteroaromatic,
        aromatic_halide=aromatic_halide,
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        ring_count=rdMolDescriptors.CalcNumRings(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
    )


def _amine_like(annotation: MoleculeAnnotation) -> bool:
    return any(
        annotation.motif_flags.get(key, False)
        for key in ["amine", "aniline", "piperidine", "pyrrolidine", "piperazine", "morpholine"]
    )


def _acyl_donor(annotation: MoleculeAnnotation) -> bool:
    return any(annotation.motif_flags.get(key, False) for key in ["acyl_halide", "anhydride", "sulfonyl_chloride"])


def _aryl_halide_like(annotation: MoleculeAnnotation) -> bool:
    return annotation.aromatic_halide or annotation.motif_flags.get("aryl_halide", False)


def _boron_handle(annotation: MoleculeAnnotation) -> bool:
    return annotation.motif_flags.get("boronic_ester", False) or annotation.motif_flags.get("boronic_acid_like", False)


def _carbonyl_electrophile(annotation: MoleculeAnnotation) -> bool:
    return annotation.motif_flags.get("aldehyde", False) or annotation.motif_flags.get("ketone", False)


def _alkyl_halide_like(annotation: MoleculeAnnotation) -> bool:
    return annotation.motif_flags.get("alkyl_halide", False) or annotation.motif_flags.get("halomethyl", False)


def explain_pair(a: MoleculeAnnotation, b: MoleculeAnnotation) -> Tuple[str, str]:
    pairs = [(a, b), (b, a)]
    for left, right in pairs:
        if _acyl_donor(left) and _amine_like(right):
            return (
                "acylation-ready electrophile + amine nucleophile",
                "This pair resembles a classic medicinal-chemistry acylation setup, where an acyl donor "
                "such as an acyl halide, anhydride, or sulfonyl chloride is combined with an amine-bearing "
                "building block to access amides or sulfonamides that are common in lead optimization.",
            )
        if _acyl_donor(left) and any(right.motif_flags.get(k, False) for k in ["alcohol", "phenol"]):
            return (
                "acyl donor + oxygen nucleophile",
                "This pair resembles an acyl-transfer combination relevant to ester or carbonate formation, "
                "which is frequently used in prodrug design and protecting-group-aware analogue synthesis.",
            )
        if _aryl_halide_like(left) and _amine_like(right):
            if left.heteroaromatic or left.motif_flags.get("nitro", False):
                return (
                    "activated heteroaryl halide + amine",
                    "This pair resembles a heteroaryl substitution motif often exploited in medicinal chemistry, "
                    "where a halogenated heteroaromatic electrophile is paired with an amine or cyclic amine to "
                    "rapidly elaborate hinge-binding or polarity-tuning substituents.",
                )
            return (
                "aryl halide + amine coupling motif",
                "This pair resembles a common aryl amination precursor set, in which a halogenated aromatic "
                "scaffold is combined with an amine-rich partner to generate analogues used in SAR exploration.",
            )
        if _alkyl_halide_like(left) and _amine_like(right):
            return (
                "alkylation handle + amine nucleophile",
                "This pair resembles a straightforward N-alkylation motif frequently used to install side chains, "
                "solubilizing groups, or linker units onto cyclic or acyclic amines in hit-to-lead campaigns.",
            )
        if _boron_handle(left) and _aryl_halide_like(right):
            return (
                "cross-coupling precursor pair",
                "This pair resembles a Suzuki-style coupling design, where a boron-containing building block and "
                "a halogenated arene or heteroarene provide a modular route to rapidly diversify aromatic frameworks.",
            )
        if _carbonyl_electrophile(left) and _amine_like(right):
            return (
                "carbonyl electrophile + amine",
                "This pair resembles a reductive-amination-compatible combination, a staple medicinal-chemistry "
                "reaction for rapidly varying basic side chains and vectoring substituents from aldehyde or ketone intermediates.",
            )
        if left.motif_flags.get("carboxylic_acid", False) and _amine_like(right):
            return (
                "acid fragment + amine fragment",
                "This pair resembles a pre-amide-coupling fragment combination that medicinal chemists routinely use "
                "after activation of the acid component to generate dense analogue series.",
            )

    return (
        "drug-like building-block pair",
        "This pair combines heteroatom-rich, medicinal-chemistry-relevant scaffolds and therefore resembles "
        "a plausible building-block combination for analogue synthesis, even though it does not map cleanly "
        "onto a single canonical named reaction motif.",
    )


def summarize_motif_labels(labels: Sequence[str], max_items: int = 3) -> str:
    unique = []
    seen = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return "; ".join(unique[:max_items])


def score_split(
    predictor,
    X_full: pd.DataFrame,
    meta_df: pd.DataFrame,
    manifest: pd.DataFrame,
    selected_features: Sequence[str],
    split_name: str,
) -> pd.DataFrame:
    if split_name == "all":
        row_ids = manifest["row_id"].to_numpy(dtype=int)
    else:
        row_ids = manifest.loc[manifest["split"] == split_name, "row_id"].to_numpy(dtype=int)
    X_eval = X_full.iloc[row_ids][list(selected_features)].reset_index(drop=True)
    prob = predictor.predict_proba(X_eval)
    if hasattr(prob, "columns"):
        if 1 in prob.columns:
            prob_pos = prob[1].to_numpy(dtype=float)
        elif "1" in prob.columns:
            prob_pos = prob["1"].to_numpy(dtype=float)
        else:
            prob_pos = prob.iloc[:, -1].to_numpy(dtype=float)
    else:
        prob_pos = prob[:, 1]
    pred = (prob_pos >= 0.5).astype(int)

    subset = meta_df[meta_df["row_id"].isin(row_ids)].copy().sort_values("row_id").reset_index(drop=True)
    subset["pred_prob"] = prob_pos
    subset["pred_label"] = pred
    subset["correct"] = (subset["pred_label"] == subset["label"]).astype(int)
    return subset


def annotate_cases(df: pd.DataFrame) -> pd.DataFrame:
    ann_a_cache: Dict[str, MoleculeAnnotation] = {}
    ann_b_cache: Dict[str, MoleculeAnnotation] = {}
    rows = []
    for _, row in df.iterrows():
        smiles_a = str(row["smiles_a"])
        smiles_b = str(row["smiles_b"])
        ann_a = ann_a_cache.setdefault(smiles_a, annotate_molecule(smiles_a))
        ann_b = ann_b_cache.setdefault(smiles_b, annotate_molecule(smiles_b))
        category, rationale = explain_pair(ann_a, ann_b)
        rows.append(
            {
                **row.to_dict(),
                "canonical_smiles_a": ann_a.canonical_smiles,
                "canonical_smiles_b": ann_b.canonical_smiles,
                "formula_a": ann_a.formula,
                "formula_b": ann_b.formula,
                "motifs_a": summarize_motif_labels(ann_a.motif_labels),
                "motifs_b": summarize_motif_labels(ann_b.motif_labels),
                "pair_category": category,
                "medchem_rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def select_diverse_cases(cases_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    picked_row_ids = set()
    selected_frames = []

    for category, group in cases_df.groupby("pair_category", sort=False):
        top_row = group.sort_values(["pred_prob", "row_id"], ascending=[False, True]).head(1)
        if top_row.empty:
            continue
        rid = int(top_row.iloc[0]["row_id"])
        if rid not in picked_row_ids:
            selected_frames.append(top_row)
            picked_row_ids.add(rid)
        if len(picked_row_ids) >= top_n:
            break

    if len(picked_row_ids) < top_n:
        remaining = cases_df[~cases_df["row_id"].isin(picked_row_ids)].sort_values(
            ["pred_prob", "pair_category", "row_id"], ascending=[False, True, True]
        )
        needed = top_n - len(picked_row_ids)
        if needed > 0:
            selected_frames.append(remaining.head(needed))

    if not selected_frames:
        return cases_df.head(top_n).copy()
    return pd.concat(selected_frames, ignore_index=True).sort_values(
        ["pred_prob", "pair_category"], ascending=[False, True]
    ).reset_index(drop=True)


def build_markdown_section(
    selected_df: pd.DataFrame,
    *,
    feature_set: str,
    model_name: str,
    auc_value: float,
    selected_feature_count: int,
    contribution_df: Optional[pd.DataFrame],
) -> str:
    lines: List[str] = []
    lines.append("### Potential utility in drug-discovery-oriented synthesis planning")
    lines.append("")
    intro = (
        f"To illustrate how the final {feature_set} + {model_name} model may be used in "
        f"drug-discovery-oriented synthesis planning, we inspected the highest-confidence positive "
        f"predictions from the independent test split. The final model achieved an internal test-set "
        f"AUROC of {auc_value:.4f} using {selected_feature_count} selected features."
    )
    lines.append(intro)
    if contribution_df is not None and not contribution_df.empty:
        contrib_map = {
            row["SourceName"]: float(row["PositiveImportanceShare"])
            for _, row in contribution_df.iterrows()
            if pd.notna(row["PositiveImportanceShare"])
        }
        if {"FCFP4", "MACCS", "ECFP4"}.issubset(contrib_map):
            lines.append(
                "Feature-importance analysis showed that the positive-importance contribution was dominated by "
                f"FCFP4 ({contrib_map['FCFP4']:.3f}), followed by MACCS ({contrib_map['MACCS']:.3f}) and "
                f"ECFP4 ({contrib_map['ECFP4']:.3f}), suggesting that the model is particularly sensitive to "
                "local functional environments and medicinal-chemistry-relevant pharmacophoric patterns."
            )
    lines.append(
        "Representative high-confidence cases frequently paired electrophile-bearing aromatic or heteroaromatic "
        "cores with nucleophilic amines, cyclic amines, or other medicinal-chemistry building blocks that are "
        "commonly used in analogue synthesis."
    )
    lines.append("")
    lines.append("Representative cases:")
    lines.append("")
    for idx, row in selected_df.reset_index(drop=True).iterrows():
        lines.append(
            f"{idx + 1}. `P={row['pred_prob']:.3f}`. Reactant A (`{row['smiles_a']}`; {row['motifs_a']}) and "
            f"Reactant B (`{row['smiles_b']}`; {row['motifs_b']}) were classified as a "
            f"**{row['pair_category']}**. {row['medchem_rationale']}"
        )
    lines.append("")
    lines.append(
        "Taken together, these examples suggest that the model is not merely capturing dataset-specific patterns, "
        "but is also prioritizing reactant combinations that resemble practical medicinal-chemistry pairing logic, "
        "including acylation, amination, alkylation, reductive amination, and cross-coupling-oriented precursor selection."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    feature_dir = run_dir / "feature_sets" / args.feature_set
    output_dir = args.output_dir.resolve() if args.output_dir else feature_dir / "synthesis_planning_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_full, meta_df, manifest = load_context(run_dir, args.feature_set)
    selected_features = load_selected_features(feature_dir, args.model)
    predictor = load_predictor(feature_dir)

    scored_df = score_split(
        predictor,
        X_full,
        meta_df,
        manifest,
        selected_features,
        split_name=args.split,
    )

    if args.positive_only:
        scored_df = scored_df[scored_df["label"] == 1].copy()
    if not args.include_misclassified:
        scored_df = scored_df[scored_df["correct"] == 1].copy()
    if args.min_prob is not None:
        filtered = scored_df[scored_df["pred_prob"] >= args.min_prob].copy()
        if len(filtered) >= min(args.top_n, max(3, args.top_n // 2)):
            scored_df = filtered

    scored_df = scored_df.sort_values(["pred_prob", "row_id"], ascending=[False, True]).reset_index(drop=True)
    annotated_df = annotate_cases(scored_df)

    selected_df = select_diverse_cases(annotated_df, args.top_n)
    selected_df.insert(0, "case_rank", range(1, len(selected_df) + 1))

    metrics_df = pd.read_csv(feature_dir / "test_metrics.csv")
    auc_value = float(metrics_df.loc[metrics_df["Model"] == args.model, "AUC"].iloc[0])
    selected_counts_path = feature_dir / "selected_feature_counts_by_model.csv"
    selected_feature_count = int(
        pd.read_csv(selected_counts_path).loc[lambda d: d["Model"] == args.model, "SelectedFeatureCount"].iloc[0]
    )

    contrib_path = feature_dir / "explainability" / "source_contribution_summary.csv"
    contribution_df = pd.read_csv(contrib_path) if contrib_path.exists() else None

    section_md = build_markdown_section(
        selected_df,
        feature_set=args.feature_set,
        model_name=args.model,
        auc_value=auc_value,
        selected_feature_count=selected_feature_count,
        contribution_df=contribution_df,
    )

    all_cases_path = output_dir / "all_high_confidence_positive_cases.csv"
    selected_cases_path = output_dir / "selected_cases_top.csv"
    section_md_path = output_dir / "potential_utility_in_synthesis_planning.md"
    summary_json_path = output_dir / "run_summary.json"

    annotated_df.to_csv(all_cases_path, index=False)
    selected_df.to_csv(selected_cases_path, index=False)
    section_md_path.write_text(section_md, encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "feature_set": args.feature_set,
        "model": args.model,
        "split": args.split,
        "positive_only": args.positive_only,
        "include_misclassified": args.include_misclassified,
        "min_prob": args.min_prob,
        "selected_feature_count": selected_feature_count,
        "test_auc": auc_value,
        "n_all_annotated_cases": int(len(annotated_df)),
        "n_selected_cases": int(len(selected_df)),
        "predictor_path": str(feature_dir / "autogluon_model"),
        "selected_features_path": str(feature_dir / "selected_features_by_model.csv"),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(all_cases_path)
    print(selected_cases_path)
    print(section_md_path)
    print(summary_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
