"""Portable FCFP4 + MACCS / AutoGluon revision workflow (no workstation paths)."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from scipy.special import xlogy
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/revision_20261007"
REFERENCE = ROOT / "reference/fm"
CONFIG = ROOT / "configs/fm_revision.json"
METRICS = [
    "AUC",
    "AP",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "MCC",
    "Brier",
]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(temp, p)


def csv(frame, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    frame.to_csv(temp, index=False, float_format="%.17g")
    os.replace(temp, p)


def verify_release():
    manifest = read(ROOT / "configs/fm_release_manifest.json")
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Changed release file: {name}")
    samples = pd.read_csv(DATA / "all_samples.csv")
    balanced = pd.read_csv(DATA / "balanced_samples.csv")
    reserve = pd.read_csv(DATA / "reserved_negative_samples.csv")
    assert samples.label.value_counts().to_dict() == {0: 14492, 1: 4964}
    assert (
        samples.sample_id.is_unique and len(balanced) == 9928 and len(reserve) == 9528
    )
    assert (
        balanced.label.value_counts().to_dict() == {0: 4964, 1: 4964}
        and reserve.label.eq(0).all()
    )
    assert set(balanced.sample_id).isdisjoint(reserve.sample_id)
    assert set(balanced.sample_id) | set(reserve.sample_id) == set(samples.sample_id)
    assert len(
        {
            tuple(sorted((a, b)))
            for a, b in zip(samples.reactant_1_smiles, samples.reactant_2_smiles)
        }
    ) == len(samples)
    neg = samples[samples.label == 0]
    chosen = np.random.default_rng(42).choice(neg.index, 4964, replace=False)
    ids = sorted(list(samples[samples.label == 1].index) + chosen.tolist())
    pd.testing.assert_frame_equal(balanced, samples.loc[ids].reset_index(drop=True))
    expected = pd.read_csv(DATA / "split_manifest.csv")
    for fold, (development, test) in enumerate(
        KFold(5, shuffle=True, random_state=42).split(balanced), 1
    ):
        train, val = train_test_split(
            development, test_size=0.25, random_state=42 + fold
        )
        for role, indices in [("train", train), ("validation", val), ("test", test)]:
            actual = expected[(expected.fold == fold) & (expected.role == role)]
            assert set(actual.sample_id) == set(balanced.iloc[indices].sample_id)
            assert balanced.iloc[indices].label.nunique() == 2
    assert expected[expected.role == "test"].sample_id.value_counts().eq(1).all()
    print("Verified sampling, five-fold manifest and all release hashes.", flush=True)
    return balanced


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    arrays = []
    for fp in [generator.GetFingerprint(mol), MACCSkeys.GenMACCSKeys(mol)]:
        a = np.zeros(fp.GetNumBits(), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, a)
        arrays.append(a)
    return np.concatenate(arrays)


def pair_vector(a, b):
    return np.concatenate([a[:2048], b[:2048], a[2048:], b[2048:]])


def prepare(out, workers):
    frame = verify_release()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT / "configs/fm_release_manifest.json")
    if (out / "prepared.json").exists():
        previous = read(out / "prepared.json")
        assert previous["release_sha256"] == sha(
            ROOT / "configs/fm_release_manifest.json"
        )
        assert previous["matrix_file_sha256"] == sha(out / "fingerprints.npy")
        return
    molecules = sorted(set(frame.reactant_1_smiles) | set(frame.reactant_2_smiles))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(fingerprint, molecules, chunksize=64))
    lookup = dict(zip(molecules, values))
    x = np.stack(
        [
            pair_vector(lookup[a], lookup[b])
            for a, b in zip(frame.reactant_1_smiles, frame.reactant_2_smiles)
        ]
    )
    assert x.shape == (9928, 4430)
    assert (
        hashlib.sha256(x.tobytes(order="C")).hexdigest()
        == manifest["fm_matrix_uint8_sha256"]
    )
    np.save(out / "fingerprints.npy", x)
    write_json(
        out / "prepared.json",
        dict(
            release_sha256=sha(ROOT / "configs/fm_release_manifest.json"),
            matrix_file_sha256=sha(out / "fingerprints.npy"),
            shape=list(x.shape),
        ),
    )
    print("F+M matrix exactly matches original training inputs.", flush=True)


def parts_for(fold):
    data = pd.read_csv(DATA / "balanced_samples.csv")
    splits = pd.read_csv(DATA / "split_manifest.csv")
    splits = splits[splits.fold == fold]
    lookup = pd.Series(np.arange(len(data)), index=data.sample_id)
    parts = {
        r: lookup.loc[splits[splits.role == r].sample_id].to_numpy()
        for r in ["train", "validation", "test"]
    }
    parts["development"] = np.sort(np.r_[parts["train"], parts["validation"]])
    return data, parts


def binary_mi(joint, first, second, n):
    cells = [joint, first - joint, second - joint, n - first - second + joint]
    if any(np.any(c < -1e-8) for c in cells):
        raise ValueError("Invalid binary counts")
    total = sum(xlogy(np.maximum(c, 0), np.maximum(c, 0)) for c in cells)
    total -= (
        xlogy(first, first)
        + xlogy(n - first, n - first)
        + xlogy(second, second)
        + xlogy(n - second, n - second)
    )
    return np.maximum((total + n * np.log(n)) / n, 0.0)


def mid_rank(relevance, redundancy, eligible):
    eligible = np.asarray(eligible, dtype=int)
    rel = relevance[eligible]
    available = np.ones(len(eligible), dtype=bool)
    sums = np.zeros(len(eligible))
    selected = []
    for i in range(len(eligible)):
        score = rel.copy() if i == 0 else rel - sums / i
        score[~available] = -np.inf
        j = int(np.argmax(score))
        selected.append(int(eligible[j]))
        available[j] = False
        sums += redundancy[eligible, eligible[j]]
    return np.asarray(selected)


def rank_fold(out, fold, threads):
    dest = out / f"fold_{fold:02d}"
    dest.mkdir(exist_ok=True)
    reference = pd.read_csv(REFERENCE / f"fold_{fold:02d}" / "ranking.csv")
    if (dest / "ranking.csv").exists():
        cached = pd.read_csv(dest / "ranking.csv")
        assert cached.feature.tolist() == reference.feature.tolist()
        assert cached.column.tolist() == reference.column.tolist()
        return cached
    data, parts = parts_for(fold)
    x = np.load(out / "fingerprints.npy", mmap_mode="r")
    a = np.asarray(x[parts["train"]], dtype=np.float64)
    y = data.label.to_numpy()[parts["train"]].astype(float)
    counts = a.sum(0)
    n = len(a)
    with threadpool_limits(limits=threads):
        relevance = binary_mi(a.T @ y, counts, y.sum(), n)
        redundancy = np.empty((a.shape[1], a.shape[1]), dtype=np.float64)
        for start in range(0, a.shape[1], 128):
            end = min(start + 128, a.shape[1])
            redundancy[start:end] = binary_mi(
                a[:, start:end].T @ a, counts[start:end, None], counts[None, :], n
            )
        order = mid_rank(
            relevance, redundancy, np.flatnonzero((counts > 0) & (counts < n))
        )
    names = pd.read_csv(REFERENCE / "feature_names.csv")
    ranked = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "column": order,
            "feature": names.iloc[order].feature.to_numpy(),
            "MI_relevance": relevance[order],
        }
    )
    assert (
        ranked.feature.tolist() == reference.feature.tolist()
    ), "Ranking differs from published input order"
    csv(ranked, dest / "ranking.csv")
    csv(data.iloc[parts["train"]][["sample_id"]], dest / "ranking_training_ids.csv")
    print(
        f"Fold {fold}: {len(ranked)} MI-MID features, exact reference ranking verified.",
        flush=True,
    )
    return ranked


def ifs_counts(n):
    step = 20 if n <= 334 else 200 if n <= 4096 else 400
    return sorted(set([1, n] + list(range(step, n + 1, step))))


def shortlist(frame):
    assert frame.k.is_unique and np.isfinite(frame.validation_AUC).all()
    return sorted(
        set(
            frame.sort_values(["validation_AUC", "k"], ascending=[False, True])
            .k.head(3)
            .astype(int)
        )
        | {int(frame.k.max())}
    )


def fit_model(path, x, y, options, threads):
    from autogluon.tabular import TabularPredictor

    actual = dict(
        options, num_cpus=threads, ag_args_fit={"num_cpus": threads, "num_gpus": 0}
    )
    if (path / "fit_complete.json").exists():
        assert read(path / "fit_complete.json")["options"] == actual
        return TabularPredictor.load(str(path))
    if path.exists():
        raise RuntimeError(
            f"Incomplete fit: inspect and archive {path} before retrying"
        )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])]).assign(
        target=y
    )
    model = TabularPredictor(
        label="target",
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(path),
        verbosity=2,
    )
    model.fit(frame, **actual)
    csv(model.leaderboard(silent=True), path / "leaderboard_internal.csv")
    write_json(
        path / "fit_complete.json",
        dict(options=actual, best_model=model.model_best, n_train=len(y)),
    )
    return model


def probability(model, x):
    return model.predict_proba(
        pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    )[1].to_numpy()


def scores(y, p, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        AUC=float(roc_auc_score(y, p)),
        AP=float(average_precision_score(y, p)),
        Accuracy=float(np.mean(y == pred)),
        Precision=float(precision_score(y, pred, zero_division=0)),
        Sensitivity=float(tp / (tp + fn)),
        Specificity=float(tn / (tn + fp)),
        F1=float(f1_score(y, pred)),
        MCC=float(matthews_corrcoef(y, pred)),
        Brier=float(brier_score_loss(y, p)),
        TN=int(tn),
        FP=int(fp),
        FN=int(fn),
        TP=int(tp),
    )


def train_fold(out, fold, threads, selection):
    cfg = read(CONFIG)
    dest = out / f"fold_{fold:02d}"
    dest.mkdir(exist_ok=True)
    signature = dict(
        config_sha256=sha(CONFIG),
        selection=selection,
        threads=threads,
        release_sha256=sha(ROOT / "configs/fm_release_manifest.json"),
        code_sha256=sha(__file__),
    )
    if (dest / "run_config.json").exists():
        assert (
            read(dest / "run_config.json") == signature
        ), "Use a new output directory for a changed run"
    else:
        write_json(dest / "run_config.json", signature)
    if (dest / "metrics.json").exists():
        return
    data, parts = parts_for(fold)
    y = data.label.to_numpy()
    x = np.load(out / "fingerprints.npy", mmap_mode="r")
    ranking = rank_fold(out, fold, threads)
    order = ranking.column.to_numpy()
    with threadpool_limits(limits=threads):
        if selection == "rerun":
            for phase in ["screening", "refinement"]:
                file = dest / f"{phase}_ifs.csv"
                rows = pd.read_csv(file).to_dict("records") if file.exists() else []
                counts = (
                    ifs_counts(len(order))
                    if phase == "screening"
                    else shortlist(pd.read_csv(dest / "screening_ifs.csv"))
                )
                for k in counts:
                    if k in {int(r["k"]) for r in rows}:
                        continue
                    model = fit_model(
                        dest / phase / f"k{k}",
                        np.asarray(x[parts["train"]][:, order[:k]], dtype=np.float32),
                        y[parts["train"]],
                        cfg[phase],
                        threads,
                    )
                    p = probability(
                        model,
                        np.asarray(
                            x[parts["validation"]][:, order[:k]], dtype=np.float32
                        ),
                    )
                    csv(
                        data.iloc[parts["validation"]][["sample_id", "label"]].assign(
                            probability=p
                        ),
                        dest / phase / f"k{k}" / "validation_predictions.csv",
                    )
                    rows.append(
                        dict(
                            k=k,
                            validation_AUC=float(
                                roc_auc_score(y[parts["validation"]], p)
                            ),
                        )
                    )
                    csv(pd.DataFrame(rows), file)
                    del model
                    gc.collect()
                    print(
                        f'Fold {fold} {phase}: k={k}, validation AUROC={rows[-1]["validation_AUC"]:.6f}',
                        flush=True,
                    )
            best = (
                pd.read_csv(dest / "refinement_ifs.csv")
                .sort_values(["validation_AUC", "k"], ascending=[False, True])
                .iloc[0]
            )
            selected = dict(
                selected_k=int(best.k),
                validation_AUC=float(best.validation_AUC),
                selection_basis="rerun validation selection",
            )
        else:
            selected = dict(read(REFERENCE / f"fold_{fold:02d}" / "selection.json"))
            selected["selection_basis"] = (
                "frozen published validation selection; not selected using new test results"
            )
        k = selected["selected_k"]
        write_json(dest / "selection.json", selected)
        csv(ranking.iloc[:k], dest / "selected_features.csv")
        model = fit_model(
            dest / "final_model",
            np.asarray(x[parts["development"]][:, order[:k]], dtype=np.float32),
            y[parts["development"]],
            cfg["final"],
            threads,
        )
        xt = np.asarray(x[parts["test"]][:, order[:k]], dtype=np.float32)
        p = probability(model, xt)
        pred = model.predict(
            pd.DataFrame(xt, columns=[f"f{i}" for i in range(k)])
        ).to_numpy()
        csv(
            data.iloc[parts["test"]][["sample_id", "label"]].assign(
                probability=p, score=p, prediction=pred
            ),
            dest / "test_predictions.csv",
        )
        fpr, tpr, thresholds = roc_curve(y[parts["test"]], p)
        csv(
            pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thresholds)),
            dest / "roc_data.csv",
        )
        write_json(
            dest / "training_provenance.json",
            dict(
                selected_k=k,
                decision_threshold=float(model.decision_threshold),
                roles={
                    r: data.iloc[idx].sample_id.tolist() for r, idx in parts.items()
                },
            ),
        )
        write_json(
            dest / "metrics.json",
            dict(SelectedFeatures=k, **scores(y[parts["test"]], p, pred)),
        )
        print(f"Fold {fold} complete.", flush=True)


def summarize(out, reference=False):
    source = REFERENCE if reference else out
    rows = []
    for fold in range(1, 6):
        p = source / f"fold_{fold:02d}" / "metrics.json"
        if p.exists():
            rows.append(dict(Fold=fold, **read(p)))
    if len(rows) != 5:
        raise ValueError(
            f"Only {len(rows)}/5 folds complete; no five-fold summary generated"
        )
    data = pd.DataFrame(rows)
    result = pd.DataFrame(
        [
            dict(
                Metric=m,
                Mean=data[m].mean(),
                SD=data[m].std(ddof=1),
                Display=f"{data[m].mean():.3f} \u00b1 {data[m].std(ddof=1):.3f}",
            )
            for m in METRICS
        ]
    )
    csv(data, out / ("reference_fold_metrics.csv" if reference else "fold_metrics.csv"))
    csv(result, out / ("reference_summary.csv" if reference else "summary.csv"))
    print(
        ("ARCHIVED PAPER RESULTS\n" if reference else "NEW TRAINING RESULTS\n")
        + result.to_string(index=False)
    )


def smoke(out, threads):
    import tempfile

    prepare(out, 2)
    data, parts = parts_for(1)
    x = np.load(out / "fingerprints.npy", mmap_mode="r")
    ranked = pd.read_csv(REFERENCE / "fold_01/ranking.csv").column.to_numpy()[:64]
    options = dict(read(CONFIG)["refinement"])
    options.update(
        time_limit=60,
        num_bag_folds=2,
        num_stack_levels=1,
        presets="medium_quality",
        hyperparameters={"RF": {"n_estimators": 20}, "XT": {"n_estimators": 20}},
    )
    with tempfile.TemporaryDirectory(prefix="smoke_", dir=out) as tmp:
        train = parts["train"][:200]
        test = parts["test"][:80]
        model = fit_model(
            Path(tmp) / "model",
            x[train][:, ranked].astype(np.float32),
            data.label.to_numpy()[train],
            options,
            threads,
        )
        p = probability(model, x[test][:, ranked].astype(np.float32))
        assert len(p) == 80 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert model.model_best and (Path(tmp) / "model/fit_complete.json").exists()
        reloaded = fit_model(Path(tmp) / "model", None, None, options, threads)
        np.testing.assert_array_equal(
            p, probability(reloaded, x[test][:, ranked].astype(np.float32))
        )
    write_json(
        out / "smoke_test.json",
        dict(
            status="passed",
            predicted_rows=80,
            note="Small integration test only; not manuscript training or evaluation.",
        ),
    )
    print("AutoGluon fit/predict integration smoke test passed.", flush=True)


def worker(task):
    out, fold, threads, selection = task
    with (out / f"fold_{fold:02d}.log").open("a") as log:
        p = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--out",
                str(out),
                "--fold",
                str(fold),
                "--threads",
                str(threads),
                "--selection",
                selection,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if p.returncode:
        raise RuntimeError(f"Fold {fold} failed; inspect {out}/fold_{fold:02d}.log")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=[
            "verify",
            "prepare",
            "rank",
            "train",
            "summarize",
            "reference-summary",
            "smoke-test",
        ],
    )
    p.add_argument("--out", type=Path, default=ROOT / "outputs/fm_revision")
    p.add_argument("--fold", type=int, choices=range(1, 6))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--selection", choices=["rerun", "paper"], default="rerun")
    a = p.parse_args()
    a.out = a.out.resolve()
    if min(a.workers, a.threads) < 1:
        p.error("Resources must be positive")
    if a.command == "verify":
        verify_release()
        return
    if a.command == "reference-summary":
        summarize(a.out, True)
        return
    if a.command == "summarize":
        summarize(a.out)
        return
    if a.command == "smoke-test":
        smoke(a.out, a.threads)
        return
    prepare(a.out, a.workers)
    if a.command == "prepare":
        return
    folds = [a.fold] if a.fold else list(range(1, 6))
    if a.command == "rank":
        for f in folds:
            rank_fold(a.out, f, a.threads)
    elif a.fold:
        train_fold(a.out, a.fold, a.threads, a.selection)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(worker, [(a.out, f, a.threads, a.selection) for f in folds]))
        summarize(a.out)


if __name__ == "__main__":
    main()
