"""Checks for the frozen release and the portable F+M training implementation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fm_autogluon as fm


class RevisionTests(unittest.TestCase):
    def test_release_sampling_and_splits(self):
        self.assertEqual(len(fm.verify_release()), 9928)

    def test_provenance_coverage(self):
        records = pd.read_csv(fm.DATA / "reaction_records.csv.gz")
        samples = pd.read_csv(fm.DATA / "all_samples.csv")
        self.assertEqual(len(records), 50473)
        self.assertEqual(set(records.sample_id), set(samples.sample_id))
        self.assertTrue(records.product_smiles.notna().all())
        self.assertTrue(records.reaction_smiles.notna().all())
        self.assertFalse(records.source_file.str.startswith("/").any())

    def test_fingerprint_layout(self):
        a, b = fm.fingerprint("CCO"), fm.fingerprint("CC(=O)O")
        self.assertEqual(a.shape, (2215,))
        pair = fm.pair_vector(a, b)
        self.assertEqual(pair.shape, (4430,))
        np.testing.assert_array_equal(pair[:2048], a[:2048])
        np.testing.assert_array_equal(pair[2048:4096], b[:2048])
        np.testing.assert_array_equal(pair[4096:4263], a[2048:])
        np.testing.assert_array_equal(pair[4263:], b[2048:])
        self.assertTrue(np.isin(pair, [0, 1]).all())

    def test_mutual_information(self):
        rng = np.random.default_rng(42)
        for n in [10, 100, 1000]:
            a, b = rng.integers(0, 2, (2, n))
            observed = fm.binary_mi(np.dot(a, b), a.sum(), b.sum(), n)
            self.assertAlmostEqual(float(observed), mutual_info_score(a, b), places=12)
        self.assertAlmostEqual(float(fm.binary_mi(0, 0, 5, 10)), 0.0)

    def test_mid_and_ifs(self):
        relevance = np.array([0.9, 0.8, 0.7])
        redundancy = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
        np.testing.assert_array_equal(
            fm.mid_rank(relevance, redundancy, [0, 1, 2]), [0, 2, 1]
        )
        self.assertEqual(
            fm.ifs_counts(4242), [1] + list(range(400, 4243, 400)) + [4242]
        )
        frame = pd.DataFrame(
            {
                "k": [1, 400, 800, 1200, 4242],
                "validation_AUC": [0.9, 0.92, 0.92, 0.91, 0.89],
            }
        )
        self.assertEqual(fm.shortlist(frame), [400, 800, 1200, 4242])

    def test_published_predictions_and_selections(self):
        for fold, count in enumerate([4242, 4249, 4000, 4231, 4239], 1):
            folder = fm.REFERENCE / f"fold_{fold:02d}"
            pred = pd.read_csv(folder / "test_predictions.csv")
            data, parts = fm.parts_for(fold)
            expected = data.iloc[parts["test"]].set_index("sample_id").label
            self.assertEqual(set(pred.sample_id), set(expected.index))
            self.assertTrue(pred.sample_id.is_unique)
            np.testing.assert_array_equal(
                pred.label.to_numpy(), expected.loc[pred.sample_id].to_numpy()
            )
            metrics = fm.read(folder / "metrics.json")
            measured = fm.scores(
                pred.label.to_numpy(), pred.score.to_numpy(), pred.prediction.to_numpy()
            )
            for metric in fm.METRICS:
                self.assertAlmostEqual(measured[metric], metrics[metric], places=10)
            self.assertEqual(fm.read(folder / "selection.json")["selected_k"], count)
            ranking = pd.read_csv(folder / "ranking.csv")
            selected = pd.read_csv(folder / "selected_features.csv")
            self.assertEqual(
                selected.feature.tolist(), ranking.feature.head(count).tolist()
            )

    def test_profiles(self):
        cfg = fm.read(fm.CONFIG)
        for phase, seconds, bags, stacks in [
            ("screening", 120, 3, 0),
            ("refinement", 900, 5, 1),
            ("final", 1800, 5, 1),
        ]:
            options = cfg[phase]
            self.assertEqual(options["time_limit"], seconds)
            self.assertEqual(options["num_bag_folds"], bags)
            self.assertEqual(options["num_stack_levels"], stacks)
            self.assertFalse(options["dynamic_stacking"])
            self.assertFalse(options["refit_full"])


if __name__ == "__main__":
    unittest.main()
