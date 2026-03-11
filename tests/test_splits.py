import unittest
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.splits import generate_split_bundle


class SplitGenerationTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "smiles": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "target_id": ["T1", "T2", "T1", "T3", "T2", "T4", "T3", "T4"],
                "target_label": ["T1", "T2", "T1", "T3", "T2", "T4", "T3", "T4"],
                "target_sequence": ["AAAA", "BBBB", "AAAA", "CCCC", "BBBB", "DDDD", "CCCC", "DDDD"],
                "affinity_type": ["KD"] * 8,
                "activity_label": ["pKd"] * 8,
                "affinity_nm": [1.0] * 8,
                "p_activity": [8.0, 7.5, 7.2, 6.8, 6.4, 6.1, 5.9, 5.6],
                "measurement_count": [1] * 8,
                "source": ["davis"] * 8,
            }
        )

    def test_cold_target_split_has_disjoint_targets(self):
        bundle = generate_split_bundle(self.df, split_type="cold_target", random_seed=7)
        self.assertTrue(set(bundle.train_df["target_id"]).isdisjoint(bundle.val_df["target_id"]))
        self.assertTrue(set(bundle.train_df["target_id"]).isdisjoint(bundle.test_df["target_id"]))

    def test_cold_ligand_split_has_disjoint_ligands(self):
        bundle = generate_split_bundle(self.df, split_type="cold_ligand", random_seed=7)
        self.assertTrue(set(bundle.train_df["smiles"]).isdisjoint(bundle.val_df["smiles"]))
        self.assertTrue(set(bundle.train_df["smiles"]).isdisjoint(bundle.test_df["smiles"]))

    def test_both_new_split_can_discard_mixed_quadrants(self):
        bundle = generate_split_bundle(self.df, split_type="both_new", random_seed=7)
        self.assertGreaterEqual(bundle.manifest["discarded_rows"], 0)
        self.assertLessEqual(
            bundle.manifest["train_rows"] + bundle.manifest["val_rows"] + bundle.manifest["test_rows"],
            len(self.df),
        )

    def test_mutation_holdout_keeps_wildtype_train_and_variants_eval(self):
        df = pd.DataFrame(
            {
                "smiles": ["A", "A", "B", "B"],
                "target_id": ["EGFR", "EGFR(T790M)", "ABL1", "ABL1(T315I)"],
                "target_label": ["EGFR", "EGFR(T790M)", "ABL1", "ABL1(T315I)"],
                "target_sequence": ["AAAA", "AAAA", "BBBB", "BBBB"],
                "affinity_type": ["KD"] * 4,
                "activity_label": ["pKd"] * 4,
                "affinity_nm": [1.0] * 4,
                "p_activity": [8.0, 7.0, 8.0, 6.0],
                "measurement_count": [1] * 4,
                "source": ["davis"] * 4,
            }
        )
        bundle = generate_split_bundle(df, split_type="mutation_holdout", random_seed=7)
        self.assertTrue((bundle.train_df["target_id"].isin(["EGFR", "ABL1"])).all())
        self.assertTrue(
            set(bundle.val_df["target_id"]).union(set(bundle.test_df["target_id"])) <= {"EGFR(T790M)", "ABL1(T315I)"}
        )

    def test_sequence_identity_split_groups_similar_targets(self):
        df = pd.DataFrame(
            {
                "smiles": ["A", "A", "B", "B", "C", "C"],
                "target_id": ["T1", "T1_mut", "T2", "T2_mut", "T3", "T4"],
                "target_label": ["T1", "T1_mut", "T2", "T2_mut", "T3", "T4"],
                "target_sequence": [
                    "AAAAAA",
                    "AAAAAT",
                    "CCCCCC",
                    "CCCCCG",
                    "GGGGGG",
                    "TTTTTT",
                ],
                "affinity_type": ["KD"] * 6,
                "activity_label": ["pKd"] * 6,
                "affinity_nm": [1.0] * 6,
                "p_activity": [8.0, 7.8, 7.5, 7.2, 6.4, 6.1],
                "measurement_count": [1] * 6,
                "source": ["davis"] * 6,
            }
        )
        bundle = generate_split_bundle(
            df,
            split_type="sequence_identity",
            random_seed=7,
            sequence_identity_threshold=0.5,
            sequence_kmer_size=2,
        )
        self.assertIn("sequence_identity_cluster_count", bundle.manifest)
        self.assertEqual(bundle.manifest["sequence_identity_metric"], "global_alignment_percent_identity")
        self.assertIn("target_sequence_identity", bundle.artifacts)
        self.assertIn("target_sequence_identity_clusters", bundle.artifacts)
        combined_rows = len(bundle.train_df) + len(bundle.val_df) + len(bundle.test_df)
        self.assertEqual(combined_rows, len(df))


if __name__ == "__main__":
    unittest.main()
