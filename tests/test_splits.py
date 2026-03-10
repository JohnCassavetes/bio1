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


if __name__ == "__main__":
    unittest.main()
