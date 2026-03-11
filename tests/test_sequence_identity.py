import unittest
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.sequence_identity import (
    SequenceIdentityConfig,
    cluster_targets_by_identity,
    compute_identity_table,
    pairwise_sequence_identity,
)


class SequenceIdentityTestCase(unittest.TestCase):
    def test_pairwise_identity_is_one_for_identical_sequences(self):
        self.assertAlmostEqual(
            pairwise_sequence_identity("ACDEFG", "ACDEFG", gap_open=-2.0, gap_extend=-0.5),
            1.0,
        )

    def test_identity_table_and_clusters(self):
        df = pd.DataFrame(
            {
                "target_id": ["T1", "T2", "T3"],
                "target_sequence": ["AAAAAA", "AAAAAT", "CCCCCC"],
            }
        )
        identity_df = compute_identity_table(df, config=SequenceIdentityConfig(threshold=0.5))
        cluster_df = cluster_targets_by_identity(identity_df, threshold=0.5)
        cluster_map = dict(zip(cluster_df["target_id"], cluster_df["sequence_identity_cluster"]))
        self.assertEqual(cluster_map["T1"], cluster_map["T2"])
        self.assertNotEqual(cluster_map["T1"], cluster_map["T3"])


if __name__ == "__main__":
    unittest.main()
