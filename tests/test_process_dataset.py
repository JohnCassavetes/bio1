import unittest
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.process_dataset import _parse_bindingdb_response, affinity_nm_to_pactivity, split_by_target


class ProcessDatasetTestCase(unittest.TestCase):
    def test_affinity_nm_to_pactivity(self):
        self.assertAlmostEqual(affinity_nm_to_pactivity(10.0), 8.0)
        self.assertAlmostEqual(affinity_nm_to_pactivity(1000.0), 6.0)

    def test_split_by_target_has_no_target_leakage(self):
        df = pd.DataFrame(
            {
                "smiles": ["A", "B", "C", "D", "E", "F"],
                "target_id": ["T1", "T1", "T2", "T3", "T4", "T5"],
                "target_label": ["T1", "T1", "T2", "T3", "T4", "T5"],
                "target_sequence": ["", "", "", "", "", ""],
                "affinity_type": ["KD"] * 6,
                "activity_label": ["pKd"] * 6,
                "affinity_nm": [10, 20, 30, 40, 50, 60],
                "p_activity": [8, 7.7, 7.5, 7.4, 7.3, 7.2],
                "measurement_count": [1] * 6,
                "source": ["davis"] * 6,
            }
        )

        train_df, val_df, test_df, assignments = split_by_target(
            df,
            train_frac=0.5,
            val_frac=0.2,
            random_seed=123,
        )

        train_targets = set(train_df["target_id"])
        val_targets = set(val_df["target_id"])
        test_targets = set(test_df["target_id"])

        self.assertTrue(train_targets.isdisjoint(val_targets))
        self.assertTrue(train_targets.isdisjoint(test_targets))
        self.assertTrue(val_targets.isdisjoint(test_targets))
        self.assertEqual(
            train_targets | val_targets | test_targets,
            set(assignments["train"]) | set(assignments["val"]) | set(assignments["test"]),
        )

    def test_parse_bindingdb_response_handles_rest_wrapper(self):
        response = {
            "getLindsByUniprotsResponse": {
                "affinities": [
                    {
                        "query": "EGFR",
                        "smile": "CCO",
                        "affinity_type": "Ki",
                        "affinity": "25",
                    }
                ]
            }
        }
        records = _parse_bindingdb_response(
            response,
            "P00533",
            {"target_name": "EGFR", "target_sequence": "ACDE"},
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_label"], "EGFR")
        self.assertEqual(records[0]["target_sequence"], "ACDE")
        self.assertEqual(records[0]["affinity_type"], "KI")


if __name__ == "__main__":
    unittest.main()
