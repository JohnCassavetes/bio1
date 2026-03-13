import unittest
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.predict_rank import prepare_inference_dataframe


class PredictRankInputTestCase(unittest.TestCase):
    def test_prepare_inference_dataframe_uses_model_defaults(self):
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "target_id": ["EGFR"],
            }
        )
        metadata = {
            "default_inference_affinity_type": "KD",
            "default_inference_source": "davis",
            "feature_metadata": {
                "affinity_types": ["IC50", "KD", "KI"],
                "source_categories": ["bindingdb", "davis"],
            },
        }

        prepared = prepare_inference_dataframe(df, metadata)

        self.assertEqual(prepared.loc[0, "affinity_type"], "KD")
        self.assertEqual(prepared.loc[0, "source"], "davis")
        self.assertEqual(prepared.loc[0, "target_label"], "EGFR")
        self.assertEqual(prepared.loc[0, "measurement_count"], 1)

    def test_prepare_inference_dataframe_rejects_unknown_source(self):
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "target_id": ["EGFR"],
                "affinity_type": ["kd"],
                "source": ["inference"],
            }
        )
        metadata = {
            "feature_metadata": {
                "affinity_types": ["IC50", "KD", "KI"],
                "source_categories": ["bindingdb", "davis"],
            },
        }

        with self.assertRaisesRegex(ValueError, "Unsupported source values"):
            prepare_inference_dataframe(df, metadata)


if __name__ == "__main__":
    unittest.main()
