import unittest
from pathlib import Path
import sys
import os

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking import literature_models
from kinase_ligand_ranking.literature_models import (
    DeepDTAExactConfig,
    _smile_to_graph,
    run_deepdta_exact,
)


class LiteratureModelTestCase(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCN", "CCC", "CCCl"],
                "target_id": ["T1", "T1", "T2", "T2"],
                "target_label": ["T1", "T1", "T2", "T2"],
                "target_sequence": ["ACDEFG", "ACDEFG", "LMNPQR", "LMNPQR"],
                "affinity_type": ["KD"] * 4,
                "activity_label": ["pKd"] * 4,
                "affinity_nm": [1.0, 2.0, 3.0, 4.0],
                "p_activity": [8.0, 7.7, 7.5, 7.3],
                "measurement_count": [1] * 4,
                "source": ["davis"] * 4,
            }
        )
        self.val_df = self.train_df.iloc[:2].copy()
        self.test_df = self.train_df.iloc[2:].copy()

    def test_smile_to_graph_produces_graphdta_feature_size(self):
        _, features, edge_index = _smile_to_graph("CCO")
        self.assertEqual(features.shape[1], 78)
        self.assertEqual(edge_index.shape[1], 2)

    @unittest.skipIf(
        literature_models.torch is None or os.environ.get("RUN_TORCH_TESTS") != "1",
        "PyTorch smoke tests are opt-in",
    )
    def test_run_deepdta_exact_smoke(self):
        predictions, metadata = run_deepdta_exact(
            self.train_df,
            self.val_df,
            self.test_df,
            config=DeepDTAExactConfig(
                max_seq_len=8,
                max_smi_len=8,
                num_windows_options=(8,),
                smi_window_lengths=(2,),
                seq_window_lengths=(2,),
                batch_size=2,
                max_epochs=1,
                patience=1,
            ),
            device="cpu",
        )
        self.assertEqual(len(predictions), len(self.test_df))
        self.assertEqual(metadata["selected_hyperparameters"]["num_windows"], 8)


if __name__ == "__main__":
    unittest.main()
