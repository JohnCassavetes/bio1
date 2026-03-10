import unittest

import numpy as np

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.metrics import top_fraction_enrichment


class MetricsTestCase(unittest.TestCase):
    def test_top_fraction_enrichment_is_above_random_for_good_ranking(self):
        y_true = np.array([8.0, 7.5, 5.0, 4.5, 4.0])
        y_pred = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        enrichment = top_fraction_enrichment(
            y_true,
            y_pred,
            active_threshold=6.0,
            top_fraction=0.4,
        )
        self.assertAlmostEqual(enrichment, 2.5)


if __name__ == "__main__":
    unittest.main()
