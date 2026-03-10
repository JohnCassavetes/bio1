import unittest
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_budgeted_policies import evaluate_policy, tune_risk_lambda


class BudgetPolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "target_id": ["T1", "T1", "T1", "T2", "T2", "T2"],
                "p_activity": [7.0, 6.2, 5.0, 8.0, 5.8, 5.2],
                "predicted_p_activity": [6.8, 6.4, 6.1, 7.4, 6.3, 6.2],
                "prediction_std": [0.1, 0.8, 1.0, 0.2, 0.5, 1.5],
                "prob_active": [0.99, 0.60, 0.40, 0.99, 0.45, 0.20],
            }
        )

    def test_evaluate_policy_returns_metrics(self):
        metrics = evaluate_policy(
            self.df,
            budget=1,
            active_threshold=6.0,
            policy="mean",
        )
        self.assertIn("hit_rate_at_budget", metrics)
        self.assertIn("mean_p_activity_at_budget", metrics)
        self.assertIn("regret_at_budget", metrics)

    def test_tune_risk_lambda_prefers_some_candidate(self):
        best_lambda, trials = tune_risk_lambda(
            self.df,
            budget=1,
            active_threshold=6.0,
            candidate_lambdas=[0.0, 0.5, 1.0],
        )
        self.assertIn(best_lambda, [0.0, 0.5, 1.0])
        self.assertEqual(len(trials), 3)


if __name__ == "__main__":
    unittest.main()
