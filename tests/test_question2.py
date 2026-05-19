import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from question.question2 import (
    Question2Result,
    build_tariff_group_schedule,
    solve_question2,
)


class Question2SolverTests(unittest.TestCase):
    def test_build_tariff_group_schedule_expands_three_groups_to_24_hours(self):
        loads, rates = build_tariff_group_schedule(
            valley_load=100.0,
            valley_rate=1200.0,
            flat_load=80.0,
            flat_rate=1100.0,
            peak_load=60.0,
            peak_rate=900.0,
        )

        self.assertEqual(loads.shape, (24,))
        self.assertEqual(rates.shape, (24,))
        self.assertTrue(np.all(loads[[0, 1, 2, 3, 4, 5, 22, 23]] == 100.0))
        self.assertTrue(np.all(rates[[6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21]] == 1100.0))
        self.assertTrue(np.all(loads[[9, 10, 18, 19]] == 60.0))

    def test_solve_question2_returns_feasible_dynamic_schedule(self):
        result = solve_question2()

        self.assertIsInstance(result, Question2Result)
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(result.loads.shape, (24,))
        self.assertEqual(result.rates.shape, (24,))
        self.assertEqual(result.prices.shape, (24,))
        self.assertGreaterEqual(result.total_work, 1.0 - 1e-6)
        self.assertLessEqual(result.weighted_error, 5.0 + 1e-6)
        self.assertLess(result.total_cost, result.baseline_static_cost)

    def test_solve_question2_matches_grouped_tariff_pattern(self):
        result = solve_question2()

        self.assertAlmostEqual(result.valley_load, 100.0, places=3)
        self.assertAlmostEqual(result.flat_load, 78.642081, places=3)
        self.assertAlmostEqual(result.peak_load, 60.0, places=3)
        self.assertGreater(result.valley_load, result.flat_load)
        self.assertGreater(result.flat_load, result.peak_load)
        self.assertTrue(np.allclose(result.rates, 1200.0, atol=1e-3))
        self.assertAlmostEqual(result.total_cost, 162.336219, places=2)
        self.assertAlmostEqual(result.baseline_static_cost, 190.368, places=3)
        self.assertAlmostEqual(result.weighted_error, 5.0, places=4)

    def test_question2_script_can_run_directly(self):
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "question" / "question2.py"

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("问题二优化结果", completed.stdout)
        self.assertIn("最小日电费", completed.stdout)


if __name__ == "__main__":
    unittest.main()

