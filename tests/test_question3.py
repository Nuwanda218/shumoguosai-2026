import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from models.parameters import DEFAULT_PARAMS
from question.question3 import (
    RESERVE_SOC_CANDIDATES,
    Question3Result,
    solve_question3,
    solve_question3_reserve_sweep,
)


class Question3SolverTests(unittest.TestCase):
    def test_solve_question3_returns_feasible_daily_cycle(self):
        result = solve_question3(reserve_soc_kwh=16.0)

        self.assertIsInstance(result, Question3Result)
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(result.loads.shape, (24,))
        self.assertEqual(result.rates.shape, (24,))
        self.assertEqual(result.cooling_powers.shape, (24,))
        self.assertEqual(result.charge_powers.shape, (24,))
        self.assertEqual(result.discharge_powers.shape, (24,))
        self.assertEqual(result.soc.shape, (25,))
        self.assertGreaterEqual(result.total_work, 1.0 - 1e-6)
        self.assertLessEqual(result.weighted_error, DEFAULT_PARAMS.error_limit_percent + 1e-5)
        self.assertGreaterEqual(np.min(result.soc), DEFAULT_PARAMS.battery_min_soc_kwh - 1e-6)
        self.assertLessEqual(np.max(result.soc), DEFAULT_PARAMS.battery_max_soc_kwh + 1e-6)
        self.assertAlmostEqual(result.soc[0], 16.0, places=5)
        self.assertAlmostEqual(result.soc[-1], 16.0, places=4)
        self.assertLessEqual(result.max_cooling_delta, DEFAULT_PARAMS.cooling_change_limit_kw + 1e-6)
        self.assertGreaterEqual(np.min(result.grid_powers), -1e-6)

    def test_reserve_sweep_solves_all_candidate_cycle_levels(self):
        results = solve_question3_reserve_sweep()

        self.assertEqual(tuple(result.reserve_soc_kwh for result in results), RESERVE_SOC_CANDIDATES)
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(all(result.total_work >= 1.0 - 1e-6 for result in results))
        self.assertTrue(all(result.weighted_error <= DEFAULT_PARAMS.error_limit_percent + 1e-5 for result in results))

        best = min(results, key=lambda item: item.total_cost)
        self.assertIn(best.reserve_soc_kwh, RESERVE_SOC_CANDIDATES)
        self.assertLessEqual(best.total_cost, max(result.total_cost for result in results))

    def test_question3_script_can_run_directly(self):
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "question" / "question3.py"

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("问题三优化结果", completed.stdout)
        self.assertIn("循环额度对比", completed.stdout)


if __name__ == "__main__":
    unittest.main()
