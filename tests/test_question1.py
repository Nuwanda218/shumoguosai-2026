import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from question.question1 import (
    Question1Result,
    build_static_schedule,
    solve_question1,
)


class Question1SolverTests(unittest.TestCase):
    def test_build_static_schedule_expands_two_variables_to_24_hours(self):
        loads, rates = build_static_schedule(85.0, 1200.0)

        self.assertEqual(loads.shape, (24,))
        self.assertEqual(rates.shape, (24,))
        self.assertTrue(np.all(loads == 85.0))
        self.assertTrue(np.all(rates == 1200.0))

    def test_solve_question1_returns_feasible_baseline_solution(self):
        result = solve_question1()

        self.assertIsInstance(result, Question1Result)
        self.assertTrue(result.success, msg=result.message)
        self.assertGreaterEqual(result.gpu_load, 60.0)
        self.assertLessEqual(result.gpu_load, 100.0)
        self.assertGreaterEqual(result.transmission_rate, 800.0)
        self.assertLessEqual(result.transmission_rate, 1200.0)
        self.assertGreaterEqual(result.total_work, 1.0 - 1e-6)
        self.assertLessEqual(result.weighted_error, 5.0 + 1e-6)

    def test_solve_question1_matches_known_linear_model_optimum(self):
        result = solve_question1()

        self.assertAlmostEqual(result.gpu_load, 85.0, places=3)
        self.assertAlmostEqual(result.transmission_rate, 1200.0, places=3)
        self.assertAlmostEqual(result.total_energy, 158.64, places=2)
        self.assertAlmostEqual(result.weighted_error, 5.0, places=3)
        self.assertAlmostEqual(result.total_work, 1.275, places=3)
        self.assertAlmostEqual(result.system_power, 6.61, places=3)

    def test_question1_script_can_run_directly(self):
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "question" / "question1.py"

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("问题一优化结果", completed.stdout)
        self.assertIn("最优 GPU 负载：85.0000%", completed.stdout)


if __name__ == "__main__":
    unittest.main()
