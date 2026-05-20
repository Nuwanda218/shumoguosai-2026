import unittest

import numpy as np

from models.parameters import DEFAULT_PARAMS
from question.question4 import Question4Result, cyclic_abs_deltas, solve_question4


class Question4SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = solve_question4()

    def test_cyclic_abs_deltas_includes_daily_boundary(self):
        values = np.array([1.0, 4.0, 2.0])

        deltas = cyclic_abs_deltas(values)

        self.assertTrue(np.allclose(deltas, np.array([3.0, 2.0, 1.0])))

    def test_solve_question4_returns_feasible_smoothed_cycle(self):
        result = self.result

        self.assertIsInstance(result, Question4Result)
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(result.loads.shape, (24,))
        self.assertEqual(result.rates.shape, (24,))
        self.assertEqual(result.soc.shape, (25,))
        self.assertGreaterEqual(result.total_work, 1.0 - 1e-6)
        self.assertLessEqual(result.weighted_error, DEFAULT_PARAMS.error_limit_percent + 1e-5)
        self.assertAlmostEqual(result.reserve_soc_kwh, 24.0)
        self.assertLessEqual(result.max_load_delta, DEFAULT_PARAMS.gpu_change_limit_percent + 1e-6)
        self.assertLessEqual(result.max_rate_delta, DEFAULT_PARAMS.rate_change_limit_mbps + 1e-6)
        self.assertLessEqual(result.max_cooling_delta, DEFAULT_PARAMS.cooling_change_limit_kw + 1e-6)

    def test_question4_identifies_non_binding_transmission_rate_constraint(self):
        result = self.result

        self.assertAlmostEqual(result.max_rate_delta, 0.0, places=6)
        self.assertTrue(np.allclose(result.rates, DEFAULT_PARAMS.rate_max_mbps))


if __name__ == "__main__":
    unittest.main()
