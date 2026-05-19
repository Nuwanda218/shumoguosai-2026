import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from question.question2 import solve_question2
from visualization.question2_plots import (
    generate_all_question2_plots,
    generate_question2_hourly_data,
    generate_rate_reason_data,
)


class Question2VisualizationTests(unittest.TestCase):
    def test_generate_question2_hourly_data_has_expected_series(self):
        result = solve_question2()
        data = generate_question2_hourly_data(result)

        self.assertEqual(data["hours"].shape, (24,))
        self.assertEqual(data["loads"].shape, (24,))
        self.assertEqual(data["rates"].shape, (24,))
        self.assertEqual(data["instant_errors"].shape, (24,))
        self.assertAlmostEqual(float(data["weighted_error"]), 5.0, places=4)
        self.assertGreater(float(data["instant_errors"][9]), 5.0)
        self.assertLess(float(data["instant_errors"][0]), 5.0)

    def test_generate_rate_reason_data_explains_rate_maximum(self):
        data = generate_rate_reason_data(load_levels=(60.0, 78.642081, 100.0), num_rates=41)

        self.assertEqual(data["rates"].shape, (41,))
        self.assertAlmostEqual(float(data["transmission_power"][0]), 0.25, places=6)
        self.assertAlmostEqual(float(data["transmission_power"][-1]), 0.41, places=6)
        for errors in data["errors_by_load"].values():
            self.assertLess(float(errors[-1]), float(errors[0]))

    def test_generate_all_question2_plots_writes_five_nonempty_pngs(self):
        result = solve_question2()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            paths = generate_all_question2_plots(result=result, output_dir=output_dir)

            self.assertEqual(
                {path.name for path in paths},
                {
                    "tariff_schedule.png",
                    "power_cost_schedule.png",
                    "work_error_contribution.png",
                    "q1_q2_comparison.png",
                    "rate_max_reason.png",
                },
            )
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
                self.assertGreater(path.stat().st_size, 10_000, msg=str(path))

    def test_question2_plots_script_can_run_directly(self):
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "visualization" / "question2_plots.py"

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()

