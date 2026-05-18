import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from question.question1 import solve_question1
from visualization.question1_plots import (
    generate_all_question1_plots,
    generate_question1_grid,
    generate_rate_sensitivity_curves,
)
from visualization.style import CHINESE_FONT_CANDIDATES, QUESTION1_COLORS


class Question1VisualizationTests(unittest.TestCase):
    def test_style_exposes_chinese_font_and_shared_colors(self):
        self.assertIn("Microsoft YaHei", CHINESE_FONT_CANDIDATES)
        self.assertIn("optimal", QUESTION1_COLORS)
        self.assertIn("error_boundary", QUESTION1_COLORS)
        self.assertIn("work_boundary", QUESTION1_COLORS)

    def test_generate_question1_grid_has_expected_shape(self):
        grid = generate_question1_grid(num_loads=21, num_rates=17)

        self.assertEqual(grid["loads"].shape, (17, 21))
        self.assertEqual(grid["rates"].shape, (17, 21))
        self.assertEqual(grid["errors"].shape, (17, 21))
        self.assertEqual(grid["powers"].shape, (17, 21))
        self.assertEqual(grid["feasible"].shape, (17, 21))

    def test_generate_question1_grid_uses_expected_error_formula(self):
        grid = generate_question1_grid(num_loads=41, num_rates=41)
        loads = grid["load_values"]
        rates = grid["rate_values"]

        load_index = int((loads == 85.0).nonzero()[0][0])
        rate_index = int((rates == 1200.0).nonzero()[0][0])

        self.assertAlmostEqual(grid["errors"][rate_index, load_index], 5.0)
        self.assertAlmostEqual(grid["powers"][rate_index, load_index], 6.61)
        self.assertTrue(bool(grid["feasible"][rate_index, load_index]))

        min_load_index = int((loads == 60.0).nonzero()[0][0])
        min_rate_index = int((rates == 800.0).nonzero()[0][0])
        self.assertAlmostEqual(grid["errors"][min_rate_index, min_load_index], 30.0)
        self.assertAlmostEqual(float(grid["errors"].min()), 2.0)
        self.assertAlmostEqual(float(grid["errors"].max()), 30.0)

    def test_rate_sensitivity_curves_show_error_decreases_with_rate(self):
        curves = generate_rate_sensitivity_curves(load_levels=(70.0, 85.0, 100.0), num_rates=41)

        self.assertEqual(set(curves["errors_by_load"]), {70.0, 85.0, 100.0})
        for errors in curves["errors_by_load"].values():
            self.assertLess(errors[-1], errors[0])
        self.assertAlmostEqual(curves["errors_by_load"][85.0][0], 25.0)
        self.assertAlmostEqual(curves["errors_by_load"][85.0][-1], 5.0)

    def test_generate_all_question1_plots_writes_four_nonempty_pngs(self):
        result = solve_question1()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            paths = generate_all_question1_plots(result=result, output_dir=output_dir)

            self.assertEqual(
                {path.name for path in paths},
                {
                    "error_surface_3d.png",
                    "power_surface_3d.png",
                    "feasible_region_2d.png",
                    "power_breakdown_curve.png",
                },
            )
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
                self.assertGreater(path.stat().st_size, 10_000, msg=str(path))


if __name__ == "__main__":
    unittest.main()
