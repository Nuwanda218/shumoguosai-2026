import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from question.question4 import Question4Result
from visualization.question4_plots import (
    Q4_LEGEND_LOCATIONS,
    Q4_ZORDERS,
    generate_all_question4_plots,
    generate_change_rate_data,
    generate_constraint_margin_data,
    generate_constraint_strength_data,
    generate_question4_hourly_data,
)


def _sample_question4_result() -> Question4Result:
    """构造轻量样例结果，避免可视化测试重复运行优化器。"""

    hours = 24
    hour_index = np.arange(hours, dtype=float)
    prices = np.array(
        [
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            1.2,
            1.2,
            1.2,
            2.0,
            2.0,
            1.2,
            1.2,
            1.2,
            1.2,
            1.2,
            1.2,
            1.2,
            2.0,
            2.0,
            1.2,
            1.2,
            0.8,
            0.8,
        ],
        dtype=float,
    )
    loads = 84.6 + 10.0 * np.cos((hour_index - 2.0) / hours * 2.0 * np.pi)
    rates = np.full(hours, 1200.0)
    cooling_powers = 1.2 + 0.12 * (loads - 60.0)
    charge_powers = np.zeros(hours)
    charge_powers[[0, 1, 22, 23]] = [8.0, 6.0, 8.0, 8.0]
    discharge_powers = np.zeros(hours)
    discharge_powers[[9, 10, 18, 19]] = [5.5, 5.3, 5.4, 5.6]
    system_powers = 2.0 + 0.05 * loads + 0.41 + cooling_powers
    grid_powers = np.maximum(system_powers + charge_powers - discharge_powers, 0.0)
    hourly_costs = prices * grid_powers
    soc = 24.0 + 8.0 * (1.0 - np.cos(np.linspace(0.0, 2.0 * np.pi, hours + 1))) / 2.0

    return Question4Result(
        success=True,
        message="sample",
        reserve_soc_kwh=24.0,
        loads=loads,
        rates=rates,
        cooling_powers=cooling_powers,
        charge_powers=charge_powers,
        discharge_powers=discharge_powers,
        soc=soc,
        prices=prices,
        system_powers=system_powers,
        grid_powers=grid_powers,
        hourly_costs=hourly_costs,
        total_cost=158.6374,
        total_system_energy=float(np.sum(system_powers)),
        total_grid_energy=float(np.sum(grid_powers)),
        total_work=1.2690,
        weighted_error=5.0,
        max_cooling_delta=0.2,
        max_simultaneous_charge_discharge=0.0,
        total_charge_energy=float(np.sum(charge_powers)),
        total_discharge_energy=float(np.sum(discharge_powers)),
        valley_charge_energy=float(np.sum(charge_powers[prices == 0.8])),
        peak_discharge_energy=float(np.sum(discharge_powers[prices == 2.0])),
        max_load_delta=1.6667,
        max_rate_delta=0.0,
        source="sample",
    )


class Question4VisualizationTests(unittest.TestCase):
    def test_time_series_layers_follow_shared_order(self):
        self.assertLess(Q4_ZORDERS["background"], Q4_ZORDERS["tariff_boundaries"])
        self.assertLess(Q4_ZORDERS["tariff_boundaries"], Q4_ZORDERS["bars"])
        self.assertLess(Q4_ZORDERS["bars"], Q4_ZORDERS["line_plots"])
        self.assertLess(Q4_ZORDERS["line_plots"], Q4_ZORDERS["legend"])

    def test_legend_locations_avoid_main_data(self):
        self.assertEqual(Q4_LEGEND_LOCATIONS["schedule_profile"], "lower left")
        self.assertEqual(Q4_LEGEND_LOCATIONS["constraint_margin_summary"], "upper left")
        self.assertEqual(Q4_LEGEND_LOCATIONS["q3_q4_comparison"], "upper left")
        self.assertEqual(Q4_LEGEND_LOCATIONS["constraint_strength_comparison"], "upper left")

    def test_generate_question4_hourly_data_has_expected_series(self):
        result = _sample_question4_result()

        data = generate_question4_hourly_data(result)

        self.assertEqual(data["hours"].shape, (24,))
        self.assertEqual(data["loads"].shape, (24,))
        self.assertEqual(data["rates"].shape, (24,))
        self.assertEqual(data["prices"].shape, (24,))
        self.assertEqual(data["cooling_powers"].shape, (24,))

    def test_generate_change_rate_data_includes_daily_boundary(self):
        result = _sample_question4_result()

        data = generate_change_rate_data(result)

        self.assertEqual(data["transition_labels"].shape, (24,))
        self.assertEqual(data["load_deltas"].shape, (24,))
        self.assertEqual(data["rate_deltas"].shape, (24,))
        self.assertEqual(data["cooling_deltas"].shape, (24,))
        self.assertEqual(data["transition_labels"][-1], "24→1")
        self.assertAlmostEqual(float(data["rate_deltas"].max()), 0.0)

    def test_generate_constraint_margin_data_normalizes_limits(self):
        result = _sample_question4_result()

        data = generate_constraint_margin_data(result)

        self.assertEqual(data["labels"].shape, (3,))
        self.assertEqual(data["usage_rates"].shape, (3,))
        self.assertAlmostEqual(float(data["usage_rates"][1]), 0.0)
        self.assertAlmostEqual(float(data["usage_rates"][2]), 100.0)

    def test_generate_constraint_strength_data_compares_main_and_smoothing_constraints(self):
        result = _sample_question4_result()

        data = generate_constraint_strength_data(result)

        self.assertEqual(data["labels"].shape, (4,))
        self.assertEqual(data["constraint_types"].shape, (4,))
        self.assertIn("加权误差", data["labels"])
        self.assertIn("冷却变化", data["labels"])
        self.assertIn("GPU负载变化", data["labels"])
        self.assertIn("传输速率变化", data["labels"])
        self.assertAlmostEqual(float(data["usage_rates"][0]), 100.0)
        self.assertAlmostEqual(float(data["usage_rates"][1]), 100.0)
        self.assertAlmostEqual(float(data["usage_rates"][2]), 20.8, places=1)
        self.assertAlmostEqual(float(data["usage_rates"][3]), 0.0)
        self.assertEqual(set(data["constraint_types"]), {"主约束", "平滑约束"})
        self.assertEqual(data["set_relation"], "主约束区域在平滑约束范围内")
        self.assertEqual(data["optimal_relation"], "问题三最优点 = 问题四最优点")

    def test_generate_all_question4_plots_writes_two_nonempty_pngs(self):
        result = _sample_question4_result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            for stale_name in [
                "schedule_profile.png",
                "change_rate_check.png",
                "constraint_margin_summary.png",
            ]:
                (output_dir / stale_name).write_bytes(b"stale")
            paths = generate_all_question4_plots(result=result, output_dir=output_dir)

            self.assertEqual(
                {path.name for path in paths},
                {
                    "constraint_strength_comparison.png",
                    "q3_q4_comparison.png",
                },
            )
            self.assertEqual(
                {path.name for path in output_dir.glob("*.png")},
                {
                    "constraint_strength_comparison.png",
                    "q3_q4_comparison.png",
                },
            )
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
                self.assertGreater(path.stat().st_size, 10_000, msg=str(path))


if __name__ == "__main__":
    unittest.main()
