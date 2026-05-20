import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from question.question3 import Question3Result
from visualization.question3_plots import (
    Q3_COMPARISON_PANEL_COUNT,
    Q3_LEGEND_LOCATIONS,
    Q3_VALUE_LABELS,
    Q3_ZORDERS,
    generate_all_question3_plots,
    generate_question3_hourly_data,
    generate_reserve_comparison_data,
)


def _sample_question3_result(reserve_soc_kwh: float = 24.0, total_cost: float = 158.6) -> Question3Result:
    """构造轻量样例结果，避免可视化测试重复运行第三问优化器。"""

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
    loads = 84.0 + 8.0 * np.cos((hour_index - 2.0) / hours * 2.0 * np.pi)
    rates = np.full(hours, 1200.0)
    cooling_powers = 1.2 + 0.12 * (loads - 60.0) + 0.15
    charge_powers = np.zeros(hours)
    charge_powers[[0, 1, 22, 23]] = [8.0, 6.0, 10.0, 8.0]
    discharge_powers = np.zeros(hours)
    discharge_powers[[9, 10, 18, 19]] = [5.5, 5.3, 5.4, 5.6]
    system_powers = 2.0 + 0.05 * loads + 0.41 + cooling_powers
    grid_powers = np.maximum(system_powers + charge_powers - discharge_powers, 0.0)
    hourly_costs = prices * grid_powers
    soc = reserve_soc_kwh + 8.0 * (1.0 - np.cos(np.linspace(0.0, 2.0 * np.pi, hours + 1))) / 2.0

    return Question3Result(
        success=True,
        message="sample",
        reserve_soc_kwh=reserve_soc_kwh,
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
        total_cost=total_cost,
        total_system_energy=float(np.sum(system_powers)),
        total_grid_energy=float(np.sum(grid_powers)),
        total_work=1.26,
        weighted_error=5.0,
        max_cooling_delta=0.2,
        max_simultaneous_charge_discharge=0.0,
        total_charge_energy=float(np.sum(charge_powers)),
        total_discharge_energy=float(np.sum(discharge_powers)),
        valley_charge_energy=float(np.sum(charge_powers[prices == 0.8])),
        peak_discharge_energy=float(np.sum(discharge_powers[prices == 2.0])),
    )


class Question3VisualizationTests(unittest.TestCase):
    def test_question3_legend_locations_match_requested_layout(self):
        self.assertEqual(Q3_LEGEND_LOCATIONS["battery_schedule"], "lower left")
        self.assertEqual(Q3_LEGEND_LOCATIONS["cooling_inertia"], "lower left")
        self.assertEqual(Q3_LEGEND_LOCATIONS["grid_power_cost"], "lower left")
        self.assertEqual(Q3_LEGEND_LOCATIONS["reserve_soc_comparison"], "lower right")
        self.assertEqual(Q3_LEGEND_LOCATIONS["q2_q3_comparison"], "lower left")
        self.assertEqual(Q3_COMPARISON_PANEL_COUNT, 1)

    def test_time_series_layers_follow_shared_order(self):
        self.assertLess(Q3_ZORDERS["background"], Q3_ZORDERS["tariff_boundaries"])
        self.assertLess(Q3_ZORDERS["tariff_boundaries"], Q3_ZORDERS["bars"])
        self.assertLess(Q3_ZORDERS["bars"], Q3_ZORDERS["line_plots"])
        self.assertLess(Q3_ZORDERS["line_plots"], Q3_ZORDERS["legend"])

    def test_reserve_soc_comparison_bar_labels_are_enabled(self):
        self.assertTrue(Q3_VALUE_LABELS["reserve_soc_comparison"])

    def test_generate_question3_hourly_data_has_expected_series(self):
        result = _sample_question3_result()

        data = generate_question3_hourly_data(result)

        self.assertEqual(data["hours"].shape, (24,))
        self.assertEqual(data["loads"].shape, (24,))
        self.assertEqual(data["rates"].shape, (24,))
        self.assertEqual(data["cooling_steady"].shape, (24,))
        self.assertEqual(data["soc_start"].shape, (24,))
        self.assertEqual(data["soc_end"].shape, (24,))
        self.assertEqual(data["battery_net_powers"].shape, (24,))
        self.assertAlmostEqual(float(data["soc_start"][0]), result.reserve_soc_kwh)
        self.assertAlmostEqual(float(data["soc_end"][-1]), result.reserve_soc_kwh)
        self.assertGreater(float(data["battery_net_powers"][0]), 0.0)
        self.assertLess(float(data["battery_net_powers"][9]), 0.0)

    def test_generate_reserve_comparison_data_orders_candidates(self):
        results = [
            _sample_question3_result(8.0, 159.2),
            _sample_question3_result(16.0, 158.6),
            _sample_question3_result(24.0, 158.6),
        ]

        data = generate_reserve_comparison_data(results)

        self.assertTrue(np.array_equal(data["reserve_soc"], np.array([8.0, 16.0, 24.0])))
        self.assertAlmostEqual(float(data["total_costs"].min()), 158.6)
        self.assertGreater(float(data["peak_discharge_energy"][-1]), 0.0)

    def test_generate_all_question3_plots_writes_six_nonempty_pngs(self):
        results = [
            _sample_question3_result(8.0, 159.2),
            _sample_question3_result(16.0, 158.6),
            _sample_question3_result(24.0, 158.6),
        ]
        recommended = results[-1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            paths = generate_all_question3_plots(
                results=results,
                recommended=recommended,
                output_dir=output_dir,
            )

            self.assertEqual(
                {path.name for path in paths},
                {
                    "reserve_soc_comparison.png",
                    "schedule_profile.png",
                    "battery_schedule.png",
                    "cooling_inertia.png",
                    "grid_power_cost.png",
                    "q2_q3_comparison.png",
                },
            )
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
                self.assertGreater(path.stat().st_size, 10_000, msg=str(path))


if __name__ == "__main__":
    unittest.main()
