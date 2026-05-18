import unittest

import numpy as np

from models.battery import grid_power, next_soc
from models.components import (
    cooling_steady_power,
    gpu_cluster_power,
    gpu_power_single,
    transmission_power,
)
from models.constraints import max_abs_cyclic_delta, within_bounds
from models.objectives import daily_cost, daily_energy, system_power
from models.parameters import DEFAULT_PARAMS, price_schedule
from models.task import analysis_error, hourly_work, total_work, weighted_average_error


class ComponentModelTests(unittest.TestCase):
    def test_gpu_power_piecewise_reference_points(self):
        self.assertAlmostEqual(gpu_power_single(60), 180.0)
        self.assertAlmostEqual(gpu_power_single(80), 240.0)
        self.assertAlmostEqual(gpu_power_single(100), 280.0)
        self.assertAlmostEqual(gpu_cluster_power(85), 2.0)

    def test_transmission_and_cooling_reference_points(self):
        self.assertAlmostEqual(transmission_power(800), 0.25)
        self.assertAlmostEqual(transmission_power(1200), 0.41)
        self.assertAlmostEqual(cooling_steady_power(60), 1.2)
        self.assertAlmostEqual(cooling_steady_power(85), 4.2)

    def test_system_power_and_energy(self):
        self.assertAlmostEqual(system_power(85, 1200), 6.61)
        loads = np.full(24, 85.0)
        rates = np.full(24, 1200.0)
        self.assertAlmostEqual(daily_energy(loads, rates), 158.64)


class TaskAndErrorTests(unittest.TestCase):
    def test_baseline_work_is_one_day(self):
        self.assertAlmostEqual(hourly_work(80, 1000), 1 / 24)
        loads = np.full(24, 80.0)
        rates = np.full(24, 1000.0)
        self.assertAlmostEqual(total_work(loads, rates), 1.0)

    def test_aggregate_gpu_error_is_feasible_but_strict_pdf_is_not(self):
        self.assertAlmostEqual(analysis_error(85, 1200), 5.0)
        self.assertAlmostEqual(analysis_error(100, 1200), 2.0)
        self.assertAlmostEqual(analysis_error(100, 1200, aggregate_gpu_effect=False), 9.0)

    def test_weighted_average_error_uses_work_as_weight(self):
        loads = np.array([100.0, 60.0])
        rates = np.array([1200.0, 1200.0])
        expected = np.average(
            [analysis_error(100, 1200), analysis_error(60, 1200)],
            weights=[hourly_work(100, 1200), hourly_work(60, 1200)],
        )
        self.assertAlmostEqual(weighted_average_error(loads, rates), expected)


class TariffBatteryAndConstraintTests(unittest.TestCase):
    def test_price_schedule_counts(self):
        prices = price_schedule()
        self.assertEqual(len(prices), DEFAULT_PARAMS.hours)
        self.assertEqual(int(np.sum(prices == 2.0)), 4)
        self.assertEqual(int(np.sum(prices == 1.2)), 12)
        self.assertEqual(int(np.sum(prices == 0.8)), 8)

    def test_daily_cost_uses_hourly_prices(self):
        loads = np.full(24, 85.0)
        rates = np.full(24, 1200.0)
        self.assertAlmostEqual(daily_cost(loads, rates), 190.368)

    def test_battery_soc_and_grid_power(self):
        self.assertAlmostEqual(next_soc(16.0, charge_power=10.0, discharge_power=0.0), 25.0)
        self.assertAlmostEqual(next_soc(16.0, charge_power=0.0, discharge_power=9.0), 6.0)
        self.assertAlmostEqual(grid_power(6.0, charge_power=2.0, discharge_power=1.5), 6.5)

    def test_bounds_and_cyclic_delta_helpers(self):
        self.assertTrue(within_bounds([60, 80, 100], 60, 100))
        self.assertFalse(within_bounds([59.9, 80, 100], 60, 100))
        self.assertAlmostEqual(max_abs_cyclic_delta([60, 68, 64]), 8.0)


if __name__ == "__main__":
    unittest.main()
