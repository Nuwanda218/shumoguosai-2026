"""目标函数模块。

这里把“系统功率、日能耗、日电费”的计算统一起来。
优化脚本只需要调用这些函数，不需要重复写 GPU、传输、冷却和电池公式。
"""

from __future__ import annotations

import numpy as np

from ._numeric import as_array, maybe_scalar
from .battery import grid_power
from .components import cooling_steady_power, gpu_cluster_power, transmission_power
from .parameters import DEFAULT_PARAMS, ModelParameters, price_schedule


def system_power(
    load,
    rate,
    *,
    cooling_power=None,
    params: ModelParameters = DEFAULT_PARAMS,
):
    """计算电池作用前的系统总负荷功率，单位 kW。

    系统总功率由三部分组成：
        GPU 集群功率 + 数据传输功率 + 冷却功率

    cooling_power 默认为 None，此时使用稳态冷却功率 C*_t。
    在问题三/四中，如果已经把实际冷却功率 C_t 作为优化变量，
    就可以通过 cooling_power 参数传入 C_t。
    """

    # Q1/Q2：冷却能立即达到稳态，使用 cooling_steady_power。
    # Q3/Q4：冷却有惯性，外部传入实际冷却功率 cooling_power。
    cooling = cooling_steady_power(load, params) if cooling_power is None else as_array(cooling_power)

    # 支持标量和 24 小时数组。三个分量形状一致时会逐小时相加。
    result = gpu_cluster_power(load, params) + transmission_power(rate, params) + cooling
    return maybe_scalar(result, load, rate)


def daily_energy(
    loads,
    rates,
    *,
    cooling_powers=None,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """计算一天总能耗，单位 kWh。

    因为每个时段长度为 1 小时，所以每小时功率 kW 求和后，
    数值上就是一天总能耗 kWh。
    """
    return float(np.sum(system_power(loads, rates, cooling_power=cooling_powers, params=params)))


def daily_cost(
    loads,
    rates,
    *,
    prices=None,
    cooling_powers=None,
    charge_power=0.0,
    discharge_power=0.0,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """计算一天电费，单位元。

    若不考虑电池，charge_power 和 discharge_power 保持默认 0。
    若考虑电池：
    - charge_power 为 24 小时充电功率数组；
    - discharge_power 为 24 小时放电功率数组。
    """

    # 不显式传入 prices 时，使用题目给定分时电价。
    hourly_prices = price_schedule(params) if prices is None else as_array(prices)

    # 先计算系统负荷，再用电池模块换算为电网购电功率。
    load_power = system_power(loads, rates, cooling_power=cooling_powers, params=params)
    purchase_power = grid_power(load_power, charge_power=charge_power, discharge_power=discharge_power)

    # 每小时费用 = 电价 * 电网购电功率；每小时长度为 1 小时。
    return float(np.sum(hourly_prices * purchase_power))
