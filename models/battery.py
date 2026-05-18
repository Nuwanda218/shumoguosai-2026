"""电池储能模块。

本模块只负责电池相关的能量递推和电网购电功率计算。
充放电互斥、SOC 上下界等约束放在后续优化模型中组合使用。
"""

from __future__ import annotations

from ._numeric import as_array, maybe_scalar
from .parameters import DEFAULT_PARAMS, ModelParameters


def next_soc(
    soc,
    *,
    charge_power,
    discharge_power,
    params: ModelParameters = DEFAULT_PARAMS,
):
    """计算下一小时末电池电量，单位 kWh。

    参数说明：
    - soc：当前小时开始时的电池电量。
    - charge_power：该小时充电功率，单位 kW。
    - discharge_power：该小时放电功率，单位 kW。

    每个时段长度为 1 小时，所以 kW 数值可直接作为该小时 kWh 变化量。
    充电效率为 90%，表示充入 10kWh 电能时，电池有效增加 9kWh；
    放电也按效率折算，输出 9kWh 时，电池需要减少 10kWh。
    """
    result = (
        as_array(soc)
        + params.battery_efficiency * as_array(charge_power)
        - as_array(discharge_power) / params.battery_efficiency
    )
    return maybe_scalar(result, soc, charge_power, discharge_power)


def grid_power(system_load_power, *, charge_power=0.0, discharge_power=0.0):
    """计算考虑电池后的电网购电功率，单位 kW。

    公式：
        P_grid = P_system + P_charge - P_discharge

    充电会增加电网购电，放电会减少电网购电。
    实际优化时还应增加 P_grid >= 0，避免出现“向电网反送电”的不合理结果。
    """
    result = as_array(system_load_power) + as_array(charge_power) - as_array(discharge_power)
    return maybe_scalar(result, system_load_power, charge_power, discharge_power)
