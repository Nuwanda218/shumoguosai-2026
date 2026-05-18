"""模型参数集中管理。

这个文件只放“题目给定参数”和“建模约定参数”，不写任何优化逻辑。
这样做的好处是：后续如果题目参数、误差阈值、电池容量等数值需要修改，
只需要改这里，其他模块会自动使用新参数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    """AI 算力调度模型的全部基础参数。

    frozen=True 表示参数对象创建后不允许被修改，避免求解过程中误改题目数据。
    如果确实需要测试不同参数，应重新创建一个 ModelParameters 实例。
    """

    # 调度周期：题目要求一天 24 小时，每小时作为一个时段。
    hours: int = 24

    # GPU 集群规模：8 块 GPU 同步调节，因此模型只使用一个负载变量 G_t。
    gpu_count: int = 8

    # GPU 负载范围和功耗分段点，单位为百分比和 W。
    gpu_min_load: float = 60.0
    gpu_mid_load: float = 80.0
    gpu_max_load: float = 100.0

    # 单块 GPU 在 60% 负载时功率为 180W。
    gpu_power_at_min_w: float = 180.0

    # 60%-80% 区间：负载每增加 1 个百分点，功率增加 3W。
    gpu_low_slope_w_per_percent: float = 3.0

    # 80%-100% 区间：负载每增加 1 个百分点，功率增加 2W。
    gpu_high_slope_w_per_percent: float = 2.0

    # 数据传输速率范围，单位 Mbps。
    rate_min_mbps: float = 800.0
    rate_baseline_mbps: float = 1000.0
    rate_max_mbps: float = 1200.0

    # 800 Mbps 时数据传输功率为 0.25kW。
    transmission_power_at_min_kw: float = 0.25

    # 每增加 1 Mbps，传输功率增加 0.08/200 = 0.0004kW。
    transmission_power_slope_kw_per_mbps: float = 0.0004

    # 冷却系统最低功率与负载增长系数。
    cooling_min_kw: float = 1.2

    # GPU 集群总负载每增加 10%，冷却功率增加 0.15kW。
    cooling_kw_per_10pct_cluster_load: float = 0.15

    # 问题三/四的冷却惯性约束：相邻小时变化不超过 0.2kW。
    cooling_change_limit_kw: float = 0.2

    # 处理量基准：G=80%、R=1000Mbps 连续 24 小时刚好完成一天任务。
    baseline_gpu_load: float = 80.0

    # 最低负荷 G=60%、R=800Mbps 时误差为 30%。
    error_at_min_percent: float = 30.0

    # 单块 GPU 负载每增加 1 个百分点，误差降低 0.025%。
    # 本项目默认采用 8 块 GPU 共同作用，所以实际误差模型会再乘 gpu_count。
    error_reduction_per_gpu_percent: float = 0.025

    # 传输速率每降低 1Mbps，误差增加 0.05%。
    error_increase_per_mbps_drop: float = 0.05

    # 题目要求误差不超过 5%。
    error_limit_percent: float = 5.0

    # 电池储能系统参数，SOC 在代码中统一用 kWh 表示。
    battery_capacity_kwh: float = 40.0
    battery_initial_soc_kwh: float = 16.0
    battery_min_soc_kwh: float = 8.0
    battery_max_soc_kwh: float = 40.0
    battery_power_limit_kw: float = 12.0
    battery_efficiency: float = 0.9

    # 问题四的设备稳定性约束。
    gpu_change_limit_percent: float = 8.0
    rate_change_limit_mbps: float = 300.0


# 默认参数对象。大多数函数不显式传参时都会使用它。
DEFAULT_PARAMS = ModelParameters()


def price_schedule(params: ModelParameters = DEFAULT_PARAMS) -> np.ndarray:
    """返回 24 小时分时电价数组，单位：元/kWh。

    数组下标就是小时编号：
    - price[0] 表示 0:00-1:00
    - price[9] 表示 9:00-10:00
    - price[23] 表示 23:00-24:00
    """
    if params.hours != 24:
        raise ValueError("price_schedule is defined for a 24-hour horizon")

    # 先全部设为谷电价 0.8 元/kWh，再覆盖峰、平时段。
    prices = np.full(params.hours, 0.8, dtype=float)

    # 峰时段：9:00-11:00、18:00-20:00。
    prices[[9, 10, 18, 19]] = 2.0

    # 平时段：6:00-9:00、11:00-18:00、20:00-22:00。
    prices[[6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21]] = 1.2
    return prices
