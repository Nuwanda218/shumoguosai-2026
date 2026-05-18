"""物理组件功率模型。

这里对应“模块设计.md”中的三个基础模块：
1. GPU 负载功耗模块
2. 数据传输速率功耗模块
3. 冷却系统稳态功耗模块

所有函数都支持两种输入：
- 单个数值：返回 float，方便手算和打印。
- numpy 数组：返回数组，方便一次计算 24 小时调度方案。
"""

from __future__ import annotations

import numpy as np

from ._numeric import as_array, maybe_scalar
from .parameters import DEFAULT_PARAMS, ModelParameters


def gpu_power_single(load, params: ModelParameters = DEFAULT_PARAMS):
    """计算单块 GPU 的功率，单位 W。

    题目给出分段线性关系：
    - 60% 负载时为 180W；
    - 60%-80%：每增加 1% 负载，功率增加 3W；
    - 80%-100%：每增加 1% 负载，功率增加 2W。

    参数 load 表示 GPU 负载百分比，例如 85 表示 85%。
    """
    load_array = as_array(load)

    # 第一段：60 <= load <= 80。
    low_branch = (
        params.gpu_power_at_min_w
        + params.gpu_low_slope_w_per_percent * (load_array - params.gpu_min_load)
    )

    # 第二段：80 < load <= 100。
    # 先算 80% 时的功率，再加上 80% 之后的增量。
    high_branch = (
        params.gpu_power_at_min_w
        + params.gpu_low_slope_w_per_percent * (params.gpu_mid_load - params.gpu_min_load)
        + params.gpu_high_slope_w_per_percent * (load_array - params.gpu_mid_load)
    )

    # np.where 可以同时处理单个数和 24 小时数组。
    result = np.where(load_array <= params.gpu_mid_load, low_branch, high_branch)
    return maybe_scalar(result, load)


def gpu_cluster_power(load, params: ModelParameters = DEFAULT_PARAMS):
    """计算 8 块 GPU 集群总功率，单位 kW。

    gpu_power_single 返回 W；电费和能耗计算使用 kW，
    因此这里乘以 GPU 数量后除以 1000。
    """
    result = params.gpu_count * as_array(gpu_power_single(load, params)) / 1000.0
    return maybe_scalar(result, load)


def transmission_power(rate, params: ModelParameters = DEFAULT_PARAMS):
    """计算数据传输功率，单位 kW。

    题目给出：800Mbps 时每小时消耗 0.25 度；
    速率每提高 200Mbps，每小时能耗增加 0.08 度。
    因为每个时段长度为 1 小时，所以“每小时能耗 kWh”数值等于“功率 kW”。
    """
    rate_array = as_array(rate)
    result = (
        params.transmission_power_at_min_kw
        + params.transmission_power_slope_kw_per_mbps * (rate_array - params.rate_min_mbps)
    )
    return maybe_scalar(result, rate)


def cooling_steady_power(load, params: ModelParameters = DEFAULT_PARAMS):
    """计算冷却系统稳态需求功率，单位 kW。

    这里的 C*_t 是“如果冷却系统能马上跟随 GPU 负载”时的理论功率。
    问题三、问题四会再引入实际冷却功率 C_t，并要求 C_t >= C*_t，
    同时限制 C_t 相邻小时变化不超过 0.2kW。
    """
    load_array = as_array(load)

    # 8 块 GPU 同步调节，集群总负载 = 8 * 单块负载。
    # 题目基准为 8 * 60% = 480%。
    cluster_load_delta = params.gpu_count * (load_array - params.gpu_min_load)

    # 集群总负载每增加 10%，冷却功率增加 0.15kW。
    result = (
        params.cooling_min_kw
        + params.cooling_kw_per_10pct_cluster_load * cluster_load_delta / 10.0
    )
    return maybe_scalar(result, load)
