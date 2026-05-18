"""任务完成量与误差模型。

这个模块不关心电价，也不关心具体优化算法。
它只回答两个问题：
1. 某一小时的 GPU 负载和传输速率能完成多少任务量？
2. 该小时对应的数据分析误差是多少？
"""

from __future__ import annotations

import numpy as np

from ._numeric import as_array, maybe_scalar
from .parameters import DEFAULT_PARAMS, ModelParameters


def hourly_work(load, rate, params: ModelParameters = DEFAULT_PARAMS):
    """计算某小时完成的“日任务量比例”。

    题目给出基准：
    G=80%、R=1000Mbps 连续运行 24 小时刚好完成一天任务。

    因此单小时处理量定义为：
        q_t = (G_t / 80) * (R_t / 1000) / 24

    返回值没有单位。例如返回 1/24 表示该小时完成一天任务量的 1/24。
    """
    load_array = as_array(load)
    rate_array = as_array(rate)

    # 使用乘积正比：GPU 和传输速率任一较低，都会降低该小时处理能力。
    result = (
        (load_array / params.baseline_gpu_load)
        * (rate_array / params.rate_baseline_mbps)
        / params.hours
    )
    return maybe_scalar(result, load, rate)


def total_work(loads, rates, params: ModelParameters = DEFAULT_PARAMS) -> float:
    """计算 24 小时总处理量。

    约束通常写作 total_work(loads, rates) >= 1。
    如果后续希望“不允许超额处理”，也可以改为等式约束 = 1。
    """
    return float(np.sum(hourly_work(loads, rates, params)))


def analysis_error(
    load,
    rate,
    *,
    aggregate_gpu_effect: bool = True,
    params: ModelParameters = DEFAULT_PARAMS,
):
    """计算数据分析误差百分比。

    默认 aggregate_gpu_effect=True，表示采用当前建模说明中的口径：
    8 块 GPU 共同参与信号分析，所以 GPU 负载提升带来的误差降低可以叠加。

    若 aggregate_gpu_effect=False，则表示严格按 PDF 字面“单块 GPU 负载每增加
    1% 误差减少 0.025%”解释。这个口径下即使 G=100、R=1200，误差仍为 9%，
    无法满足 5% 误差约束。
    """
    load_array = as_array(load)
    rate_array = as_array(rate)

    # 是否把 8 块 GPU 的误差降低效果叠加。
    gpu_factor = params.gpu_count if aggregate_gpu_effect else 1

    # E = 30 - 0.025 * gpu_factor * (G - 60) + 0.05 * (800 - R)
    # 当 R 高于 800 时，(800 - R) 为负数，误差会降低。
    result = (
        params.error_at_min_percent
        - params.error_reduction_per_gpu_percent * gpu_factor * (load_array - params.gpu_min_load)
        + params.error_increase_per_mbps_drop * (params.rate_min_mbps - rate_array)
    )
    return maybe_scalar(result, load, rate)


def weighted_average_error(loads, rates, params: ModelParameters = DEFAULT_PARAMS) -> float:
    """计算全天处理量加权平均误差。

    不是简单平均 24 个小时的误差，而是按每小时完成的任务量 q_t 加权：

        E_bar = sum(q_t * E_t) / sum(q_t)

    这样低处理量小时对全天结果影响小，高处理量小时影响大。
    该函数用于问题二之后的动态调度，避免逐小时误差约束把每小时都锁死。
    """
    work = as_array(hourly_work(loads, rates, params))
    total = float(np.sum(work))
    if total <= 0:
        raise ValueError("total workload must be positive to compute weighted error")

    # 默认使用 8-GPU 叠加误差模型。
    errors = as_array(analysis_error(loads, rates, params=params))
    return float(np.sum(work * errors) / total)
