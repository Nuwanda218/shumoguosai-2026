"""通用约束检查工具。

这些函数不直接构造 scipy 约束字典，而是提供可复用的数值检查：
- 判断变量是否在上下界内；
- 计算日周期相邻小时变化量；
- 计算某类约束的最大违约量。

后续 Q1-Q4 的求解脚本可以用这些函数做结果校验。
"""

from __future__ import annotations

import numpy as np

from ._numeric import as_array


def within_bounds(values, lower: float, upper: float, *, tol: float = 1e-9) -> bool:
    """判断所有数值是否位于 [lower, upper] 内。

    tol 是容差。优化器可能返回 60.0000000001 或 59.9999999999，
    这种极小数值误差不应被视为违反约束。
    """
    array = as_array(values)
    return bool(np.all(array >= lower - tol) and np.all(array <= upper + tol))


def cyclic_deltas(values) -> np.ndarray:
    """计算日周期相邻小时变化量。

    对 24 小时序列 [x0, x1, ..., x23]，返回：
        [x1-x0, x2-x1, ..., x23-x22, x0-x23]

    最后一项 x0-x23 用来表示“次日 0 点”和“当天 23 点”的衔接，
    避免调度结果在日边界突然跳变。
    """
    array = as_array(values)
    if array.ndim != 1:
        raise ValueError("cyclic_deltas expects a one-dimensional sequence")
    return np.roll(array, -1) - array


def max_abs_cyclic_delta(values) -> float:
    """返回日周期相邻变化量的最大绝对值。"""
    return float(np.max(np.abs(cyclic_deltas(values))))


def bounds_violation(values, lower: float, upper: float) -> float:
    """计算上下界约束的最大违约量。

    返回 0 表示没有违反约束；
    返回正数表示至少有一个值越界，数值越大表示越界越严重。
    """
    array = as_array(values)
    below = np.maximum(lower - array, 0.0)
    above = np.maximum(array - upper, 0.0)
    return float(np.max(np.maximum(below, above)))


def cyclic_delta_violation(values, limit: float) -> float:
    """计算日周期变化率约束的最大违约量。

    例如问题四 GPU 负载变化率要求 limit=8；
    若最大相邻变化为 10，则返回 2。
    """
    return float(max(max_abs_cyclic_delta(values) - limit, 0.0))
