"""数值辅助函数。

这些函数是内部工具，主要解决一个细节：
模型函数既要支持单个数值，也要支持 24 小时 numpy 数组。

例如 gpu_power_single(85) 应返回普通 float；
gpu_power_single(np.array([...])) 应返回 numpy 数组。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_array(value: Any) -> np.ndarray:
    """把输入转成 float 类型 numpy 数组，方便统一计算。"""
    return np.asarray(value, dtype=float)


def is_scalar_like(value: Any) -> bool:
    """判断输入是否像单个数值，而不是列表或数组。"""
    return np.asarray(value).ndim == 0


def maybe_scalar(result: Any, *references: Any):
    """根据输入类型决定返回 float 还是 ndarray。

    如果所有输入都是单个数值，就返回 float，方便打印和断言；
    只要有一个输入是数组，就返回数组，方便 24 小时逐时计算。
    """
    array = np.asarray(result, dtype=float)
    if all(is_scalar_like(reference) for reference in references):
        return float(array)
    return array
