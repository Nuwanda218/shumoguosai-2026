"""问题一：静态最小能耗求解。

本文件只负责问题一的求解，不做可视化、不写结果文件。

模型口径：
- 一天 24 小时采用同一 GPU 负载 G 和传输速率 R；
- 底层功率、处理量、误差公式全部复用 models/；
- 目标为最小化一天总能耗；
- 约束为总处理量 >= 1、日均加权误差 <= 5%、变量上下界。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

# 支持两种运行方式：
# 1. 推荐：在项目根目录运行 `python -m question.question1`
# 2. 兼容：直接运行 `python question/question1.py`
#
# 第 2 种方式下，Python 默认只把 question/ 加入 sys.path，
# 项目根目录不在导入路径中，因此会找不到 models 包。
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.components import cooling_steady_power, gpu_cluster_power, transmission_power
from models.constraints import within_bounds
from models.objectives import daily_energy, system_power
from models.parameters import DEFAULT_PARAMS, ModelParameters
from models.task import total_work, weighted_average_error


@dataclass(frozen=True)
class Question1Result:
    """问题一求解结果。

    字段尽量直接对应论文中需要展示的量，避免后续写论文时再反复计算。
    """

    success: bool
    message: str
    gpu_load: float
    transmission_rate: float
    gpu_power: float
    transmission_power: float
    cooling_power: float
    system_power: float
    total_energy: float
    total_work: float
    weighted_error: float


def build_static_schedule(
    gpu_load: float,
    transmission_rate: float,
    params: ModelParameters = DEFAULT_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """把静态方案扩展成 24 小时数组。

    models.task.total_work() 和 weighted_average_error() 面向 24 小时序列，
    所以问题一虽然只有两个决策变量，也统一转成 hourly schedule。
    """

    loads = np.full(params.hours, float(gpu_load), dtype=float)
    rates = np.full(params.hours, float(transmission_rate), dtype=float)
    return loads, rates


def _objective(
    variables: np.ndarray,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """优化目标：一天总能耗 kWh。"""

    gpu_load, transmission_rate = variables
    loads, rates = build_static_schedule(gpu_load, transmission_rate, params)
    return daily_energy(loads, rates, params=params)


def _total_work_margin(
    variables: np.ndarray,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """处理量约束裕量；SLSQP 要求不等式函数 >= 0。"""

    gpu_load, transmission_rate = variables
    loads, rates = build_static_schedule(gpu_load, transmission_rate, params)
    return total_work(loads, rates, params) - 1.0


def _weighted_error_margin(
    variables: np.ndarray,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """误差约束裕量；返回 5 - 加权误差。"""

    gpu_load, transmission_rate = variables
    loads, rates = build_static_schedule(gpu_load, transmission_rate, params)
    return params.error_limit_percent - weighted_average_error(loads, rates, params)


def _make_result(
    variables: np.ndarray,
    *,
    success: bool,
    message: str,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Question1Result:
    """根据求解变量汇总论文所需指标。"""

    gpu_load = float(variables[0])
    transmission_rate = float(variables[1])
    loads, rates = build_static_schedule(gpu_load, transmission_rate, params)

    gpu_power_value = float(gpu_cluster_power(gpu_load, params))
    transmission_power_value = float(transmission_power(transmission_rate, params))
    cooling_power_value = float(cooling_steady_power(gpu_load, params))
    system_power_value = float(system_power(gpu_load, transmission_rate, params=params))

    return Question1Result(
        success=bool(success),
        message=str(message),
        gpu_load=gpu_load,
        transmission_rate=transmission_rate,
        gpu_power=gpu_power_value,
        transmission_power=transmission_power_value,
        cooling_power=cooling_power_value,
        system_power=system_power_value,
        total_energy=daily_energy(loads, rates, params=params),
        total_work=total_work(loads, rates, params),
        weighted_error=weighted_average_error(loads, rates, params),
    )


def solve_question1(params: ModelParameters = DEFAULT_PARAMS) -> Question1Result:
    """求解问题一静态最小能耗方案。"""

    bounds = [
        (params.gpu_min_load, params.gpu_max_load),
        (params.rate_min_mbps, params.rate_max_mbps),
    ]

    constraints = [
        {"type": "ineq", "fun": lambda x: _total_work_margin(x, params)},
        {"type": "ineq", "fun": lambda x: _weighted_error_margin(x, params)},
    ]

    result = minimize(
        fun=lambda x: _objective(x, params),
        x0=np.array([85.0, params.rate_max_mbps], dtype=float),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1000},
    )

    candidate = result.x if result.x is not None else np.array([np.nan, np.nan])
    summary = _make_result(
        candidate,
        success=result.success,
        message=result.message,
        params=params,
    )

    # SLSQP 的 success 只表示数值算法收敛。这里再做一次模型约束检查，
    # 避免把边界误差或不可行点误当作可用结果。
    feasible = (
        summary.success
        and within_bounds([summary.gpu_load], params.gpu_min_load, params.gpu_max_load)
        and within_bounds([summary.transmission_rate], params.rate_min_mbps, params.rate_max_mbps)
        and summary.total_work >= 1.0 - 1e-7
        and summary.weighted_error <= params.error_limit_percent + 1e-7
    )

    if feasible:
        return summary

    return _make_result(
        candidate,
        success=False,
        message=f"{result.message}; post-check failed",
        params=params,
    )


def format_question1_result(result: Question1Result) -> str:
    """生成适合命令行查看的中文结果摘要。"""

    lines = [
        "========== 问题一优化结果 ==========",
        f"求解状态：{'成功' if result.success else '失败'}",
        f"状态信息：{result.message}",
        f"最优 GPU 负载：{result.gpu_load:.4f}%",
        f"最优数据传输速率：{result.transmission_rate:.4f} Mbps",
        f"GPU 集群功率：{result.gpu_power:.4f} kW",
        f"数据传输功率：{result.transmission_power:.4f} kW",
        f"冷却功率：{result.cooling_power:.4f} kW",
        f"系统总功率：{result.system_power:.4f} kW",
        f"一天总能耗：{result.total_energy:.4f} kWh",
        f"总处理量：{result.total_work:.4f}",
        f"加权平均误差：{result.weighted_error:.4f}%",
    ]
    return "\n".join(lines)


def main() -> None:
    """命令行入口：python -m question.question1。"""

    result = solve_question1()
    print(format_question1_result(result))


if __name__ == "__main__":
    main()
