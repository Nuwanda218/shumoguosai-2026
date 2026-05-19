"""问题二：分时电价下的最小电费调度。

本文件只负责问题二求解，不做可视化、不写结果文件。

模型口径：
- 在问题一约束基础上加入 24 小时分时电价；
- 目标由最小能耗改为最小日电费；
- 误差约束采用全天处理量加权平均误差 <= 5%，不采用逐小时瞬时误差；
- 同一电价类别内采用同一组调度变量，最终展开为 24 小时调度表。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

# 支持两种运行方式：
# 1. 推荐：在项目根目录运行 `python -m question.question2`
# 2. 兼容：直接运行 `python question/question2.py`
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.components import cooling_steady_power, gpu_cluster_power, transmission_power
from models.constraints import within_bounds
from models.objectives import daily_cost, daily_energy, system_power
from models.parameters import DEFAULT_PARAMS, ModelParameters, price_schedule
from models.task import total_work, weighted_average_error


@dataclass(frozen=True)
class Question2Result:
    """问题二求解结果。

    既保存峰、平、谷三类调度变量，也保存展开后的 24 小时数组。
    后续写表格或图像时可以直接使用这些字段，避免重复计算。
    """

    success: bool
    message: str
    valley_load: float
    valley_rate: float
    flat_load: float
    flat_rate: float
    peak_load: float
    peak_rate: float
    loads: np.ndarray
    rates: np.ndarray
    prices: np.ndarray
    powers: np.ndarray
    hourly_costs: np.ndarray
    total_cost: float
    baseline_static_cost: float
    cost_saving: float
    cost_saving_rate: float
    total_energy: float
    total_work: float
    weighted_error: float


def tariff_group_indices(params: ModelParameters = DEFAULT_PARAMS) -> dict[str, np.ndarray]:
    """返回峰、平、谷三类电价对应的小时索引。

    索引采用 Python 习惯：0 表示 0:00-1:00，23 表示 23:00-24:00。
    """

    prices = price_schedule(params)
    return {
        "valley": np.where(np.isclose(prices, 0.8))[0],
        "flat": np.where(np.isclose(prices, 1.2))[0],
        "peak": np.where(np.isclose(prices, 2.0))[0],
    }


def build_tariff_group_schedule(
    *,
    valley_load: float,
    valley_rate: float,
    flat_load: float,
    flat_rate: float,
    peak_load: float,
    peak_rate: float,
    params: ModelParameters = DEFAULT_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """把峰、平、谷三类调度变量展开为 24 小时数组。"""

    loads = np.empty(params.hours, dtype=float)
    rates = np.empty(params.hours, dtype=float)
    groups = tariff_group_indices(params)

    loads[groups["valley"]] = float(valley_load)
    rates[groups["valley"]] = float(valley_rate)
    loads[groups["flat"]] = float(flat_load)
    rates[groups["flat"]] = float(flat_rate)
    loads[groups["peak"]] = float(peak_load)
    rates[groups["peak"]] = float(peak_rate)
    return loads, rates


def _variables_to_schedule(
    variables: np.ndarray,
    params: ModelParameters = DEFAULT_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """把优化变量向量转成 24 小时负载和传输速率。"""

    return build_tariff_group_schedule(
        valley_load=variables[0],
        valley_rate=variables[1],
        flat_load=variables[2],
        flat_rate=variables[3],
        peak_load=variables[4],
        peak_rate=variables[5],
        params=params,
    )


def _objective(
    variables: np.ndarray,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """优化目标：一天总电费。"""

    loads, rates = _variables_to_schedule(variables, params)
    return daily_cost(loads, rates, params=params)


def _total_work_margin(
    variables: np.ndarray,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """处理量约束裕量；SLSQP 要求不等式函数 >= 0。"""

    loads, rates = _variables_to_schedule(variables, params)
    return total_work(loads, rates, params) - 1.0


def _weighted_error_margin(
    variables: np.ndarray,
    params: ModelParameters = DEFAULT_PARAMS,
) -> float:
    """误差约束裕量；返回 5 - 全天处理量加权平均误差。"""

    loads, rates = _variables_to_schedule(variables, params)
    return params.error_limit_percent - weighted_average_error(loads, rates, params)


def _baseline_static_cost(params: ModelParameters = DEFAULT_PARAMS) -> float:
    """计算问题一固定方案在问题二电价下的日电费。"""

    loads = np.full(params.hours, 85.0, dtype=float)
    rates = np.full(params.hours, params.rate_max_mbps, dtype=float)
    return daily_cost(loads, rates, params=params)


def _make_result(
    variables: np.ndarray,
    *,
    success: bool,
    message: str,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Question2Result:
    """根据求解变量汇总问题二结果。"""

    loads, rates = _variables_to_schedule(variables, params)
    prices = price_schedule(params)
    powers = np.asarray(system_power(loads, rates, params=params), dtype=float)
    hourly_costs = prices * powers
    total_cost = float(np.sum(hourly_costs))
    baseline_cost = _baseline_static_cost(params)
    cost_saving = baseline_cost - total_cost
    cost_saving_rate = cost_saving / baseline_cost if baseline_cost > 0 else 0.0

    return Question2Result(
        success=bool(success),
        message=str(message),
        valley_load=float(variables[0]),
        valley_rate=float(variables[1]),
        flat_load=float(variables[2]),
        flat_rate=float(variables[3]),
        peak_load=float(variables[4]),
        peak_rate=float(variables[5]),
        loads=loads,
        rates=rates,
        prices=prices,
        powers=powers,
        hourly_costs=hourly_costs,
        total_cost=total_cost,
        baseline_static_cost=baseline_cost,
        cost_saving=cost_saving,
        cost_saving_rate=cost_saving_rate,
        total_energy=daily_energy(loads, rates, params=params),
        total_work=total_work(loads, rates, params),
        weighted_error=weighted_average_error(loads, rates, params),
    )


def _initial_guesses(params: ModelParameters = DEFAULT_PARAMS) -> list[np.ndarray]:
    """提供多组初始点，降低 SLSQP 陷入差解的概率。"""

    return [
        np.array([100.0, params.rate_max_mbps, 85.0, params.rate_max_mbps, 60.0, 800.0]),
        np.array([100.0, params.rate_max_mbps, 80.0, params.rate_max_mbps, 60.0, 800.0]),
        np.array([95.0, params.rate_max_mbps, 80.0, params.rate_max_mbps, 60.0, 800.0]),
        np.array([85.0, params.rate_max_mbps, 85.0, params.rate_max_mbps, 85.0, params.rate_max_mbps]),
        np.array([100.0, params.rate_max_mbps, 100.0, params.rate_max_mbps, 60.0, 800.0]),
    ]


def solve_question2(params: ModelParameters = DEFAULT_PARAMS) -> Question2Result:
    """求解问题二分时电价下的最小电费方案。"""

    bounds = [
        (params.gpu_min_load, params.gpu_max_load),
        (params.rate_min_mbps, params.rate_max_mbps),
        (params.gpu_min_load, params.gpu_max_load),
        (params.rate_min_mbps, params.rate_max_mbps),
        (params.gpu_min_load, params.gpu_max_load),
        (params.rate_min_mbps, params.rate_max_mbps),
    ]
    constraints = [
        {"type": "ineq", "fun": lambda x: _total_work_margin(x, params)},
        {"type": "ineq", "fun": lambda x: _weighted_error_margin(x, params)},
    ]

    best_result = None
    best_summary = None
    for start in _initial_guesses(params):
        result = minimize(
            fun=lambda x: _objective(x, params),
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000},
        )
        candidate = result.x if result.x is not None else start
        summary = _make_result(
            candidate,
            success=result.success,
            message=result.message,
            params=params,
        )
        if best_summary is None or summary.total_cost < best_summary.total_cost:
            best_result = result
            best_summary = summary

    assert best_result is not None
    assert best_summary is not None

    feasible = (
        best_summary.success
        and within_bounds(best_summary.loads, params.gpu_min_load, params.gpu_max_load)
        and within_bounds(best_summary.rates, params.rate_min_mbps, params.rate_max_mbps)
        and best_summary.total_work >= 1.0 - 1e-7
        and best_summary.weighted_error <= params.error_limit_percent + 1e-7
    )

    if feasible:
        return best_summary

    return _make_result(
        best_result.x if best_result.x is not None else _initial_guesses(params)[0],
        success=False,
        message=f"{best_result.message}; post-check failed",
        params=params,
    )


def format_question2_result(result: Question2Result) -> str:
    """生成适合命令行查看的中文结果摘要。"""

    lines = [
        "========== 问题二优化结果 ==========",
        f"求解状态：{'成功' if result.success else '失败'}",
        f"状态信息：{result.message}",
        f"谷时段 GPU 负载：{result.valley_load:.4f}%，传输速率：{result.valley_rate:.4f} Mbps",
        f"平时段 GPU 负载：{result.flat_load:.4f}%，传输速率：{result.flat_rate:.4f} Mbps",
        f"峰时段 GPU 负载：{result.peak_load:.4f}%，传输速率：{result.peak_rate:.4f} Mbps",
        f"最小日电费：{result.total_cost:.4f} 元",
        f"问题一固定方案日电费：{result.baseline_static_cost:.4f} 元",
        f"电费降低：{result.cost_saving:.4f} 元",
        f"电费降低率：{100 * result.cost_saving_rate:.2f}%",
        f"一天总能耗：{result.total_energy:.4f} kWh",
        f"总处理量：{result.total_work:.4f}",
        f"加权平均误差：{result.weighted_error:.4f}%",
        "",
        "时段\t电价\tGPU负载\t传输速率\t系统功率\t小时电费",
    ]
    for hour, (price, load, rate, power, cost) in enumerate(
        zip(result.prices, result.loads, result.rates, result.powers, result.hourly_costs),
        start=1,
    ):
        lines.append(
            f"{hour:02d}\t{price:.1f}\t{load:.3f}\t{rate:.1f}\t{power:.4f}\t{cost:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    """命令行入口：python -m question.question2。"""

    result = solve_question2()
    print(format_question2_result(result))


if __name__ == "__main__":
    main()

