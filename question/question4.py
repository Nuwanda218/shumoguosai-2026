"""问题四：加入 GPU 负载和传输速率变化率约束的动态调度。

本文件只负责问题四求解，不做可视化、不写结果文件。

模型口径：
- 继承问题三的电池储能、冷却惯性和长期循环 SOC 口径；
- 默认使用问题三推荐的循环保留电量 24 kWh；
- 新增 GPU 负载变化率约束 |G_{t+1}-G_t| <= 8；
- 新增传输速率变化率约束 |R_{t+1}-R_t| <= 300；
- 两个变化率约束均包含 24 小时日周期边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

# 支持两种运行方式：
# 1. 推荐：在项目根目录运行 `python -m question.question4`
# 2. 兼容：直接运行 `python question/question4.py`
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.components import cooling_steady_power
from models.parameters import DEFAULT_PARAMS, ModelParameters
from question.question3 import (
    Question3Result,
    _bounds,
    _combine_variables,
    _cooling_deltas,
    _initial_guesses,
    _is_feasible,
    _make_constraints,
    _make_result,
    _objective,
    _split_variables,
    format_question3_result,
    solve_question3,
)


RECOMMENDED_RESERVE_SOC_KWH = 24.0


@dataclass
class Question4Result(Question3Result):
    """问题四求解结果。

    问题四与问题三的变量完全相同，只额外记录 GPU 负载和传输速率的
    最大日周期变化幅度，以及该结果是否直接沿用问题三可行解。
    """

    max_load_delta: float
    max_rate_delta: float
    source: str


def cyclic_signed_deltas(values: np.ndarray) -> np.ndarray:
    """计算包含日周期边界的相邻差值。

    返回值第 t 项为下一小时减当前小时；最后一项为第 1 小时减第 24 小时。
    """

    values = np.asarray(values, dtype=float)
    return np.roll(values, -1) - values


def cyclic_abs_deltas(values: np.ndarray) -> np.ndarray:
    """计算包含日周期边界的相邻绝对变化幅度。"""

    return np.abs(cyclic_signed_deltas(values))


def _to_question4_result(result: Question3Result, *, source: str) -> Question4Result:
    """把问题三结果扩展为问题四结果。"""

    return Question4Result(
        **result.__dict__,
        max_load_delta=float(np.max(cyclic_abs_deltas(result.loads))),
        max_rate_delta=float(np.max(cyclic_abs_deltas(result.rates))),
        source=source,
    )


def is_question4_feasible(
    result: Question4Result | Question3Result,
    params: ModelParameters = DEFAULT_PARAMS,
    *,
    tol: float = 1e-4,
) -> bool:
    """判断结果是否满足问题四新增约束和问题三基础约束。"""

    base_result = result if isinstance(result, Question3Result) else Question3Result(**result.__dict__)
    max_load_delta = float(np.max(cyclic_abs_deltas(result.loads)))
    max_rate_delta = float(np.max(cyclic_abs_deltas(result.rates)))

    return bool(
        _is_feasible(base_result, params, tol=tol)
        and max_load_delta <= params.gpu_change_limit_percent + tol
        and max_rate_delta <= params.rate_change_limit_mbps + tol
    )


def _make_question4_constraints(reserve_soc_kwh: float, params: ModelParameters) -> list[dict]:
    """构造问题四 SLSQP 约束列表。"""

    constraints = list(_make_constraints(reserve_soc_kwh, params))

    def arrays(x: np.ndarray) -> tuple[np.ndarray, ...]:
        return _split_variables(x, params)

    def load_delta_upper_margin(x: np.ndarray) -> np.ndarray:
        loads, *_ = arrays(x)
        return params.gpu_change_limit_percent - cyclic_signed_deltas(loads)

    def load_delta_lower_margin(x: np.ndarray) -> np.ndarray:
        loads, *_ = arrays(x)
        return params.gpu_change_limit_percent + cyclic_signed_deltas(loads)

    def rate_delta_upper_margin(x: np.ndarray) -> np.ndarray:
        _, rates, *_ = arrays(x)
        return params.rate_change_limit_mbps - cyclic_signed_deltas(rates)

    def rate_delta_lower_margin(x: np.ndarray) -> np.ndarray:
        _, rates, *_ = arrays(x)
        return params.rate_change_limit_mbps + cyclic_signed_deltas(rates)

    constraints.extend(
        [
            {"type": "ineq", "fun": load_delta_upper_margin},
            {"type": "ineq", "fun": load_delta_lower_margin},
            {"type": "ineq", "fun": rate_delta_upper_margin},
            {"type": "ineq", "fun": rate_delta_lower_margin},
        ]
    )
    return constraints


def _question4_initial_guesses(
    reserve_soc_kwh: float,
    params: ModelParameters,
    warm_start: Question3Result | None,
) -> list[np.ndarray]:
    """构造问题四初始解列表。"""

    guesses: list[np.ndarray] = []
    if warm_start is not None:
        guesses.append(
            _combine_variables(
                warm_start.loads,
                warm_start.rates,
                warm_start.cooling_powers,
                warm_start.charge_powers,
                warm_start.discharge_powers,
            )
        )
    guesses.extend(_initial_guesses(reserve_soc_kwh, params))
    return guesses


def _solve_question4_by_optimization(
    *,
    reserve_soc_kwh: float,
    params: ModelParameters,
    warm_start: Question3Result | None,
) -> Question4Result:
    """在问题三结果不满足新增约束时，重新求解问题四模型。"""

    best_feasible: Question4Result | None = None
    best_any: Question4Result | None = None
    constraints = _make_question4_constraints(reserve_soc_kwh, params)

    for start in _question4_initial_guesses(reserve_soc_kwh, params, warm_start):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step, clipping to bounds",
                category=RuntimeWarning,
            )
            opt = minimize(
                fun=lambda x: _objective(x, reserve_soc_kwh=reserve_soc_kwh, params=params),
                x0=start,
                method="SLSQP",
                bounds=_bounds(params),
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1800, "disp": False},
            )

        candidate = opt.x if opt.x is not None else start
        summary = _make_result(
            candidate,
            reserve_soc_kwh=reserve_soc_kwh,
            success=opt.success,
            message=opt.message,
            params=params,
        )
        question4_summary = _to_question4_result(summary, source="question4-optimized")

        if best_any is None or question4_summary.total_cost < best_any.total_cost:
            best_any = question4_summary
        if is_question4_feasible(question4_summary, params):
            if best_feasible is None or question4_summary.total_cost < best_feasible.total_cost:
                best_feasible = Question4Result(
                    **{**question4_summary.__dict__, "success": True}
                )

    if best_feasible is not None:
        return best_feasible
    assert best_any is not None
    return best_any


def solve_question4(
    reserve_soc_kwh: float = RECOMMENDED_RESERVE_SOC_KWH,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Question4Result:
    """求解问题四模型。

    问题四只是在问题三基础上增加约束。若问题三推荐方案已经满足新增
    变化率约束，则该方案同时也是问题四最优解；否则再启动带新增约束的
    重新优化。
    """

    question3_result = solve_question3(reserve_soc_kwh=reserve_soc_kwh, params=params)
    inherited = _to_question4_result(question3_result, source="question3-feasible")
    if is_question4_feasible(inherited, params):
        return inherited

    return _solve_question4_by_optimization(
        reserve_soc_kwh=reserve_soc_kwh,
        params=params,
        warm_start=question3_result,
    )


def format_question4_result(result: Question4Result, params: ModelParameters = DEFAULT_PARAMS) -> str:
    """生成问题四中文摘要。"""

    lines = [
        "========== 问题四优化结果 ==========",
        f"结果来源：{result.source}",
        f"循环保留电量：{result.reserve_soc_kwh:.1f} kWh",
        f"求解状态：{'成功' if result.success else '失败'}",
        f"状态信息：{result.message}",
        f"最小日电费：{result.total_cost:.4f} 元",
        f"系统总能耗：{result.total_system_energy:.4f} kWh",
        f"电网购电量：{result.total_grid_energy:.4f} kWh",
        f"总处理量：{result.total_work:.4f}",
        f"加权平均误差：{result.weighted_error:.4f}%",
        f"最大GPU负载变化：{result.max_load_delta:.4f}% / 限制 {params.gpu_change_limit_percent:.1f}%",
        f"最大传输速率变化：{result.max_rate_delta:.4f} Mbps / 限制 {params.rate_change_limit_mbps:.1f} Mbps",
        f"最大冷却功率变化：{result.max_cooling_delta:.4f} kW / 限制 {params.cooling_change_limit_kw:.1f} kW",
        f"总充电量：{result.total_charge_energy:.4f} kWh",
        f"总放电量：{result.total_discharge_energy:.4f} kWh",
        f"峰时段放电量：{result.peak_discharge_energy:.4f} kWh",
        "",
        "小时\t电价\tGPU负载\t传输速率\t实际冷却\t充电\t放电\tSOC起点\t购电功率\t小时电费",
    ]
    for hour in range(params.hours):
        lines.append(
            f"{hour + 1:02d}\t"
            f"{result.prices[hour]:.1f}\t"
            f"{result.loads[hour]:.3f}\t"
            f"{result.rates[hour]:.1f}\t"
            f"{result.cooling_powers[hour]:.4f}\t"
            f"{result.charge_powers[hour]:.4f}\t"
            f"{result.discharge_powers[hour]:.4f}\t"
            f"{result.soc[hour]:.4f}\t"
            f"{result.grid_powers[hour]:.4f}\t"
            f"{result.hourly_costs[hour]:.4f}"
        )
    lines.append(f"终止SOC\t\t\t\t\t\t\t{result.soc[-1]:.4f}")
    return "\n".join(lines)


def format_question4_comparison(result: Question4Result, params: ModelParameters = DEFAULT_PARAMS) -> str:
    """生成问题四新增约束是否绑定的解释。"""

    rate_binding = result.max_rate_delta >= params.rate_change_limit_mbps - 1e-5
    load_binding = result.max_load_delta >= params.gpu_change_limit_percent - 1e-5
    cooling_binding = result.max_cooling_delta >= params.cooling_change_limit_kw - 1e-5

    return "\n".join(
        [
            "========== 新增约束分析 ==========",
            f"传输速率变化率约束是否绑定：{'是' if rate_binding else '否'}",
            f"GPU负载变化率约束是否绑定：{'是' if load_binding else '否'}",
            f"冷却功率变化率约束是否绑定：{'是' if cooling_binding else '否'}",
            "说明：若问题三方案已经满足问题四新增约束，则问题四最优费用与问题三相同。",
        ]
    )


def main() -> None:
    """命令行入口：python -m question.question4。"""

    result = solve_question4()
    print(format_question4_result(result))
    print()
    print(format_question4_comparison(result))


if __name__ == "__main__":
    main()
