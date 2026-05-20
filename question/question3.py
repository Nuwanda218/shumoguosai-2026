"""问题三：加入电池储能和冷却惯性的日电费优化。

本文件只负责问题三求解与命令行结果输出，不包含可视化代码。

建模口径：
- 继承问题二的处理量约束和处理量加权平均误差约束；
- 使用 24 小时完整调度变量，而不是峰、平、谷三类变量；
- 引入实际冷却功率 C_t，并限制相邻小时变化不超过 0.2kW；
- 引入电池充放电功率和 SOC 递推；
- 对多个每日循环保留电量 S_res 做梯度对比。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

# 支持直接运行本文件：
#     python question/question3.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.battery import grid_power, next_soc
from models.components import cooling_steady_power
from models.objectives import system_power
from models.parameters import DEFAULT_PARAMS, ModelParameters, price_schedule
from models.task import analysis_error, hourly_work, total_work, weighted_average_error
from question.question2 import build_tariff_group_schedule, tariff_group_indices


# 循环保留电量候选值，单位 kWh。
# 8kWh 对应题目允许的最低 20% SOC；16kWh 对应题目给定初始 40% SOC。
RESERVE_SOC_CANDIDATES = (8.0, 12.0, 16.0, 20.0, 24.0, 32.0)


@dataclass
class Question3Result:
    """问题三单个循环保留电量下的求解结果。"""

    success: bool
    message: str
    reserve_soc_kwh: float
    loads: np.ndarray
    rates: np.ndarray
    cooling_powers: np.ndarray
    charge_powers: np.ndarray
    discharge_powers: np.ndarray
    soc: np.ndarray
    prices: np.ndarray
    system_powers: np.ndarray
    grid_powers: np.ndarray
    hourly_costs: np.ndarray
    total_cost: float
    total_system_energy: float
    total_grid_energy: float
    total_work: float
    weighted_error: float
    max_cooling_delta: float
    max_simultaneous_charge_discharge: float
    total_charge_energy: float
    total_discharge_energy: float
    valley_charge_energy: float
    peak_discharge_energy: float


def _split_variables(variables: np.ndarray, params: ModelParameters) -> tuple[np.ndarray, ...]:
    """把一维优化变量拆成 5 组 24 小时数组。"""

    hours = params.hours
    loads = variables[0:hours]
    rates = variables[hours : 2 * hours]
    cooling_powers = variables[2 * hours : 3 * hours]
    charge_powers = variables[3 * hours : 4 * hours]
    discharge_powers = variables[4 * hours : 5 * hours]
    return loads, rates, cooling_powers, charge_powers, discharge_powers


def _combine_variables(
    loads: np.ndarray,
    rates: np.ndarray,
    cooling_powers: np.ndarray,
    charge_powers: np.ndarray,
    discharge_powers: np.ndarray,
) -> np.ndarray:
    """把 5 组 24 小时数组合并成优化器使用的一维变量。"""

    return np.concatenate([loads, rates, cooling_powers, charge_powers, discharge_powers]).astype(float)


def _soc_trajectory(
    charge_powers: np.ndarray,
    discharge_powers: np.ndarray,
    *,
    reserve_soc_kwh: float,
    params: ModelParameters,
) -> np.ndarray:
    """根据充放电功率递推 25 个 SOC 节点。

    soc[0] 表示第 1 小时开始时电量，soc[24] 表示第 24 小时结束后、
    下一天开始前的电量。
    """

    soc = np.empty(params.hours + 1, dtype=float)
    soc[0] = float(reserve_soc_kwh)
    for hour in range(params.hours):
        soc[hour + 1] = next_soc(
            soc[hour],
            charge_power=charge_powers[hour],
            discharge_power=discharge_powers[hour],
            params=params,
        )
    return soc


def _cooling_deltas(cooling_powers: np.ndarray) -> np.ndarray:
    """计算包含日周期边界的冷却功率相邻小时变化。"""

    return np.roll(cooling_powers, -1) - cooling_powers


def _objective(
    variables: np.ndarray,
    *,
    reserve_soc_kwh: float,
    params: ModelParameters,
) -> float:
    """优化目标：考虑电池后的日电费。"""

    loads, rates, cooling_powers, charge_powers, discharge_powers = _split_variables(variables, params)
    purchase_power = _grid_powers(loads, rates, cooling_powers, charge_powers, discharge_powers, params)
    return float(np.sum(price_schedule(params) * purchase_power))


def _grid_powers(
    loads: np.ndarray,
    rates: np.ndarray,
    cooling_powers: np.ndarray,
    charge_powers: np.ndarray,
    discharge_powers: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """计算 24 小时电网购电功率。"""

    load_power = system_power(loads, rates, cooling_power=cooling_powers, params=params)
    return grid_power(load_power, charge_power=charge_powers, discharge_power=discharge_powers)


def _make_constraints(reserve_soc_kwh: float, params: ModelParameters) -> list[dict]:
    """构造 SLSQP 使用的约束列表。

    SLSQP 的不等式约束格式为 fun(x) >= 0。向量约束表示每个元素都需要
    大于等于 0。
    """

    def arrays(x: np.ndarray) -> tuple[np.ndarray, ...]:
        return _split_variables(x, params)

    def total_work_margin(x: np.ndarray) -> float:
        loads, rates, *_ = arrays(x)
        return total_work(loads, rates, params) - 1.0

    def weighted_error_margin(x: np.ndarray) -> float:
        loads, rates, *_ = arrays(x)
        return params.error_limit_percent - weighted_average_error(loads, rates, params=params)

    def cooling_requirement_margin(x: np.ndarray) -> np.ndarray:
        loads, _, cooling_powers, *_ = arrays(x)
        return cooling_powers - cooling_steady_power(loads, params)

    def cooling_delta_upper_margin(x: np.ndarray) -> np.ndarray:
        _, _, cooling_powers, *_ = arrays(x)
        return params.cooling_change_limit_kw - _cooling_deltas(cooling_powers)

    def cooling_delta_lower_margin(x: np.ndarray) -> np.ndarray:
        _, _, cooling_powers, *_ = arrays(x)
        return params.cooling_change_limit_kw + _cooling_deltas(cooling_powers)

    def soc_lower_margin(x: np.ndarray) -> np.ndarray:
        *_, charge_powers, discharge_powers = arrays(x)
        soc = _soc_trajectory(
            charge_powers,
            discharge_powers,
            reserve_soc_kwh=reserve_soc_kwh,
            params=params,
        )
        return soc - params.battery_min_soc_kwh

    def soc_upper_margin(x: np.ndarray) -> np.ndarray:
        *_, charge_powers, discharge_powers = arrays(x)
        soc = _soc_trajectory(
            charge_powers,
            discharge_powers,
            reserve_soc_kwh=reserve_soc_kwh,
            params=params,
        )
        return params.battery_max_soc_kwh - soc

    def final_soc_equal_reserve(x: np.ndarray) -> float:
        *_, charge_powers, discharge_powers = arrays(x)
        soc = _soc_trajectory(
            charge_powers,
            discharge_powers,
            reserve_soc_kwh=reserve_soc_kwh,
            params=params,
        )
        return soc[-1] - reserve_soc_kwh

    def grid_nonnegative_margin(x: np.ndarray) -> np.ndarray:
        loads, rates, cooling_powers, charge_powers, discharge_powers = arrays(x)
        return _grid_powers(loads, rates, cooling_powers, charge_powers, discharge_powers, params)

    return [
        {"type": "ineq", "fun": total_work_margin},
        {"type": "ineq", "fun": weighted_error_margin},
        {"type": "ineq", "fun": cooling_requirement_margin},
        {"type": "ineq", "fun": cooling_delta_upper_margin},
        {"type": "ineq", "fun": cooling_delta_lower_margin},
        {"type": "ineq", "fun": soc_lower_margin},
        {"type": "ineq", "fun": soc_upper_margin},
        {"type": "ineq", "fun": grid_nonnegative_margin},
        {"type": "eq", "fun": final_soc_equal_reserve},
    ]


def _bounds(params: ModelParameters) -> list[tuple[float, float]]:
    """返回问题三所有优化变量的上下界。"""

    hours = params.hours
    cooling_max_kw = cooling_steady_power(params.gpu_max_load, params)
    return (
        [(params.gpu_min_load, params.gpu_max_load)] * hours
        + [(params.rate_min_mbps, params.rate_max_mbps)] * hours
        + [(params.cooling_min_kw, cooling_max_kw)] * hours
        + [(0.0, params.battery_power_limit_kw)] * hours
        + [(0.0, params.battery_power_limit_kw)] * hours
    )


def _battery_arbitrage_guess(
    loads: np.ndarray,
    rates: np.ndarray,
    cooling_powers: np.ndarray,
    *,
    reserve_soc_kwh: float,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """构造一个可行的电池初始猜测。

    该猜测只在当天前半段谷时段充电，然后在峰时段放电，并保证最终 SOC
    回到 reserve_soc_kwh。它不追求最优，只帮助优化器更快找到含电池调度的区域。
    """

    charge = np.zeros(params.hours, dtype=float)
    discharge = np.zeros(params.hours, dtype=float)
    groups = tariff_group_indices(params)
    system = system_power(loads, rates, cooling_power=cooling_powers, params=params)

    soc = float(reserve_soc_kwh)
    # 只使用 0:00-6:00 的谷时段作为初始充电猜测，避免 22:00-24:00
    # 充电后无处释放，破坏最终 SOC 回到 reserve 的条件。
    early_valley_hours = [idx for idx in groups["valley"] if idx < 6]
    for hour in early_valley_hours:
        room = params.battery_max_soc_kwh - soc
        if room <= 1e-9:
            break
        power = min(params.battery_power_limit_kw, room / params.battery_efficiency)
        charge[hour] = power
        soc = next_soc(soc, charge_power=power, discharge_power=0.0, params=params)

    for hour in groups["peak"]:
        available_output = max((soc - reserve_soc_kwh) * params.battery_efficiency, 0.0)
        if available_output <= 1e-9:
            break
        power = min(params.battery_power_limit_kw, available_output, system[hour])
        discharge[hour] = power
        soc = next_soc(soc, charge_power=0.0, discharge_power=power, params=params)

    # 若因为峰时段系统功率较低导致没有放完，则把多余电量按反向效率折算掉。
    # 这里不强行修正；最终 SOC 等式约束会由优化器负责满足。
    return charge, discharge


def _initial_guesses(reserve_soc_kwh: float, params: ModelParameters) -> list[np.ndarray]:
    """生成多个可行初始点，降低非线性优化陷入差解的概率。"""

    hours = params.hours

    # 静态问题一临界点：G=85, R=1200, C=4.2。它满足任务量和误差约束，
    # 也天然满足冷却功率日周期变化约束。
    static_loads = np.full(hours, 85.0, dtype=float)
    static_rates = np.full(hours, params.rate_max_mbps, dtype=float)
    static_cooling = cooling_steady_power(static_loads, params)
    zero_battery = np.zeros(hours, dtype=float)

    guesses = [
        _combine_variables(static_loads, static_rates, static_cooling, zero_battery, zero_battery)
    ]

    # 问题二动态负载作为第二个初始点。由于冷却惯性限制较强，
    # 这里先把冷却功率设为最大稳态需求，确保初始点满足 C_t >= C*_t
    # 和相邻小时变化约束。
    q2_loads, q2_rates = build_tariff_group_schedule(
        valley_load=100.0,
        valley_rate=params.rate_max_mbps,
        flat_load=78.642081,
        flat_rate=params.rate_max_mbps,
        peak_load=60.0,
        peak_rate=params.rate_max_mbps,
        params=params,
    )
    q2_cooling = np.full(hours, cooling_steady_power(params.gpu_max_load, params), dtype=float)
    guesses.append(_combine_variables(q2_loads, q2_rates, q2_cooling, zero_battery, zero_battery))

    # 在静态负载基础上加入一个简单的电池套利初始猜测。
    charge, discharge = _battery_arbitrage_guess(
        static_loads,
        static_rates,
        static_cooling,
        reserve_soc_kwh=reserve_soc_kwh,
        params=params,
    )
    guesses.append(_combine_variables(static_loads, static_rates, static_cooling, charge, discharge))
    return guesses


def _make_result(
    variables: np.ndarray,
    *,
    reserve_soc_kwh: float,
    success: bool,
    message: str,
    params: ModelParameters,
) -> Question3Result:
    """把优化变量转换为包含各类指标的结果对象。"""

    loads, rates, cooling_powers, charge_powers, discharge_powers = _split_variables(variables, params)

    # 数值优化可能出现 1e-9 量级的边界误差；这里仅用于输出清理。
    loads = np.clip(loads, params.gpu_min_load, params.gpu_max_load)
    rates = np.clip(rates, params.rate_min_mbps, params.rate_max_mbps)
    cooling_powers = np.clip(
        cooling_powers,
        params.cooling_min_kw,
        cooling_steady_power(params.gpu_max_load, params),
    )
    charge_powers = np.clip(charge_powers, 0.0, params.battery_power_limit_kw)
    discharge_powers = np.clip(discharge_powers, 0.0, params.battery_power_limit_kw)

    prices = price_schedule(params)
    soc = _soc_trajectory(
        charge_powers,
        discharge_powers,
        reserve_soc_kwh=reserve_soc_kwh,
        params=params,
    )
    system_powers = system_power(loads, rates, cooling_power=cooling_powers, params=params)
    grid_powers = grid_power(system_powers, charge_power=charge_powers, discharge_power=discharge_powers)
    hourly_costs = prices * grid_powers
    groups = tariff_group_indices(params)

    return Question3Result(
        success=bool(success),
        message=str(message),
        reserve_soc_kwh=float(reserve_soc_kwh),
        loads=loads,
        rates=rates,
        cooling_powers=cooling_powers,
        charge_powers=charge_powers,
        discharge_powers=discharge_powers,
        soc=soc,
        prices=prices,
        system_powers=system_powers,
        grid_powers=grid_powers,
        hourly_costs=hourly_costs,
        total_cost=float(np.sum(hourly_costs)),
        total_system_energy=float(np.sum(system_powers)),
        total_grid_energy=float(np.sum(grid_powers)),
        total_work=total_work(loads, rates, params),
        weighted_error=weighted_average_error(loads, rates, params=params),
        max_cooling_delta=float(np.max(np.abs(_cooling_deltas(cooling_powers)))),
        max_simultaneous_charge_discharge=float(np.max(np.minimum(charge_powers, discharge_powers))),
        total_charge_energy=float(np.sum(charge_powers)),
        total_discharge_energy=float(np.sum(discharge_powers)),
        valley_charge_energy=float(np.sum(charge_powers[groups["valley"]])),
        peak_discharge_energy=float(np.sum(discharge_powers[groups["peak"]])),
    )


def _is_feasible(result: Question3Result, params: ModelParameters, *, tol: float = 1e-4) -> bool:
    """判断结果是否满足问题三主要约束。"""

    return bool(
        result.total_work >= 1.0 - tol
        and result.weighted_error <= params.error_limit_percent + tol
        and np.min(result.soc) >= params.battery_min_soc_kwh - tol
        and np.max(result.soc) <= params.battery_max_soc_kwh + tol
        and abs(result.soc[0] - result.reserve_soc_kwh) <= tol
        and abs(result.soc[-1] - result.reserve_soc_kwh) <= tol
        and result.max_cooling_delta <= params.cooling_change_limit_kw + tol
        and np.min(result.grid_powers) >= -tol
        and np.min(result.cooling_powers - cooling_steady_power(result.loads, params)) >= -tol
    )


def solve_question3(
    reserve_soc_kwh: float = 8.0,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Question3Result:
    """求解指定循环保留电量下的问题三模型。"""

    if not (params.battery_min_soc_kwh <= reserve_soc_kwh <= params.battery_max_soc_kwh):
        raise ValueError("reserve_soc_kwh must be within battery SOC bounds")

    best_feasible: Question3Result | None = None
    best_any: Question3Result | None = None
    constraints = _make_constraints(reserve_soc_kwh, params)

    for start in _initial_guesses(reserve_soc_kwh, params):
        with warnings.catch_warnings():
            # SLSQP 在搜索过程中可能临时走到边界外，SciPy 会自动裁剪回边界内。
            # 这是优化器中间步骤，不代表最终解越界；最终可行性由 _is_feasible 检查。
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

        if best_any is None or summary.total_cost < best_any.total_cost:
            best_any = summary
        if _is_feasible(summary, params):
            if best_feasible is None or summary.total_cost < best_feasible.total_cost:
                # 即使 SLSQP 状态不是 success，只要约束已经满足且目标更低，
                # 对本题数值结果仍然可用；message 保留优化器原始状态。
                best_feasible = Question3Result(**{**summary.__dict__, "success": True})

    if best_feasible is not None:
        return best_feasible
    assert best_any is not None
    return best_any


def solve_question3_reserve_sweep(
    candidates: tuple[float, ...] = RESERVE_SOC_CANDIDATES,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> list[Question3Result]:
    """对多个循环保留电量分别求解问题三模型。"""

    return [solve_question3(reserve_soc_kwh=reserve, params=params) for reserve in candidates]


def select_recommended_result(
    results: list[Question3Result],
    *,
    cost_tolerance: float = 1e-2,
) -> Question3Result:
    """从循环额度对比结果中选择推荐方案。

    若多个方案日电费差距小于 cost_tolerance，说明这些方案经济性几乎相同。
    此时优先选择保留电量更高的方案，以获得更大的运行冗余。
    """

    feasible_results = [result for result in results if result.success]
    candidates = feasible_results if feasible_results else results
    min_cost = min(result.total_cost for result in candidates)
    near_best = [result for result in candidates if result.total_cost <= min_cost + cost_tolerance]
    return max(near_best, key=lambda result: result.reserve_soc_kwh)


def format_question3_result(result: Question3Result) -> str:
    """生成单个循环额度结果的中文摘要。"""

    lines = [
        "========== 问题三优化结果 ==========",
        f"循环保留电量：{result.reserve_soc_kwh:.1f} kWh",
        f"求解状态：{'成功' if result.success else '失败'}",
        f"状态信息：{result.message}",
        f"最小日电费：{result.total_cost:.4f} 元",
        f"系统总能耗：{result.total_system_energy:.4f} kWh",
        f"电网购电量：{result.total_grid_energy:.4f} kWh",
        f"总处理量：{result.total_work:.4f}",
        f"加权平均误差：{result.weighted_error:.4f}%",
        f"总充电量：{result.total_charge_energy:.4f} kWh",
        f"总放电量：{result.total_discharge_energy:.4f} kWh",
        f"谷时段充电量：{result.valley_charge_energy:.4f} kWh",
        f"峰时段放电量：{result.peak_discharge_energy:.4f} kWh",
        f"最大冷却功率变化：{result.max_cooling_delta:.4f} kW",
        f"最大同时充放电量：{result.max_simultaneous_charge_discharge:.6f} kW",
        "",
        "时段\t电价\tGPU负载\t传输速率\t冷却功率\t充电\t放电\tSOC始\t电网功率\t小时电费",
    ]
    for hour in range(result.loads.size):
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


def format_reserve_sweep(results: list[Question3Result]) -> str:
    """生成循环额度对比表。"""

    lines = [
        "========== 循环额度对比 ==========",
        "保留电量\t状态\t日电费\t购电量\t总处理量\t加权误差\t充电量\t放电量\t峰时段放电\t最大冷却变化",
    ]
    for result in results:
        lines.append(
            f"{result.reserve_soc_kwh:.1f}\t"
            f"{'成功' if result.success else '失败'}\t"
            f"{result.total_cost:.4f}\t"
            f"{result.total_grid_energy:.4f}\t"
            f"{result.total_work:.4f}\t"
            f"{result.weighted_error:.4f}\t"
            f"{result.total_charge_energy:.4f}\t"
            f"{result.total_discharge_energy:.4f}\t"
            f"{result.peak_discharge_energy:.4f}\t"
            f"{result.max_cooling_delta:.4f}"
        )

    best = select_recommended_result(results)
    lines.extend(
        [
            "",
            f"推荐循环保留电量：{best.reserve_soc_kwh:.1f} kWh",
            f"推荐方案日电费：{best.total_cost:.4f} 元",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """命令行入口：python -m question.question3。"""

    results = solve_question3_reserve_sweep()
    best = select_recommended_result(results)
    print(format_reserve_sweep(results))
    print()
    print(format_question3_result(best))


if __name__ == "__main__":
    main()
