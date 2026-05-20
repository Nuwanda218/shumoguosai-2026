"""问题三可视化图像生成。

本模块只负责把问题三求解结果整理成论文图像，不修改优化模型。
所有图中文字均使用中文，图像统一保存到 outputs/question3/。
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 支持直接运行本文件：
#     python visualization/question3_plots.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.components import cooling_steady_power
from models.parameters import DEFAULT_PARAMS, ModelParameters
from models.task import analysis_error, hourly_work
from question.question2 import Question2Result, solve_question2
from question.question3 import (
    Question3Result,
    select_recommended_result,
    solve_question3_reserve_sweep,
)
from visualization.style import DEFAULT_STYLE, PAPER_PALETTE, apply_chinese_style


TARIFF_SHADE_ALPHA = 0.54
TARIFF_BOUNDARY_COLOR = PAPER_PALETTE[0]
TARIFF_BOUNDARY_HALO = "#FFFFFF"
Q3_COLORS = {
    "valley": PAPER_PALETTE[3],
    "flat": PAPER_PALETTE[2],
    "peak": PAPER_PALETTE[9],
    "load": PAPER_PALETTE[0],
    "rate": PAPER_PALETTE[7],
    "charge": PAPER_PALETTE[2],
    "discharge": PAPER_PALETTE[8],
    "soc": PAPER_PALETTE[5],
    "cooling_actual": PAPER_PALETTE[0],
    "cooling_steady": PAPER_PALETTE[8],
    "system": PAPER_PALETTE[5],
    "grid": PAPER_PALETTE[0],
    "cost": PAPER_PALETTE[9],
    "q2": PAPER_PALETTE[1],
    "q3": PAPER_PALETTE[7],
}
Q3_LEGEND_LOCATIONS = {
    "reserve_soc_comparison": "lower right",
    "battery_schedule": "lower left",
    "cooling_inertia": "lower left",
    "grid_power_cost": "lower left",
    "q2_q3_comparison": "lower left",
}
Q3_COMPARISON_PANEL_COUNT = 1
Q3_ZORDERS = {
    "background": 0,
    "tariff_boundaries": 2,
    "bars": 20,
    "line_plots": 60,
    "legend": 100,
}
Q3_VALUE_LABELS = {
    "reserve_soc_comparison": True,
}


def ensure_output_dir(output_dir: str | Path) -> Path:
    """确保输出目录存在，并返回 Path 对象。"""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, path: Path) -> Path:
    """保存图像并关闭画布，避免批量绘图时占用过多内存。"""

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _tariff_color(price: float) -> str:
    """按电价返回峰、平、谷背景色。"""

    if np.isclose(price, 2.0):
        return Q3_COLORS["peak"]
    if np.isclose(price, 1.2):
        return Q3_COLORS["flat"]
    return Q3_COLORS["valley"]


def _tariff_periods(prices: np.ndarray) -> list[tuple[int, int, float]]:
    """把逐小时电价合并为连续时段，便于绘制背景色带。"""

    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        return []

    periods: list[tuple[int, int, float]] = []
    start_hour = 1
    current_price = prices[0]
    for hour_index in range(2, prices.size + 1):
        next_price = prices[hour_index - 1]
        if not np.isclose(next_price, current_price):
            periods.append((start_hour, hour_index - 1, current_price))
            start_hour = hour_index
            current_price = next_price
    periods.append((start_hour, prices.size, current_price))
    return periods


def _shade_tariff_background(ax: plt.Axes, prices: np.ndarray) -> list[Patch]:
    """为 24 小时图添加峰、平、谷电价背景。"""

    for start_hour, end_hour, price in _tariff_periods(prices):
        ax.axvspan(
            start_hour - 0.5,
            end_hour + 0.5,
            color=_tariff_color(price),
            alpha=TARIFF_SHADE_ALPHA,
            zorder=Q3_ZORDERS["background"],
        )

    return [
        Patch(facecolor=Q3_COLORS["peak"], alpha=TARIFF_SHADE_ALPHA, label="峰时段"),
        Patch(facecolor=Q3_COLORS["flat"], alpha=TARIFF_SHADE_ALPHA, label="平时段"),
        Patch(facecolor=Q3_COLORS["valley"], alpha=TARIFF_SHADE_ALPHA, label="谷时段"),
    ]


def _draw_tariff_boundaries(ax: plt.Axes, prices: np.ndarray) -> None:
    """在峰、平、谷切换处绘制双层分界线。"""

    prices = np.asarray(prices, dtype=float)
    for hour_index in range(2, prices.size + 1):
        if np.isclose(prices[hour_index - 1], prices[hour_index - 2]):
            continue

        boundary_x = hour_index - 0.5
        ax.axvline(
            boundary_x,
            color=TARIFF_BOUNDARY_HALO,
            linewidth=5.0,
            alpha=0.98,
            zorder=Q3_ZORDERS["tariff_boundaries"],
        )
        ax.axvline(
            boundary_x,
            color=TARIFF_BOUNDARY_COLOR,
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            alpha=0.98,
            zorder=Q3_ZORDERS["tariff_boundaries"] + 1,
        )


def _make_twin_axis_transparent(ax: plt.Axes) -> None:
    """透明化双 Y 轴底色，使电价背景色带保持可见。"""

    ax.patch.set_visible(False)


def _set_hour_axis(ax: plt.Axes) -> None:
    """统一 24 小时横轴刻度。"""

    ax.set_xlim(0.5, 24.5)
    ax.set_xticks(np.arange(1, 25, 1))
    ax.set_xlabel("小时")


def _style_legend(legend) -> None:
    """统一图例样式，保证图例位于最上层且不透明。"""

    if legend is None:
        return
    legend.set_zorder(Q3_ZORDERS["legend"])
    frame = legend.get_frame()
    frame.set_alpha(1.0)
    frame.set_facecolor("#FFFFFF")
    frame.set_edgecolor("#D0D0D0")


def generate_question3_hourly_data(
    result: Question3Result,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, np.ndarray | float]:
    """整理问题三推荐方案的逐小时绘图数据。"""

    hours = np.arange(1, params.hours + 1)
    loads = np.asarray(result.loads, dtype=float)
    rates = np.asarray(result.rates, dtype=float)
    prices = np.asarray(result.prices, dtype=float)
    charge_powers = np.asarray(result.charge_powers, dtype=float)
    discharge_powers = np.asarray(result.discharge_powers, dtype=float)
    soc = np.asarray(result.soc, dtype=float)

    if soc.size >= params.hours + 1:
        soc_start = soc[: params.hours]
        soc_end = soc[1 : params.hours + 1]
    else:
        # 兼容只有 24 个 SOC 采样点的外部结果，最后一小时终值沿用末值。
        soc_start = soc[: params.hours]
        soc_end = np.concatenate([soc[1:params.hours], soc[-1:]])

    return {
        "hours": hours,
        "loads": loads,
        "rates": rates,
        "prices": prices,
        "cooling_steady": cooling_steady_power(loads, params),
        "cooling_powers": np.asarray(result.cooling_powers, dtype=float),
        "charge_powers": charge_powers,
        "discharge_powers": discharge_powers,
        "battery_net_powers": charge_powers - discharge_powers,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "system_powers": np.asarray(result.system_powers, dtype=float),
        "grid_powers": np.asarray(result.grid_powers, dtype=float),
        "hourly_costs": np.asarray(result.hourly_costs, dtype=float),
        "instant_errors": analysis_error(loads, rates, params=params),
        "hourly_work": hourly_work(loads, rates, params=params),
        "weighted_error": float(result.weighted_error),
        "total_cost": float(result.total_cost),
        "reserve_soc_kwh": float(result.reserve_soc_kwh),
    }


def generate_reserve_comparison_data(results: list[Question3Result]) -> dict[str, np.ndarray]:
    """整理不同循环保留电量的对比数据。"""

    sorted_results = sorted(results, key=lambda item: item.reserve_soc_kwh)
    return {
        "reserve_soc": np.array([item.reserve_soc_kwh for item in sorted_results], dtype=float),
        "total_costs": np.array([item.total_cost for item in sorted_results], dtype=float),
        "grid_energy": np.array([item.total_grid_energy for item in sorted_results], dtype=float),
        "charge_energy": np.array([item.total_charge_energy for item in sorted_results], dtype=float),
        "discharge_energy": np.array([item.total_discharge_energy for item in sorted_results], dtype=float),
        "peak_discharge_energy": np.array([item.peak_discharge_energy for item in sorted_results], dtype=float),
        "max_cooling_delta": np.array([item.max_cooling_delta for item in sorted_results], dtype=float),
    }


def plot_reserve_soc_comparison(
    results: list[Question3Result],
    recommended: Question3Result,
    output_dir: str | Path,
) -> Path:
    """绘制循环保留电量对日电费和峰时段放电量的影响。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "reserve_soc_comparison.png"
    data = generate_reserve_comparison_data(results)
    x = np.arange(data["reserve_soc"].size)

    fig, ax_cost = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    cost_min = float(np.min(data["total_costs"]))
    cost_max = float(np.max(data["total_costs"]))
    cost_baseline = cost_min - 0.25
    cost_bars = ax_cost.bar(
        x,
        data["total_costs"] - cost_baseline,
        bottom=cost_baseline,
        width=0.58,
        color=Q3_COLORS["flat"],
        edgecolor=Q3_COLORS["load"],
        linewidth=1.0,
        label="日电费",
        zorder=Q3_ZORDERS["bars"],
    )
    if Q3_VALUE_LABELS["reserve_soc_comparison"]:
        for bar, cost in zip(cost_bars, data["total_costs"]):
            ax_cost.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(cost) + 0.015,
                f"{float(cost):.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=Q3_COLORS["system"],
                zorder=Q3_ZORDERS["line_plots"],
            )
    ax_cost.plot(
        x,
        data["total_costs"],
        color=Q3_COLORS["load"],
        marker="o",
        linewidth=2.2,
        label="日电费趋势",
        zorder=Q3_ZORDERS["line_plots"],
    )
    recommended_index = int(np.where(np.isclose(data["reserve_soc"], recommended.reserve_soc_kwh))[0][-1])
    ax_cost.scatter(
        [recommended_index],
        [recommended.total_cost],
        s=95,
        color=Q3_COLORS["discharge"],
        edgecolor="#FFFFFF",
        linewidth=1.2,
        label="推荐循环额度",
        zorder=Q3_ZORDERS["line_plots"] + 5,
    )
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels([f"{value:.0f}" for value in data["reserve_soc"]])
    ax_cost.set_xlabel("每日结束保留电量（kWh）")
    ax_cost.set_ylabel("日电费（元）")
    ax_cost.set_title("不同循环保留电量下的经济性对比")
    ax_cost.set_ylim(cost_baseline, cost_max + 0.28)
    ax_cost.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=Q3_ZORDERS["background"] + 1)

    ax_discharge = ax_cost.twinx()
    ax_discharge.plot(
        x,
        data["peak_discharge_energy"],
        color=Q3_COLORS["cost"],
        marker="s",
        linewidth=2.0,
        label="峰时段放电量",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax_discharge.set_ylabel("峰时段放电量（kWh）")
    _make_twin_axis_transparent(ax_discharge)

    handles_1, labels_1 = ax_cost.get_legend_handles_labels()
    handles_2, labels_2 = ax_discharge.get_legend_handles_labels()
    legend = ax_discharge.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc=Q3_LEGEND_LOCATIONS["reserve_soc_comparison"],
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def plot_schedule_profile(result: Question3Result, output_dir: str | Path) -> Path:
    """绘制 GPU 负载与传输速率的逐小时调度。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "schedule_profile.png"
    data = generate_question3_hourly_data(result)
    hours = data["hours"]

    fig, ax_load = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    tariff_handles = _shade_tariff_background(ax_load, data["prices"])
    ax_load.plot(
        hours,
        data["loads"],
        color=Q3_COLORS["load"],
        marker="o",
        linewidth=2.4,
        label="GPU负载",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax_load.set_ylabel("GPU负载（%）")
    ax_load.set_ylim(58, 102)
    ax_load.set_title("问题三逐小时调度策略")
    ax_load.grid(axis="y", color="#FFFFFF", linewidth=0.8, alpha=0.65, zorder=Q3_ZORDERS["background"] + 1)
    _set_hour_axis(ax_load)

    ax_rate = ax_load.twinx()
    ax_rate.plot(
        hours,
        data["rates"],
        color=Q3_COLORS["rate"],
        marker="^",
        linewidth=2.0,
        linestyle="--",
        label="数据传输速率",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax_rate.set_ylabel("数据传输速率（Mbps）")
    ax_rate.set_ylim(780, 1220)
    _make_twin_axis_transparent(ax_rate)
    _draw_tariff_boundaries(ax_load, data["prices"])

    handles_1, labels_1 = ax_load.get_legend_handles_labels()
    handles_2, labels_2 = ax_rate.get_legend_handles_labels()
    legend = ax_load.legend(
        tariff_handles + handles_1 + handles_2,
        [item.get_label() for item in tariff_handles] + labels_1 + labels_2,
        loc="lower left",
        ncol=2,
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def plot_battery_schedule(result: Question3Result, output_dir: str | Path) -> Path:
    """绘制电池充放电功率和 SOC 轨迹。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "battery_schedule.png"
    data = generate_question3_hourly_data(result)
    hours = data["hours"]

    fig, ax_power = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    tariff_handles = _shade_tariff_background(ax_power, data["prices"])
    ax_power.bar(
        hours,
        data["charge_powers"],
        width=0.72,
        color=Q3_COLORS["charge"],
        edgecolor="#FFFFFF",
        linewidth=0.5,
        label="充电功率",
        zorder=Q3_ZORDERS["bars"],
    )
    ax_power.bar(
        hours,
        -data["discharge_powers"],
        width=0.72,
        color=Q3_COLORS["discharge"],
        edgecolor="#FFFFFF",
        linewidth=0.5,
        label="放电功率",
        zorder=Q3_ZORDERS["bars"],
    )
    ax_power.axhline(0.0, color="#333333", linewidth=0.9, zorder=Q3_ZORDERS["line_plots"])
    ax_power.set_ylabel("充放电功率（kW）")
    ax_power.set_title("电池储能充放电与荷电状态变化")
    ax_power.grid(axis="y", color="#FFFFFF", linewidth=0.8, alpha=0.65, zorder=Q3_ZORDERS["background"] + 1)
    _set_hour_axis(ax_power)

    ax_soc = ax_power.twinx()
    ax_soc.plot(
        hours,
        data["soc_end"],
        color=Q3_COLORS["soc"],
        marker="o",
        linewidth=2.4,
        label="小时末SOC",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax_soc.axhline(DEFAULT_PARAMS.battery_min_soc_kwh, color=Q3_COLORS["discharge"], linestyle=":", linewidth=1.8)
    ax_soc.axhline(result.reserve_soc_kwh, color=Q3_COLORS["load"], linestyle="--", linewidth=1.8)
    ax_soc.set_ylabel("电池电量（kWh）")
    ax_soc.set_ylim(6, DEFAULT_PARAMS.battery_capacity_kwh + 2)
    _make_twin_axis_transparent(ax_soc)
    _draw_tariff_boundaries(ax_power, data["prices"])

    handles_1, labels_1 = ax_power.get_legend_handles_labels()
    handles_2, labels_2 = ax_soc.get_legend_handles_labels()
    extra_handles = [
        Line2D([0], [0], color=Q3_COLORS["discharge"], linestyle=":", linewidth=1.8, label="最低SOC"),
        Line2D([0], [0], color=Q3_COLORS["load"], linestyle="--", linewidth=1.8, label="循环保留电量"),
    ]
    legend = ax_soc.legend(
        tariff_handles + handles_1 + handles_2 + extra_handles,
        [item.get_label() for item in tariff_handles] + labels_1 + labels_2 + [item.get_label() for item in extra_handles],
        loc=Q3_LEGEND_LOCATIONS["battery_schedule"],
        ncol=3,
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def plot_cooling_inertia(result: Question3Result, output_dir: str | Path) -> Path:
    """绘制冷却稳态需求和惯性约束下的实际冷却功率。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "cooling_inertia.png"
    data = generate_question3_hourly_data(result)
    hours = data["hours"]

    fig, ax = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    tariff_handles = _shade_tariff_background(ax, data["prices"])
    ax.plot(
        hours,
        data["cooling_steady"],
        color=Q3_COLORS["cooling_steady"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        label="稳态冷却需求",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax.plot(
        hours,
        data["cooling_powers"],
        color=Q3_COLORS["cooling_actual"],
        marker="o",
        linewidth=2.5,
        label="实际冷却功率",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax.fill_between(
        hours,
        data["cooling_steady"],
        data["cooling_powers"],
        color=Q3_COLORS["flat"],
        alpha=0.36,
        label="惯性预冷差值",
        zorder=Q3_ZORDERS["bars"],
    )
    ax.set_ylabel("冷却功率（kW）")
    ax.set_title("冷却系统惯性约束下的功率平滑")
    ax.grid(axis="y", color="#FFFFFF", linewidth=0.8, alpha=0.65, zorder=Q3_ZORDERS["background"] + 1)
    _set_hour_axis(ax)
    _draw_tariff_boundaries(ax, data["prices"])

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        tariff_handles + handles,
        [item.get_label() for item in tariff_handles] + labels,
        loc=Q3_LEGEND_LOCATIONS["cooling_inertia"],
        ncol=2,
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def plot_grid_power_cost(result: Question3Result, output_dir: str | Path) -> Path:
    """绘制系统功率、电网购电功率和小时电费。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "grid_power_cost.png"
    data = generate_question3_hourly_data(result)
    hours = data["hours"]

    fig, ax_power = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    ax_cost = ax_power.twinx()
    ax_cost.set_zorder(1)
    ax_power.set_zorder(2)
    ax_power.patch.set_visible(False)
    _make_twin_axis_transparent(ax_cost)

    tariff_handles = _shade_tariff_background(ax_cost, data["prices"])
    ax_cost.bar(
        hours,
        data["hourly_costs"],
        width=0.62,
        color=Q3_COLORS["cost"],
        alpha=0.70,
        label="小时电费",
        zorder=Q3_ZORDERS["bars"],
    )
    ax_cost.set_ylabel("小时电费（元）")
    _draw_tariff_boundaries(ax_cost, data["prices"])

    ax_power.plot(
        hours,
        data["system_powers"],
        color=Q3_COLORS["system"],
        marker="o",
        linewidth=2.7,
        label="系统功率",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax_power.plot(
        hours,
        data["grid_powers"],
        color=Q3_COLORS["grid"],
        marker="^",
        linewidth=2.2,
        linestyle="--",
        label="电网购电功率",
        zorder=Q3_ZORDERS["line_plots"],
    )
    ax_power.set_ylabel("功率（kW）")
    ax_power.set_title("储能作用下的购电功率与小时电费")
    ax_power.grid(axis="y", color="#FFFFFF", linewidth=0.8, alpha=0.65, zorder=Q3_ZORDERS["background"] + 1)
    _set_hour_axis(ax_power)

    handles_1, labels_1 = ax_power.get_legend_handles_labels()
    handles_2, labels_2 = ax_cost.get_legend_handles_labels()
    legend = ax_power.legend(
        tariff_handles + handles_1 + handles_2,
        [item.get_label() for item in tariff_handles] + labels_1 + labels_2,
        loc=Q3_LEGEND_LOCATIONS["grid_power_cost"],
        ncol=3,
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def plot_q2_q3_comparison(
    result: Question3Result,
    output_dir: str | Path,
    *,
    q2_result: Question2Result | None = None,
) -> Path:
    """绘制问题二与问题三核心指标对比。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "q2_q3_comparison.png"
    q2_result = q2_result if q2_result is not None else solve_question2()

    left_labels = ["日电费", "购电量", "系统能耗"]
    q2_left = np.array([q2_result.total_cost, q2_result.total_energy, q2_result.total_energy], dtype=float)
    q3_left = np.array([result.total_cost, result.total_grid_energy, result.total_system_energy], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    bar_width = 0.36

    x_left = np.arange(len(left_labels))
    ax.bar(
        x_left - bar_width / 2,
        q2_left,
        width=bar_width,
        color=Q3_COLORS["q2"],
        label="问题二",
        zorder=Q3_ZORDERS["bars"],
    )
    ax.bar(
        x_left + bar_width / 2,
        q3_left,
        width=bar_width,
        color=Q3_COLORS["q3"],
        label="问题三",
        zorder=Q3_ZORDERS["bars"],
    )
    ax.set_xticks(x_left)
    ax.set_xticklabels(left_labels)
    ax.set_ylabel("金额或电量")
    ax.set_title("问题二与问题三费用与能耗对比")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=Q3_ZORDERS["background"] + 1)

    legend = ax.legend(loc=Q3_LEGEND_LOCATIONS["q2_q3_comparison"])
    _style_legend(legend)

    return _save_and_close(fig, output_path)


def generate_all_question3_plots(
    *,
    results: list[Question3Result] | None = None,
    recommended: Question3Result | None = None,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "question3",
    q2_result: Question2Result | None = None,
) -> list[Path]:
    """生成问题三全部 6 张图像。"""

    if results is None:
        results = solve_question3_reserve_sweep()
    if recommended is None:
        recommended = select_recommended_result(results)

    output_path = ensure_output_dir(output_dir)
    return [
        plot_reserve_soc_comparison(results, recommended, output_path),
        plot_schedule_profile(recommended, output_path),
        plot_battery_schedule(recommended, output_path),
        plot_cooling_inertia(recommended, output_path),
        plot_grid_power_cost(recommended, output_path),
        plot_q2_q3_comparison(recommended, output_path, q2_result=q2_result),
    ]


def main() -> None:
    """命令行入口：求解第三问并生成全部图像。"""

    paths = generate_all_question3_plots()
    print("已生成问题三图像：")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
