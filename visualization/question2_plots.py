"""问题二可视化图像生成。

本模块只负责绘图，不修改问题二求解逻辑。
所有图中文字均使用中文，图像统一保存到 outputs/question2/。
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
#     python visualization/question2_plots.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.components import transmission_power
from models.objectives import daily_energy, system_power
from models.parameters import DEFAULT_PARAMS, ModelParameters, price_schedule
from models.task import analysis_error, hourly_work, total_work, weighted_average_error
from question.question2 import Question2Result, solve_question2
from visualization.style import DEFAULT_STYLE, PAPER_PALETTE, QUESTION1_COLORS, apply_chinese_style


def ensure_output_dir(output_dir: str | Path) -> Path:
    """确保输出目录存在，并返回 Path 对象。"""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, path: Path) -> Path:
    """保存图像并关闭画布。"""

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _tariff_color(price: float) -> str:
    """按电价返回背景色。

    问题二的多张图都需要靠背景色区分峰、平、谷时段。
    这里使用比问题一更深的色值，避免时段区域在论文图中不明显。
    """

    if np.isclose(price, 2.0):
        return PAPER_PALETTE[9]
    if np.isclose(price, 1.2):
        return PAPER_PALETTE[2]
    return PAPER_PALETTE[3]


TARIFF_SHADE_ALPHA = 0.56
TARIFF_BOUNDARY_COLOR = PAPER_PALETTE[0]
TARIFF_BOUNDARY_HALO = "#FFFFFF"


def _tariff_periods(prices: np.ndarray) -> list[tuple[int, int, float]]:
    """把逐小时电价合并为连续时段。

    绘图时若逐小时绘制背景色，小时网格和色块边界会互相干扰。
    先合并为连续峰、平、谷时段，再绘制整段背景，时段结构更清楚。
    """

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
            zorder=0,
        )

    return [
        Patch(facecolor=PAPER_PALETTE[9], alpha=TARIFF_SHADE_ALPHA, label="峰时段"),
        Patch(facecolor=PAPER_PALETTE[2], alpha=TARIFF_SHADE_ALPHA, label="平时段"),
        Patch(facecolor=PAPER_PALETTE[3], alpha=TARIFF_SHADE_ALPHA, label="谷时段"),
    ]


def _make_twin_axis_transparent(ax: plt.Axes) -> None:
    """双 Y 轴默认覆盖在底图上，透明化后背景色带不会被遮住。"""

    ax.patch.set_visible(False)


def _draw_tariff_boundaries(ax: plt.Axes, prices: np.ndarray) -> None:
    """在峰、平、谷切换处绘制醒目的双层时段分界线。"""

    prices = np.asarray(prices, dtype=float)
    for hour_index in range(2, prices.size + 1):
        if np.isclose(prices[hour_index - 1], prices[hour_index - 2]):
            continue

        boundary_x = hour_index - 0.5
        # 先画白色粗底线，再叠加深色虚线。浅色、深色背景上都能看清。
        ax.axvline(
            boundary_x,
            color=TARIFF_BOUNDARY_HALO,
            linewidth=5.0,
            alpha=0.98,
            zorder=18,
        )
        ax.axvline(
            boundary_x,
            color=TARIFF_BOUNDARY_COLOR,
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            alpha=0.98,
            zorder=19,
        )


def generate_question2_hourly_data(
    result: Question2Result,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, np.ndarray | float]:
    """生成问题二 24 小时绘图数据。"""

    loads = np.asarray(result.loads, dtype=float)
    rates = np.asarray(result.rates, dtype=float)
    prices = np.asarray(result.prices, dtype=float)
    works = np.asarray(hourly_work(loads, rates, params), dtype=float)
    instant_errors = np.asarray(analysis_error(loads, rates, params=params), dtype=float)
    powers = np.asarray(system_power(loads, rates, params=params), dtype=float)

    return {
        "hours": np.arange(1, params.hours + 1),
        "loads": loads,
        "rates": rates,
        "prices": prices,
        "powers": powers,
        "hourly_costs": prices * powers,
        "works": works,
        "instant_errors": instant_errors,
        "weighted_error": weighted_average_error(loads, rates, params),
        "total_work": total_work(loads, rates, params),
    }


def generate_rate_reason_data(
    *,
    load_levels: tuple[float, ...] = (60.0, 78.642081, 100.0),
    num_rates: int = 81,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, object]:
    """生成解释传输速率取上限的敏感性数据。"""

    rates = np.linspace(params.rate_min_mbps, params.rate_max_mbps, num_rates)
    return {
        "rates": rates,
        "transmission_power": transmission_power(rates, params),
        "errors_by_load": {
            float(load): analysis_error(float(load), rates, params=params)
            for load in load_levels
        },
        "works_by_load": {
            float(load): hourly_work(float(load), rates, params)
            for load in load_levels
        },
    }


def _baseline_question1_metrics(params: ModelParameters = DEFAULT_PARAMS) -> dict[str, float]:
    """计算问题一固定方案在问题二电价下的对比指标。"""

    loads = np.full(params.hours, 85.0, dtype=float)
    rates = np.full(params.hours, params.rate_max_mbps, dtype=float)
    prices = price_schedule(params)
    powers = np.asarray(system_power(loads, rates, params=params), dtype=float)
    return {
        "cost": float(np.sum(prices * powers)),
        "energy": daily_energy(loads, rates, params=params),
        "work": total_work(loads, rates, params),
        "error": weighted_average_error(loads, rates, params),
    }


def plot_tariff_schedule(
    result: Question2Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制 24 小时电价与调度曲线。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    data = generate_question2_hourly_data(result, params=params)

    fig, ax_load = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    tariff_handles = _shade_tariff_background(ax_load, data["prices"])
    ax_rate = ax_load.twinx()
    _make_twin_axis_transparent(ax_rate)

    load_line = ax_load.step(
        data["hours"],
        data["loads"],
        where="mid",
        color=QUESTION1_COLORS["optimal"],
        linewidth=2.6,
        label="GPU负载",
        zorder=3,
    )[0]
    rate_line = ax_rate.step(
        data["hours"],
        data["rates"],
        where="mid",
        color=QUESTION1_COLORS["text_blue"],
        linewidth=2.2,
        linestyle="--",
        label="数据传输速率",
        zorder=3,
    )[0]

    ax_load.set_title("问题二分时电价调度结果")
    ax_load.set_xlabel("小时")
    ax_load.set_ylabel("GPU负载（%）")
    ax_rate.set_ylabel("数据传输速率（Mbps）")
    ax_load.set_xlim(0.5, params.hours + 0.5)
    ax_load.set_xticks(np.arange(1, params.hours + 1, 1))
    ax_load.set_ylim(55, 105)
    ax_rate.set_ylim(760, 1240)
    ax_load.grid(alpha=0.25, axis="y")
    _draw_tariff_boundaries(ax_rate, data["prices"])
    ax_load.legend(
        [load_line, rate_line] + tariff_handles,
        ["GPU负载", "数据传输速率", "峰时段", "平时段", "谷时段"],
        loc="lower left",
        ncol=3,
    )
    return _save_and_close(fig, output_dir / "tariff_schedule.png")


def plot_power_cost_schedule(
    result: Question2Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制 24 小时系统功率与小时电费。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    data = generate_question2_hourly_data(result, params=params)

    fig, ax_power = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    tariff_handles = _shade_tariff_background(ax_power, data["prices"])
    ax_cost = ax_power.twinx()
    _make_twin_axis_transparent(ax_cost)

    cost_bars = ax_cost.bar(
        data["hours"],
        data["hourly_costs"],
        color=PAPER_PALETTE[8],
        alpha=0.9,
        width=0.62,
        edgecolor="white",
        linewidth=0.6,
        label="小时电费",
        zorder=2,
    )
    power_line = ax_power.step(
        data["hours"],
        data["powers"],
        where="mid",
        color=QUESTION1_COLORS["total"],
        linewidth=2.5,
        label="系统功率",
        zorder=4,
    )[0]
    # twin y 轴按“整层坐标轴”绘制，右轴柱状图可能压住左轴线条。
    # 这里在右轴最顶层复制一条系统功率线，但沿用左轴数据坐标，
    # 因此视觉层级最高，数值位置仍对应左侧“系统功率”坐标轴。
    power_line.set_visible(False)
    power_line_top = Line2D(
        data["hours"],
        data["powers"],
        color=QUESTION1_COLORS["total"],
        linewidth=2.8,
        drawstyle="steps-mid",
        label="系统功率",
        transform=ax_power.transData,
        zorder=30,
        clip_on=False,
    )
    ax_cost.add_artist(power_line_top)

    ax_power.set_title("问题二系统功率与小时电费")
    ax_power.set_xlabel("小时")
    ax_power.set_ylabel("系统功率（kW）")
    ax_cost.set_ylabel("小时电费（元）")
    ax_power.set_xlim(0.5, params.hours + 0.5)
    ax_power.set_xticks(np.arange(1, params.hours + 1, 1))
    ax_power.grid(alpha=0.25, axis="y")
    _draw_tariff_boundaries(ax_cost, data["prices"])
    # 图中柱状图属于右轴 ax_cost。图例必须挂在右轴上，
    # 否则会被 twin y 轴覆盖，看起来像“叠在最底层”。
    legend = ax_cost.legend(
        [power_line_top, cost_bars] + tariff_handles,
        ["系统功率", "小时电费", "峰时段", "平时段", "谷时段"],
        loc="lower left",
        ncol=3,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#CCCCCC",
    )
    legend.set_zorder(100)
    return _save_and_close(fig, output_dir / "power_cost_schedule.png")


def plot_work_error_contribution(
    result: Question2Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制处理量与瞬时误差贡献图。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    data = generate_question2_hourly_data(result, params=params)

    fig, ax_work = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    tariff_handles = _shade_tariff_background(ax_work, data["prices"])
    ax_error = ax_work.twinx()
    _make_twin_axis_transparent(ax_error)

    work_bars = ax_work.bar(
        data["hours"],
        data["works"],
        color=PAPER_PALETTE[1],
        alpha=0.9,
        width=0.62,
        edgecolor="white",
        linewidth=0.6,
        label="每小时处理量",
        zorder=2,
    )
    error_line = ax_error.step(
        data["hours"],
        data["instant_errors"],
        where="mid",
        color=QUESTION1_COLORS["error"],
        linewidth=2.5,
        label="瞬时误差率",
        zorder=4,
    )[0]
    limit_line = ax_error.axhline(
        params.error_limit_percent,
        color=PAPER_PALETTE[8],
        linestyle=":",
        linewidth=2.2,
        label="误差5%线",
    )
    ax_error.annotate(
        f"加权平均误差={data['weighted_error']:.2f}%",
        xy=(18.5, params.error_limit_percent),
        xytext=(12.6, params.error_limit_percent + 3.8),
        arrowprops={"arrowstyle": "->", "color": QUESTION1_COLORS["error"]},
        color=QUESTION1_COLORS["error"],
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )

    ax_work.set_title("问题二处理量与误差贡献")
    ax_work.set_xlabel("小时")
    ax_work.set_ylabel("每小时处理量")
    ax_error.set_ylabel("瞬时误差率（%）")
    ax_work.set_xlim(0.5, params.hours + 0.5)
    ax_work.set_xticks(np.arange(1, params.hours + 1, 1))
    ax_work.grid(alpha=0.25, axis="y")
    _draw_tariff_boundaries(ax_error, data["prices"])
    ax_work.legend(
        [work_bars, error_line, limit_line] + tariff_handles,
        ["每小时处理量", "瞬时误差率", "误差5%线", "峰时段", "平时段", "谷时段"],
        loc="upper left",
        ncol=3,
    )
    return _save_and_close(fig, output_dir / "work_error_contribution.png")


def plot_q1_q2_comparison(
    result: Question2Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制问题一固定方案与问题二动态方案对比图。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    baseline = _baseline_question1_metrics(params)
    metrics = [
        ("日电费（元）", baseline["cost"], result.total_cost),
        ("日能耗（kWh）", baseline["energy"], result.total_energy),
        ("总处理量", baseline["work"], result.total_work),
        ("加权误差（%）", baseline["error"], result.weighted_error),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.ravel()
    for ax, (title, q1_value, q2_value) in zip(axes, metrics):
        bars = ax.bar(
            ["问题一固定方案", "问题二动态方案"],
            [q1_value, q2_value],
            color=[PAPER_PALETTE[2], QUESTION1_COLORS["optimal"]],
            alpha=0.86,
            width=0.55,
        )
        ax.set_title(title)
        ax.grid(alpha=0.22, axis="y")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    saving_rate = 100 * result.cost_saving_rate
    axes[0].annotate(
        f"节省{result.cost_saving:.3f}元\n降低{saving_rate:.2f}%",
        xy=(1, result.total_cost),
        xytext=(0.25, max(baseline["cost"], result.total_cost) * 0.82),
        arrowprops={"arrowstyle": "->", "color": QUESTION1_COLORS["optimal"]},
        color=QUESTION1_COLORS["optimal"],
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.suptitle("问题一固定方案与问题二动态方案对比", fontsize=14)
    return _save_and_close(fig, output_dir / "q1_q2_comparison.png")


def plot_rate_max_reason(
    result: Question2Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制传输速率取上限原因图。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    load_levels = (result.peak_load, result.flat_load, result.valley_load)
    data = generate_rate_reason_data(load_levels=load_levels, params=params)

    fig, ax_power = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    ax_error = ax_power.twinx()

    power_line = ax_power.plot(
        data["rates"],
        data["transmission_power"],
        color=QUESTION1_COLORS["text_blue"],
        linewidth=2.6,
        label="传输功率",
    )[0]

    error_lines = []
    line_colors = [PAPER_PALETTE[8], QUESTION1_COLORS["optimal"], PAPER_PALETTE[0]]
    for color, (load, errors) in zip(line_colors, data["errors_by_load"].items()):
        line = ax_error.plot(
            data["rates"],
            errors,
            color=color,
            linewidth=2.2,
            label=f"GPU负载{load:.1f}%时误差",
        )[0]
        error_lines.append(line)

    max_rate_line = ax_power.axvline(
        params.rate_max_mbps,
        color=QUESTION1_COLORS["optimal"],
        linestyle=":",
        linewidth=2.2,
        label="最高传输速率",
    )
    ax_power.annotate(
        "传输功率仅增加0.16kW",
        xy=(params.rate_max_mbps, transmission_power(params.rate_max_mbps, params)),
        xytext=(1010, 0.39),
        arrowprops={"arrowstyle": "->", "color": QUESTION1_COLORS["text_blue"]},
        color=QUESTION1_COLORS["text_blue"],
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax_error.annotate(
        "误差率最多下降20个百分点",
        xy=(params.rate_max_mbps, analysis_error(result.flat_load, params.rate_max_mbps, params=params)),
        xytext=(850, 14.0),
        arrowprops={"arrowstyle": "->", "color": QUESTION1_COLORS["error"]},
        color=QUESTION1_COLORS["error"],
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax_power.set_title("数据传输速率取上限的原因")
    ax_power.set_xlabel("数据传输速率（Mbps）")
    ax_power.set_ylabel("传输功率（kW）")
    ax_error.set_ylabel("误差率（%）")
    ax_power.set_xlim(params.rate_min_mbps, params.rate_max_mbps + 8)
    ax_power.grid(alpha=0.25)
    ax_power.legend(
        [power_line, max_rate_line] + error_lines,
        ["传输功率", "最高传输速率"] + [line.get_label() for line in error_lines],
        loc="upper right",
        ncol=2,
    )
    return _save_and_close(fig, output_dir / "rate_max_reason.png")


def generate_all_question2_plots(
    *,
    result: Question2Result | None = None,
    output_dir: str | Path = "outputs/question2",
    params: ModelParameters = DEFAULT_PARAMS,
) -> list[Path]:
    """生成问题二全部图像。"""

    result = solve_question2(params) if result is None else result
    output_dir = ensure_output_dir(output_dir)
    return [
        plot_tariff_schedule(result, output_dir, params=params),
        plot_power_cost_schedule(result, output_dir, params=params),
        plot_work_error_contribution(result, output_dir, params=params),
        plot_q1_q2_comparison(result, output_dir, params=params),
        plot_rate_max_reason(result, output_dir, params=params),
    ]


def main() -> None:
    """命令行入口：python -m visualization.question2_plots。"""

    paths = generate_all_question2_plots()
    print("已生成问题二图像：")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
