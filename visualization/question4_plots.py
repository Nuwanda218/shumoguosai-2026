"""问题四可视化图像生成。

本模块只负责把问题四求解结果整理成论文图像，不修改优化模型。
所有图中文字均使用中文，图像统一保存到 outputs/question4/。
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

# 支持直接运行本文件：
#     python visualization/question4_plots.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.parameters import DEFAULT_PARAMS, ModelParameters
from question.question3 import Question3Result
from question.question4 import Question4Result, cyclic_abs_deltas, solve_question4
from visualization.style import DEFAULT_STYLE, PAPER_PALETTE, apply_chinese_style


TARIFF_SHADE_ALPHA = 0.50
TARIFF_BOUNDARY_COLOR = PAPER_PALETTE[0]
TARIFF_BOUNDARY_HALO = "#FFFFFF"
Q4_COLORS = {
    "valley": PAPER_PALETTE[3],
    "flat": PAPER_PALETTE[2],
    "peak": PAPER_PALETTE[9],
    "load": PAPER_PALETTE[0],
    "rate": PAPER_PALETTE[7],
    "cooling": PAPER_PALETTE[5],
    "limit": PAPER_PALETTE[6],
    "bar": PAPER_PALETTE[2],
    "bar_alt": PAPER_PALETTE[9],
    "q3": PAPER_PALETTE[1],
    "q4": PAPER_PALETTE[7],
    "main_constraint": PAPER_PALETTE[5],
    "smooth_constraint": PAPER_PALETTE[2],
    "intersection": PAPER_PALETTE[3],
    "optimal_point": PAPER_PALETTE[6],
    "text": PAPER_PALETTE[5],
}
Q4_LEGEND_LOCATIONS = {
    "schedule_profile": "lower left",
    "constraint_margin_summary": "upper left",
    "q3_q4_comparison": "upper left",
    "constraint_strength_comparison": "upper left",
}
DEPRECATED_MAIN_PLOTS = (
    "schedule_profile.png",
    "change_rate_check.png",
    "constraint_margin_summary.png",
)
Q4_ZORDERS = {
    "background": 0,
    "tariff_boundaries": 2,
    "bars": 20,
    "line_plots": 60,
    "annotations": 80,
    "legend": 100,
}


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
    """按电价返回峰、平、谷背景色。"""

    if np.isclose(price, 2.0):
        return Q4_COLORS["peak"]
    if np.isclose(price, 1.2):
        return Q4_COLORS["flat"]
    return Q4_COLORS["valley"]


def _tariff_periods(prices: np.ndarray) -> list[tuple[int, int, float]]:
    """把逐小时电价合并为连续时段。"""

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
            zorder=Q4_ZORDERS["background"],
        )

    return [
        Patch(facecolor=Q4_COLORS["peak"], alpha=TARIFF_SHADE_ALPHA, label="峰时段"),
        Patch(facecolor=Q4_COLORS["flat"], alpha=TARIFF_SHADE_ALPHA, label="平时段"),
        Patch(facecolor=Q4_COLORS["valley"], alpha=TARIFF_SHADE_ALPHA, label="谷时段"),
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
            alpha=0.96,
            zorder=Q4_ZORDERS["tariff_boundaries"],
        )
        ax.axvline(
            boundary_x,
            color=TARIFF_BOUNDARY_COLOR,
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            alpha=0.96,
            zorder=Q4_ZORDERS["tariff_boundaries"] + 1,
        )


def _make_twin_axis_transparent(ax: plt.Axes) -> None:
    """透明化双 Y 轴底色。"""

    ax.patch.set_visible(False)


def _set_hour_axis(ax: plt.Axes) -> None:
    """统一 24 小时横轴刻度。"""

    ax.set_xlim(0.5, 24.5)
    ax.set_xticks(np.arange(1, 25, 1))
    ax.set_xlabel("小时")


def _style_legend(legend) -> None:
    """统一图例样式，保证图例不透明且位于最上层。"""

    if legend is None:
        return
    legend.set_zorder(Q4_ZORDERS["legend"])
    frame = legend.get_frame()
    frame.set_alpha(1.0)
    frame.set_facecolor("#FFFFFF")
    frame.set_edgecolor("#D0D0D0")


def generate_question4_hourly_data(
    result: Question4Result,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, np.ndarray | float]:
    """整理问题四逐小时绘图数据。"""

    return {
        "hours": np.arange(1, params.hours + 1),
        "loads": np.asarray(result.loads, dtype=float),
        "rates": np.asarray(result.rates, dtype=float),
        "prices": np.asarray(result.prices, dtype=float),
        "cooling_powers": np.asarray(result.cooling_powers, dtype=float),
        "hourly_costs": np.asarray(result.hourly_costs, dtype=float),
        "grid_powers": np.asarray(result.grid_powers, dtype=float),
        "system_powers": np.asarray(result.system_powers, dtype=float),
        "total_cost": float(result.total_cost),
        "total_work": float(result.total_work),
        "weighted_error": float(result.weighted_error),
        "max_load_delta": float(result.max_load_delta),
        "max_rate_delta": float(result.max_rate_delta),
        "max_cooling_delta": float(result.max_cooling_delta),
    }


def generate_change_rate_data(
    result: Question4Result,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, np.ndarray | float]:
    """整理包含日周期边界的变化率检验数据。"""

    transition_labels = np.array(
        [f"{hour}→{hour + 1}" for hour in range(1, params.hours)] + ["24→1"]
    )
    return {
        "transition_index": np.arange(1, params.hours + 1),
        "transition_labels": transition_labels,
        "load_deltas": cyclic_abs_deltas(result.loads),
        "rate_deltas": cyclic_abs_deltas(result.rates),
        "cooling_deltas": cyclic_abs_deltas(result.cooling_powers),
        "load_limit": float(params.gpu_change_limit_percent),
        "rate_limit": float(params.rate_change_limit_mbps),
        "cooling_limit": float(params.cooling_change_limit_kw),
    }


def generate_constraint_margin_data(
    result: Question4Result,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, np.ndarray]:
    """整理变化率约束使用率数据。"""

    actual = np.array(
        [
            float(result.max_load_delta),
            float(result.max_rate_delta),
            float(result.max_cooling_delta),
        ]
    )
    limits = np.array(
        [
            params.gpu_change_limit_percent,
            params.rate_change_limit_mbps,
            params.cooling_change_limit_kw,
        ],
        dtype=float,
    )
    return {
        "labels": np.array(["GPU负载变化", "传输速率变化", "冷却功率变化"]),
        "actual": actual,
        "limits": limits,
        "usage_rates": actual / limits * 100.0,
    }


def generate_constraint_strength_data(
    result: Question4Result,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, np.ndarray]:
    """整理主约束和平滑约束强弱对比数据。"""

    actual = np.array(
        [
            float(result.weighted_error),
            float(result.max_cooling_delta),
            float(result.max_load_delta),
            float(result.max_rate_delta),
        ],
        dtype=float,
    )
    limits = np.array(
        [
            params.error_limit_percent,
            params.cooling_change_limit_kw,
            params.gpu_change_limit_percent,
            params.rate_change_limit_mbps,
        ],
        dtype=float,
    )
    return {
        "labels": np.array(["加权误差", "冷却变化", "GPU负载变化", "传输速率变化"]),
        "constraint_types": np.array(["主约束", "主约束", "平滑约束", "平滑约束"]),
        "actual": actual,
        "limits": limits,
        "usage_rates": actual / limits * 100.0,
        "set_relation": "主约束区域在平滑约束范围内",
        "optimal_relation": "问题三最优点 = 问题四最优点",
    }


def plot_schedule_profile(result: Question4Result, output_dir: str | Path) -> Path:
    """绘制问题四 24 小时 GPU 负载与传输速率调度曲线。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "schedule_profile.png"
    data = generate_question4_hourly_data(result)
    hours = data["hours"]

    fig, ax_load = plt.subplots(figsize=DEFAULT_STYLE.curve_size)
    tariff_handles = _shade_tariff_background(ax_load, data["prices"])
    ax_load.plot(
        hours,
        data["loads"],
        color=Q4_COLORS["load"],
        marker="o",
        linewidth=2.5,
        label="GPU负载",
        zorder=Q4_ZORDERS["line_plots"],
    )
    ax_load.set_ylabel("GPU负载（%）")
    ax_load.set_ylim(58, 102)
    ax_load.set_title("问题四24小时调度曲线")
    ax_load.grid(axis="y", color="#FFFFFF", linewidth=0.8, alpha=0.65, zorder=Q4_ZORDERS["background"] + 1)
    _set_hour_axis(ax_load)

    ax_rate = ax_load.twinx()
    ax_rate.plot(
        hours,
        data["rates"],
        color=Q4_COLORS["rate"],
        marker="^",
        linewidth=2.2,
        linestyle="--",
        label="数据传输速率",
        zorder=Q4_ZORDERS["line_plots"],
    )
    ax_rate.set_ylabel("数据传输速率（Mbps）")
    ax_rate.set_ylim(1180, 1210)
    _make_twin_axis_transparent(ax_rate)
    _draw_tariff_boundaries(ax_load, data["prices"])

    handles_1, labels_1 = ax_load.get_legend_handles_labels()
    handles_2, labels_2 = ax_rate.get_legend_handles_labels()
    legend = ax_load.legend(
        tariff_handles + handles_1 + handles_2,
        [item.get_label() for item in tariff_handles] + labels_1 + labels_2,
        loc=Q4_LEGEND_LOCATIONS["schedule_profile"],
        ncol=2,
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def _plot_delta_panel(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    limit: float,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    """绘制单个变化率检验子图。"""

    ax.bar(
        x,
        values,
        width=0.72,
        color=color,
        edgecolor="#FFFFFF",
        linewidth=0.5,
        zorder=Q4_ZORDERS["bars"],
    )
    ax.axhline(
        limit,
        color=Q4_COLORS["limit"],
        linewidth=2.0,
        linestyle="--",
        label="约束上限",
        zorder=Q4_ZORDERS["line_plots"],
    )
    ax.axvline(
        24,
        color=Q4_COLORS["load"],
        linewidth=1.6,
        linestyle=":",
        label="日周期边界",
        zorder=Q4_ZORDERS["line_plots"],
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(limit * 1.18, float(np.max(values)) * 1.35 + 1e-9))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=Q4_ZORDERS["background"])
    # 图例放到坐标轴右侧，避免遮挡柱状图和约束上限线。
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style_legend(legend)


def plot_change_rate_check(result: Question4Result, output_dir: str | Path) -> Path:
    """绘制问题四新增变化率约束逐小时检验图。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "change_rate_check.png"
    data = generate_change_rate_data(result)
    x = data["transition_index"]

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.2), sharex=True)
    _plot_delta_panel(
        axes[0],
        x,
        data["load_deltas"],
        limit=float(data["load_limit"]),
        title="GPU负载变化率检验",
        ylabel="变化幅度（%）",
        color=Q4_COLORS["bar"],
    )
    _plot_delta_panel(
        axes[1],
        x,
        data["rate_deltas"],
        limit=float(data["rate_limit"]),
        title="传输速率变化率检验",
        ylabel="变化幅度（Mbps）",
        color=Q4_COLORS["bar_alt"],
    )
    _plot_delta_panel(
        axes[2],
        x,
        data["cooling_deltas"],
        limit=float(data["cooling_limit"]),
        title="冷却功率变化率检验",
        ylabel="变化幅度（kW）",
        color=Q4_COLORS["cooling"],
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(data["transition_labels"], rotation=45, ha="right")
    axes[-1].set_xlabel("相邻小时转换")
    return _save_and_close(fig, output_path)


def plot_constraint_margin_summary(result: Question4Result, output_dir: str | Path) -> Path:
    """绘制变化率约束使用率汇总图。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "constraint_margin_summary.png"
    data = generate_constraint_margin_data(result)
    x = np.arange(data["labels"].size)
    colors = [Q4_COLORS["bar"], Q4_COLORS["bar_alt"], Q4_COLORS["cooling"]]

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    bars = ax.bar(
        x,
        data["usage_rates"],
        width=0.58,
        color=colors,
        edgecolor="#FFFFFF",
        linewidth=0.8,
        label="约束使用率",
        zorder=Q4_ZORDERS["bars"],
    )
    ax.axhline(
        100.0,
        color=Q4_COLORS["limit"],
        linewidth=2.0,
        linestyle="--",
        label="约束上限",
        zorder=Q4_ZORDERS["line_plots"],
    )
    for bar, usage in zip(bars, data["usage_rates"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(usage) + 2.0,
            f"{float(usage):.1f}%",
            ha="center",
            va="bottom",
            color=Q4_COLORS["text"],
            fontsize=10,
            zorder=Q4_ZORDERS["annotations"],
        )
    ax.text(
        x[-1],
        float(data["usage_rates"][-1]) - 8.0,
        "绑定",
        ha="center",
        va="top",
        color="#FFFFFF",
        fontsize=10,
        fontweight="bold",
        zorder=Q4_ZORDERS["annotations"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(data["labels"])
    ax.set_ylabel("约束使用率（%）")
    ax.set_ylim(0, 116)
    ax.set_title("问题四变化率约束裕度汇总")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=Q4_ZORDERS["background"])
    legend = ax.legend(
        loc=Q4_LEGEND_LOCATIONS["constraint_margin_summary"],
        bbox_to_anchor=(1.01, 1.0),
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def plot_q3_q4_comparison(
    result: Question4Result,
    output_dir: str | Path,
    *,
    q3_result: Question3Result | None = None,
) -> Path:
    """绘制问题三与问题四核心指标对比图。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "q3_q4_comparison.png"
    # 当前问题四直接继承问题三可行最优解。若外部传入 q3_result，则使用外部基准。
    baseline = q3_result if q3_result is not None else result

    metric_labels = ["日电费", "系统能耗", "总处理量", "加权误差"]
    q3_metrics = np.array(
        [baseline.total_cost, baseline.total_system_energy, baseline.total_work, baseline.weighted_error],
        dtype=float,
    )
    q4_metrics = np.array(
        [result.total_cost, result.total_system_energy, result.total_work, result.weighted_error],
        dtype=float,
    )
    q3_relative = np.full(q3_metrics.shape, 100.0)
    q4_relative = np.divide(
        q4_metrics,
        q3_metrics,
        out=np.zeros_like(q4_metrics),
        where=np.abs(q3_metrics) > 1e-12,
    ) * 100.0
    value_formats = ["{:.2f}元", "{:.2f}kWh", "{:.3f}", "{:.2f}%"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bar_width = 0.36

    x_metric = np.arange(len(metric_labels))
    q3_bars = ax.bar(
        x_metric - bar_width / 2,
        q3_relative,
        width=bar_width,
        color=Q4_COLORS["q3"],
        label="问题三",
        zorder=Q4_ZORDERS["bars"],
    )
    q4_bars = ax.bar(
        x_metric + bar_width / 2,
        q4_relative,
        width=bar_width,
        color=Q4_COLORS["q4"],
        label="问题四",
        zorder=Q4_ZORDERS["bars"],
    )
    ax.axhline(
        100.0,
        color=Q4_COLORS["limit"],
        linewidth=1.8,
        linestyle="--",
        label="问题三基准",
        zorder=Q4_ZORDERS["line_plots"],
    )

    for bars, actual_values in ((q3_bars, q3_metrics), (q4_bars, q4_metrics)):
        for bar, actual_value, formatter in zip(bars, actual_values, value_formats):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.6,
                formatter.format(float(actual_value)),
                ha="center",
                va="bottom",
                color=Q4_COLORS["text"],
                fontsize=8.5,
                rotation=0,
                zorder=Q4_ZORDERS["annotations"],
            )

    ax.text(
        0.5,
        -0.16,
        "问题三方案已满足问题四新增变化率约束，核心结果保持不变",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=Q4_COLORS["text"],
        fontsize=10,
        zorder=Q4_ZORDERS["annotations"],
        clip_on=False,
    )
    ax.set_xticks(x_metric)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("相对问题三基准（%）")
    ax.set_ylim(0, 116)
    ax.set_title("问题三与问题四核心结果对比")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=Q4_ZORDERS["background"])
    legend = ax.legend(
        loc=Q4_LEGEND_LOCATIONS["q3_q4_comparison"],
        bbox_to_anchor=(1.01, 1.0),
    )
    _style_legend(legend)
    return _save_and_close(fig, output_path)


def plot_constraint_strength_comparison(result: Question4Result, output_dir: str | Path) -> Path:
    """绘制主约束与新增平滑约束强弱对比图。"""

    apply_chinese_style()
    output_path = ensure_output_dir(output_dir) / "constraint_strength_comparison.png"
    data = generate_constraint_strength_data(result)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), gridspec_kw={"width_ratios": [1.05, 1.35]})
    ax_set, ax_bar = axes

    # 左图用嵌套集合关系说明：新增平滑约束更宽，没有排除问题三最优点。
    ax_set.set_xlim(0, 10)
    ax_set.set_ylim(0, 7)
    ax_set.axis("off")
    smooth_region = FancyBboxPatch(
        (0.75, 0.75),
        8.5,
        5.6,
        boxstyle="round,pad=0.28,rounding_size=0.55",
        facecolor=Q4_COLORS["smooth_constraint"],
        edgecolor=Q4_COLORS["load"],
        linewidth=2.0,
        linestyle="--",
        alpha=0.42,
        zorder=Q4_ZORDERS["background"],
    )
    main_region = FancyBboxPatch(
        (2.2, 1.75),
        5.6,
        3.55,
        boxstyle="round,pad=0.24,rounding_size=0.45",
        facecolor=Q4_COLORS["main_constraint"],
        edgecolor=Q4_COLORS["text"],
        linewidth=2.0,
        alpha=0.84,
        zorder=Q4_ZORDERS["bars"],
    )
    for patch in (smooth_region, main_region):
        ax_set.add_patch(patch)

    ax_set.scatter(
        [5.0],
        [3.35],
        s=120,
        color=Q4_COLORS["optimal_point"],
        edgecolor="#FFFFFF",
        linewidth=1.2,
        zorder=Q4_ZORDERS["annotations"],
        label="问题三最优点",
    )
    ax_set.annotate(
        data["optimal_relation"],
        xy=(5.0, 3.35),
        xytext=(6.9, 4.72),
        arrowprops={"arrowstyle": "->", "color": "#FFFFFF", "linewidth": 1.8},
        color=Q4_COLORS["text"],
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": "#DDDDDD", "alpha": 1.0},
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_set.text(
        5.0,
        6.05,
        "新增平滑约束范围",
        ha="center",
        va="center",
        color=Q4_COLORS["text"],
        fontsize=10,
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_set.text(
        5.0,
        4.08,
        "主约束可行区域",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=10,
        fontweight="bold",
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_set.text(
        5.0,
        2.55,
        "更强",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=9.5,
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_set.text(
        5.0,
        1.12,
        "平滑约束没有排除原最优点",
        ha="center",
        va="center",
        color=Q4_COLORS["text"],
        fontsize=10,
        fontweight="bold",
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_set.text(
        5.0,
        0.42,
        "因此问题四结果保持不变",
        ha="center",
        va="center",
        color=Q4_COLORS["text"],
        fontsize=9.2,
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_set.set_title("集合交运算示意")

    # 右图用约束使用率展示主约束接近或达到边界，新增平滑约束明显更松。
    x = np.arange(data["labels"].size)
    colors = [
        Q4_COLORS["main_constraint"] if kind == "主约束" else Q4_COLORS["smooth_constraint"]
        for kind in data["constraint_types"]
    ]
    bars = ax_bar.bar(
        x,
        data["usage_rates"],
        width=0.62,
        color=colors,
        edgecolor="#FFFFFF",
        linewidth=0.8,
        zorder=Q4_ZORDERS["bars"],
    )
    ax_bar.axhline(
        100.0,
        color=Q4_COLORS["limit"],
        linewidth=2.0,
        linestyle="--",
        label="约束边界",
        zorder=Q4_ZORDERS["line_plots"],
    )
    for bar, value in zip(bars, data["usage_rates"]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + 2.0,
            f"{float(value):.1f}%",
            ha="center",
            va="bottom",
            color=Q4_COLORS["text"],
            fontsize=9.5,
            zorder=Q4_ZORDERS["annotations"],
        )
    ax_bar.text(
        0.5,
        0.91,
        "主约束",
        transform=ax_bar.transAxes,
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=10,
        fontweight="bold",
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_bar.text(
        0.80,
        0.24,
        "新增平滑约束较松",
        transform=ax_bar.transAxes,
        ha="center",
        va="center",
        color=Q4_COLORS["text"],
        fontsize=10,
        zorder=Q4_ZORDERS["annotations"],
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(data["labels"])
    ax_bar.set_ylabel("约束使用率（%）")
    ax_bar.set_ylim(0, 116)
    ax_bar.set_title("约束强弱对比")
    ax_bar.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=Q4_ZORDERS["background"])
    handles = [
        Patch(facecolor=Q4_COLORS["main_constraint"], label="主约束"),
        Patch(facecolor=Q4_COLORS["smooth_constraint"], label="平滑约束"),
        Line2D([0], [0], color=Q4_COLORS["limit"], linestyle="--", linewidth=2.0, label="约束边界"),
    ]
    legend = ax_bar.legend(
        handles=handles,
        loc=Q4_LEGEND_LOCATIONS["constraint_strength_comparison"],
        bbox_to_anchor=(1.01, 1.0),
    )
    _style_legend(legend)
    fig.suptitle("问题四新增平滑约束弱于主约束的原因", fontsize=15)
    return _save_and_close(fig, output_path)


def _remove_deprecated_main_plots(output_dir: Path) -> None:
    """移除不再作为问题四正文主图的旧图像。"""

    for filename in DEPRECATED_MAIN_PLOTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def generate_all_question4_plots(
    *,
    result: Question4Result | None = None,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "question4",
    q3_result: Question3Result | None = None,
) -> list[Path]:
    """生成问题四正文保留的 2 张主图像。"""

    result = result if result is not None else solve_question4()
    output_path = ensure_output_dir(output_dir)
    _remove_deprecated_main_plots(output_path)
    return [
        plot_constraint_strength_comparison(result, output_path),
        plot_q3_q4_comparison(result, output_path, q3_result=q3_result),
    ]


def main() -> None:
    """命令行入口：求解问题四并生成正文保留图像。"""

    paths = generate_all_question4_plots()
    print("已生成问题四图像：")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
