"""问题一可视化图像生成。

本模块只负责绘图，不修改问题一求解逻辑。
它可以接收 question.question1.solve_question1() 返回的结果，
也会直接复用 models/ 中的功耗、误差和任务量公式生成网格数据。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # required by Matplotlib 3D projection

from models.components import cooling_steady_power, gpu_cluster_power, transmission_power
from models.objectives import system_power
from models.parameters import DEFAULT_PARAMS, ModelParameters
from models.task import analysis_error, hourly_work
from question.question1 import Question1Result, solve_question1
from visualization.style import DEFAULT_STYLE, QUESTION1_COLORS, apply_chinese_style


def ensure_output_dir(output_dir: str | Path) -> Path:
    """确保输出目录存在，并返回 Path 对象。"""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_question1_grid(
    *,
    num_loads: int = 81,
    num_rates: int = 81,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, np.ndarray]:
    """生成问题一二维网格数据。

    返回的数据用于 3D 曲面图和 2D 可行域图。
    """

    load_values = np.linspace(params.gpu_min_load, params.gpu_max_load, num_loads)
    rate_values = np.linspace(params.rate_min_mbps, params.rate_max_mbps, num_rates)
    loads, rates = np.meshgrid(load_values, rate_values)

    errors = analysis_error(loads, rates, params=params)
    powers = system_power(loads, rates, params=params)
    work = hourly_work(loads, rates, params=params) * params.hours
    feasible = (work >= 1.0) & (errors <= params.error_limit_percent)

    return {
        "load_values": load_values,
        "rate_values": rate_values,
        "loads": loads,
        "rates": rates,
        "errors": errors,
        "powers": powers,
        "work": work,
        "feasible": feasible,
    }


def generate_rate_sensitivity_curves(
    *,
    load_levels: tuple[float, ...] = (70.0, 85.0, 100.0),
    num_rates: int = 81,
    params: ModelParameters = DEFAULT_PARAMS,
) -> dict[str, object]:
    """生成传输速率敏感性曲线数据。

    用于说明：当 GPU 负载固定时，提高传输速率会降低误差，
    但传输功率只缓慢增加。
    """

    rates = np.linspace(params.rate_min_mbps, params.rate_max_mbps, num_rates)
    errors_by_load = {
        float(load): analysis_error(float(load), rates, params=params)
        for load in load_levels
    }
    transmission = transmission_power(rates, params)
    return {
        "rates": rates,
        "errors_by_load": errors_by_load,
        "transmission_power": transmission,
    }


def _save_and_close(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_surface_3d(
    result: Question1Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制误差率三维高度图。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    grid = generate_question1_grid(params=params)

    fig = plt.figure(figsize=DEFAULT_STYLE.surface_size)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        grid["loads"],
        grid["rates"],
        grid["errors"],
        cmap="YlOrRd",
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    # 明确画出 E=5% 平面。只用轻量平面 + 网格线，避免遮挡完整误差曲面。
    error_plane = np.full_like(grid["errors"], params.error_limit_percent)
    ax.plot_surface(
        grid["loads"],
        grid["rates"],
        error_plane,
        color="#4D96FF",
        alpha=0.12,
        linewidth=0,
    )
    ax.plot_wireframe(
        grid["loads"],
        grid["rates"],
        error_plane,
        color="#1D5FD1",
        linewidth=0.55,
        rstride=8,
        cstride=8,
        alpha=0.95,
    )
    ax.contour(
        grid["loads"],
        grid["rates"],
        grid["errors"],
        levels=[params.error_limit_percent],
        colors=[QUESTION1_COLORS["error_boundary"]],
        linewidths=3.0,
        zdir="z",
        offset=params.error_limit_percent,
    )

    # 增加若干条固定 GPU 负载的三维切片线，直接展示：
    # 在同一 GPU 负载下，传输速率越高，误差率越低。
    rate_values = grid["rate_values"]
    slice_styles = [
        (70.0, "#1F77B4"),
        (85.0, QUESTION1_COLORS["optimal"]),
        (100.0, "#2CA02C"),
    ]
    for load_value, color in slice_styles:
        load_line = np.full_like(rate_values, load_value)
        error_line = analysis_error(load_line, rate_values, params=params)
        ax.plot(
            load_line,
            rate_values,
            error_line,
            color=color,
            linewidth=3.0 if load_value == result.gpu_load else 2.2,
            label=f"G={load_value:.0f}%切片",
        )

    # 标注误差最高和最低的两个角点，让完整范围更明确。
    ax.scatter([60], [800], [30], color="#8B0000", s=45, depthshade=False)
    ax.text(60, 800, 31.0, "最高误差30%", color="#8B0000")
    ax.scatter([100], [1200], [2], color="#006400", s=45, depthshade=False)
    ax.text(100, 1200, 3.2, "最低误差2%", color="#006400")
    ax.scatter(
        [result.gpu_load],
        [result.transmission_rate],
        [result.weighted_error],
        color=QUESTION1_COLORS["optimal"],
        s=65,
        label="最优点",
        depthshade=False,
    )
    ax.text(
        result.gpu_load,
        result.transmission_rate,
        result.weighted_error + 1.0,
        "最优点",
        color=QUESTION1_COLORS["optimal"],
    )
    ax.set_title("误差率随GPU负载和传输速率变化")
    ax.set_xlabel("GPU负载 G（%）", labelpad=8)
    ax.set_ylabel("数据传输速率 R（Mbps）", labelpad=8)
    ax.set_zlabel("误差率 E（%）", labelpad=10)
    ax.set_xlim(params.gpu_min_load, params.gpu_max_load)
    ax.set_ylim(params.rate_min_mbps, params.rate_max_mbps)
    ax.set_zlim(0, 32)
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_yticks(np.arange(800, 1201, 100))
    ax.set_zticks(np.arange(0, 31, 5))
    ax.set_box_aspect((1.05, 1.2, 0.82))
    ax.view_init(elev=26, azim=-48)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.1, label="误差率（%）")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=QUESTION1_COLORS["optimal"], markersize=8, label="最优点"),
            Patch(facecolor="#4D96FF", alpha=0.28, edgecolor="#1D5FD1", label="误差5%平面"),
            Line2D([0], [0], color=QUESTION1_COLORS["error_boundary"], linewidth=3, label="误差5%边界线"),
            Line2D([0], [0], color=QUESTION1_COLORS["optimal"], linewidth=3, label="G=85%切片：R增大误差下降"),
        ],
        loc="upper left",
    )
    return _save_and_close(fig, output_dir / "error_surface_3d.png")


def plot_power_surface_3d(
    result: Question1Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制系统总功率三维高度图。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    grid = generate_question1_grid(params=params)

    fig = plt.figure(figsize=DEFAULT_STYLE.surface_size)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        grid["loads"],
        grid["rates"],
        grid["powers"],
        cmap="viridis",
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )
    ax.scatter(
        [result.gpu_load],
        [result.transmission_rate],
        [result.system_power],
        color=QUESTION1_COLORS["optimal"],
        s=65,
        label="最优点",
        depthshade=False,
    )
    ax.text(
        result.gpu_load,
        result.transmission_rate,
        result.system_power + 0.25,
        "最优点",
        color=QUESTION1_COLORS["optimal"],
    )
    ax.set_title("总功率随GPU负载和传输速率变化")
    ax.set_xlabel("GPU负载 G（%）", labelpad=8)
    ax.set_ylabel("数据传输速率 R（Mbps）", labelpad=8)
    ax.set_zlabel("系统总功率 P（kW）", labelpad=10)
    ax.set_xlim(params.gpu_min_load, params.gpu_max_load)
    ax.set_ylim(params.rate_min_mbps, params.rate_max_mbps)
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_yticks(np.arange(800, 1201, 100))
    ax.set_box_aspect((1.0, 1.15, 0.75))
    ax.view_init(elev=28, azim=-132)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.1, label="系统总功率（kW）")
    ax.legend(loc="upper left")
    return _save_and_close(fig, output_dir / "power_surface_3d.png")


def plot_feasible_region_2d(
    result: Question1Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制任务量、误差和变量边界共同形成的二维可行域。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)
    grid = generate_question1_grid(params=params)

    fig, ax = plt.subplots(figsize=DEFAULT_STYLE.plane_size)

    # 灰色底图表示不可行区域；可行区域用总功率等高线填色。
    feasible_levels = np.where(grid["feasible"], np.nan, 1.0)
    ax.contourf(
        grid["loads"],
        grid["rates"],
        feasible_levels,
        levels=[0.5, 1.5],
        colors=[QUESTION1_COLORS["infeasible"]],
        alpha=0.85,
    )
    feasible_power = np.ma.masked_where(~grid["feasible"], grid["powers"])
    power_fill = ax.contourf(
        grid["loads"],
        grid["rates"],
        feasible_power,
        levels=12,
        cmap="YlGnBu",
        alpha=0.92,
    )
    ax.contour(
        grid["loads"],
        grid["rates"],
        feasible_power,
        levels=8,
        colors="white",
        linewidths=0.8,
        alpha=0.45,
    )
    fig.colorbar(power_fill, ax=ax, label="可行域内系统总功率（kW）")

    load_values = grid["load_values"]
    work_boundary = 80000.0 / load_values
    error_boundary = 1540.0 - 4.0 * load_values
    work_valid = (work_boundary >= params.rate_min_mbps) & (work_boundary <= params.rate_max_mbps)
    error_valid = (error_boundary >= params.rate_min_mbps) & (error_boundary <= params.rate_max_mbps)

    ax.plot(
        load_values[work_valid],
        work_boundary[work_valid],
        color=QUESTION1_COLORS["work_boundary"],
        linewidth=2.0,
        label="任务量边界",
    )
    ax.plot(
        load_values[error_valid],
        error_boundary[error_valid],
        color=QUESTION1_COLORS["error_boundary"],
        linewidth=2.5,
        label="误差5%边界",
    )
    ax.scatter(
        [result.gpu_load],
        [result.transmission_rate],
        color=QUESTION1_COLORS["optimal"],
        s=70,
        label="最优点",
        zorder=5,
    )
    ax.annotate(
        "最优点",
        xy=(result.gpu_load, result.transmission_rate),
        xytext=(result.gpu_load - 13, result.transmission_rate - 35),
        arrowprops={"arrowstyle": "->", "color": QUESTION1_COLORS["optimal"]},
        color=QUESTION1_COLORS["optimal"],
    )
    ax.set_xlim(params.gpu_min_load, params.gpu_max_load)
    ax.set_ylim(params.rate_min_mbps, params.rate_max_mbps)
    ax.text(
        61.2,
        815,
        "灰色：不可行区域",
        color="#555555",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    ax.text(
        61.2,
        1180,
        "可行域颜色：系统总功率",
        color="#16425B",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    ax.set_title("问题一可行域与总功率等高线")
    ax.set_xlabel("GPU负载 G（%）")
    ax.set_ylabel("数据传输速率 R（Mbps）")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    return _save_and_close(fig, output_dir / "feasible_region_2d.png")


def plot_power_breakdown_curve(
    result: Question1Result,
    output_dir: str | Path,
    *,
    params: ModelParameters = DEFAULT_PARAMS,
) -> Path:
    """绘制 R=1200 时功耗分解和误差曲线。"""

    apply_chinese_style()
    output_dir = ensure_output_dir(output_dir)

    loads = np.linspace(params.gpu_min_load, params.gpu_max_load, 161)
    rate = params.rate_max_mbps
    gpu_power = gpu_cluster_power(loads, params)
    cooling_power = cooling_steady_power(loads, params)
    transmission = np.full_like(loads, transmission_power(rate, params), dtype=float)
    total = system_power(loads, rate, params=params)
    errors = analysis_error(loads, rate, params=params)

    fig, (ax_power, ax_rate_error) = plt.subplots(
        1,
        2,
        figsize=DEFAULT_STYLE.curve_size,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    ax_error = ax_power.twinx()

    ax_power.plot(loads, gpu_power, label="GPU功率", color=QUESTION1_COLORS["gpu"], linewidth=2)
    ax_power.plot(loads, cooling_power, label="冷却功率", color=QUESTION1_COLORS["cooling"], linewidth=2)
    ax_power.plot(loads, transmission, label="传输功率", color=QUESTION1_COLORS["transmission"], linewidth=2)
    ax_power.plot(loads, total, label="系统总功率", color=QUESTION1_COLORS["total"], linewidth=2.4)
    ax_error.plot(loads, errors, label="误差率", color=QUESTION1_COLORS["error"], linewidth=2, linestyle="--")
    ax_error.axhline(
        params.error_limit_percent,
        color=QUESTION1_COLORS["error_boundary"],
        linestyle=":",
        linewidth=2,
        label="误差5%线",
    )
    ax_power.axvline(
        result.gpu_load,
        color=QUESTION1_COLORS["optimal"],
        linestyle="-.",
        linewidth=1.8,
        label="最优GPU负载",
    )

    ax_power.set_title("固定R=1200 Mbps：功耗分解与误差")
    ax_power.set_xlabel("GPU负载 G（%）")
    ax_power.set_ylabel("功率（kW）")
    ax_error.set_ylabel("误差率（%）")
    ax_power.grid(alpha=0.25)

    lines_left, labels_left = ax_power.get_legend_handles_labels()
    lines_right, labels_right = ax_error.get_legend_handles_labels()
    ax_power.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left", ncol=2)

    sensitivity = generate_rate_sensitivity_curves(params=params)
    rate_values = sensitivity["rates"]
    line_colors = {
        70.0: QUESTION1_COLORS["work_boundary"],
        85.0: QUESTION1_COLORS["optimal"],
        100.0: QUESTION1_COLORS["transmission"],
    }
    for load, error_values in sensitivity["errors_by_load"].items():
        linewidth = 3.0 if abs(load - result.gpu_load) < 1e-9 else 2.0
        ax_rate_error.plot(
            rate_values,
            error_values,
            linewidth=linewidth,
            color=line_colors.get(load),
            label=f"G={load:.0f}%",
        )
    ax_rate_error.axhline(
        params.error_limit_percent,
        color=QUESTION1_COLORS["error_boundary"],
        linestyle=":",
        linewidth=2,
        label="误差5%线",
    )
    ax_rate_error.axvline(
        result.transmission_rate,
        color=QUESTION1_COLORS["optimal"],
        linestyle="-.",
        linewidth=1.8,
        label="最优传输速率",
    )
    ax_rate_error.annotate(
        "G=85%时：R从800到1200\n误差由25%降至5%",
        xy=(1200, params.error_limit_percent),
        xytext=(935, 9.5),
        arrowprops={"arrowstyle": "->", "color": QUESTION1_COLORS["optimal"]},
        color=QUESTION1_COLORS["optimal"],
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    ax_rate_error.text(
        805,
        3.0,
        "传输功率仅从0.25kW增至0.41kW",
        color="#444444",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    ax_rate_error.set_title("固定GPU负载：误差随传输速率变化")
    ax_rate_error.set_xlabel("数据传输速率 R（Mbps）")
    ax_rate_error.set_ylabel("误差率（%）")
    ax_rate_error.set_ylim(0, 30)
    ax_rate_error.grid(alpha=0.25)
    ax_rate_error.legend(loc="upper right")

    return _save_and_close(fig, output_dir / "power_breakdown_curve.png")


def generate_all_question1_plots(
    *,
    result: Question1Result | None = None,
    output_dir: str | Path = "outputs/question1",
    params: ModelParameters = DEFAULT_PARAMS,
) -> list[Path]:
    """生成问题一四张图，并返回图片路径列表。"""

    result = solve_question1(params) if result is None else result
    output_dir = ensure_output_dir(output_dir)
    return [
        plot_error_surface_3d(result, output_dir, params=params),
        plot_power_surface_3d(result, output_dir, params=params),
        plot_feasible_region_2d(result, output_dir, params=params),
        plot_power_breakdown_curve(result, output_dir, params=params),
    ]


def main() -> None:
    paths = generate_all_question1_plots()
    print("已生成问题一图像：")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
