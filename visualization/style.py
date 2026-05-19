"""统一图像样式设置。

所有图像的中文字体、颜色、尺寸和分辨率都集中在这里。
后续如果要统一调整论文图表风格，只改这个文件。
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]


PAPER_PALETTE = [
    "#8074C8",
    "#7895C1",
    "#A8CBDF",
    "#D6EFF4",
    "#F2FAFC",
    "#992224",
    "#B54764",
    "#E3625D",
    "#EF8B67",
    "#F0C284",
    "#F5EBAE",
    "#F7FBC9",
]


POWER_CMAP = LinearSegmentedColormap.from_list(
    "question1_power",
    ["#F2FAFC", "#D6EFF4", "#A8CBDF", "#7895C1", "#8074C8"],
)
POWER_SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "question1_power_surface",
    ["#D6EFF4", "#A8CBDF", "#7895C1", "#8074C8", "#992224"],
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "question1_error",
    ["#F7FBC9", "#F5EBAE", "#F0C284", "#EF8B67", "#E3625D", "#B54764", "#992224"],
)


QUESTION1_COLORS = {
    "optimal": "#992224",
    "minimum": "#8074C8",
    "error_plane": "#F5EBAE",
    "error_boundary": "#E3625D",
    "work_boundary": "#7895C1",
    "feasible": "#D6EFF4",
    "infeasible": "#D9D9D9",
    "gpu": "#7895C1",
    "cooling": "#EF8B67",
    "transmission": "#A8CBDF",
    "total": "#992224",
    "error": "#B54764",
    "text_dark": "#992224",
    "text_blue": "#8074C8",
}


@dataclass(frozen=True)
class FigureStyle:
    """论文图像通用样式。"""

    dpi: int = 180
    surface_size: tuple[float, float] = (9.6, 7.1)
    plane_size: tuple[float, float] = (8.0, 6.2)
    curve_size: tuple[float, float] = (12.0, 5.8)


DEFAULT_STYLE = FigureStyle()


def apply_chinese_style() -> None:
    """应用中文字体和通用 Matplotlib 设置。"""

    plt.rcParams["font.sans-serif"] = CHINESE_FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = DEFAULT_STYLE.dpi
    plt.rcParams["savefig.dpi"] = DEFAULT_STYLE.dpi
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
