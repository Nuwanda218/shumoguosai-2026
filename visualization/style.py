"""统一图像样式设置。

所有图像的中文字体、颜色、尺寸和分辨率都集中在这里。
后续如果要统一调整论文图表风格，只改这个文件。
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]


QUESTION1_COLORS = {
    "optimal": "#D62728",
    "error_boundary": "#FF7F0E",
    "work_boundary": "#1F77B4",
    "feasible": "#B7E4C7",
    "infeasible": "#D9D9D9",
    "gpu": "#4C78A8",
    "cooling": "#F58518",
    "transmission": "#54A24B",
    "total": "#222222",
    "error": "#D62728",
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
