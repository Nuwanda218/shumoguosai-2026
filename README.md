# AI 算力节能调度建模项目

本项目用于 2026 校选数学建模题“人工智能算力的电力消耗”。当前已经完成基础模型模块和问题一求解、可视化初版；后续将在同一结构下继续实现问题二到问题四。

## 项目结构

```text
建模/
  README.md              # 项目总说明
  2026校选.md            # 题面整理
  模型说明.md            # 数学模型说明文档
  模块设计.md            # 初始模块设计草稿
  2026国赛校选题目.pdf   # 原始题目 PDF

  models/                # 正式可复用模型模块
    parameters.py        # 统一参数、电价表
    components.py        # GPU、传输、冷却功率模型
    task.py              # 处理量、误差、加权平均误差
    battery.py           # 电池 SOC、电网购电功率
    constraints.py       # 边界、周期变化率、违约量检查
    objectives.py        # 系统功率、日能耗、日电费
    _numeric.py          # 内部数值辅助函数
    readme.md            # models 模块说明

  tests/                 # 单元测试
    test_model_modules.py
    test_question1.py
    test_visualization_question1.py

  question/              # 问题一到问题四求解脚本和设计文档
    question1.py         # 问题一：最小能耗静态求解
    问题一设计.md

  visualization/         # 论文图表生成模块
    style.py             # 中文字体、论文配色、图像尺寸
    question1_plots.py   # 问题一图像生成
    问题一图像说明.md

  outputs/               # 求解结果、图片、表格
    README.md
    question1/
      README.md
      error_surface_3d.png
      power_surface_3d.png
      feasible_region_2d.png
      power_breakdown_curve.png

  问题一结果/            # 问题一论文内容导出
    问题一.md
    问题一.tex
    问题一.pdf

  huo/                   # 队友早期草稿代码，保留作参考
    问题一.py
    问题二.py
```

## 建模口径

### 处理量模型

题目给出基准：每块 GPU 负载 80%、传输速率 1000 Mbps，连续运行 24 小时刚好完成一天处理量。

因此第 `t` 小时处理量定义为：

```text
q_t = (G_t / 80) * (R_t / 1000) / 24
```

全天处理量约束：

```text
sum(q_t) >= 1
```

### 误差模型

采用 8 块 GPU 共同降低误差的建模口径：

```text
E_t = 30 - 0.025 * 8 * (G_t - 60) + 0.05 * (800 - R_t)
```

即：

```text
E_t = 30 - 0.2 * (G_t - 60) + 0.05 * (800 - R_t)
```

动态调度中使用全天处理量加权平均误差：

```text
E_bar = sum(q_t * E_t) / sum(q_t) <= 5
```

说明：如果严格按 PDF 字面理解为“整体系统中 GPU 负载每增加 1% 只让误差减少 0.025%”，则最大配置 `G=100, R=1200` 时误差仍为 9%，无法满足 5% 约束。因此项目采用 8-GPU 叠加解释。

## 问题一当前结果

问题一目标为最小化全天总能耗。由于问题一不引入分时电价、电池和冷却惯性，24 小时采用同一组静态调度变量：

```text
G_t = G
R_t = R
```

当前求解结果：

```text
GPU负载 G = 85%
数据传输速率 R = 1200 Mbps
系统功率 P = 6.61 kW
日总能耗 = 158.64 kWh
日任务完成量 = 1.275
误差率 E = 5.0%
```

结果解释：

```text
误差约束 E <= 5 等价于 4G + R >= 1540。
当 R = 1200 Mbps 时，只需 G >= 85% 即可满足误差约束。
提高传输速率的功率代价较小，但降误差效果明显；
提高 GPU 负载会同时增加 GPU 功率和冷却功率。
因此模型优先把 R 提高到上限，再选择刚好满足误差约束的最低 GPU 负载。
```

运行问题一求解：

```bash
python -m question.question1
```

也可以直接运行脚本：

```bash
python question/question1.py
```

生成问题一图像：

```bash
python -m visualization.question1_plots
```

也可以直接运行脚本：

```bash
python visualization/question1_plots.py
```

当前问题一输出图像：

- `outputs/question1/error_surface_3d.png`：误差率三维曲面和 5% 约束平面。
- `outputs/question1/power_surface_3d.png`：系统总功率三维曲面。
- `outputs/question1/feasible_region_2d.png`：任务量、误差约束和可行域功率分布。
- `outputs/question1/power_breakdown_curve.png`：功耗分解和传输速率敏感性。

当前问题一论文结果文件：

- `问题一结果/问题一.md`：问题一文字说明、模型推导、结果解释和图像分析。
- `问题一结果/问题一.tex`：由 Markdown 转换得到的 LaTeX 文件。
- `问题一结果/问题一.pdf`：问题一阶段性论文结果 PDF。

## models 模块职责

`models/` 是正式模型基础层，不直接求解具体问题。

### `parameters.py`

保存题目参数和默认参数对象。

常用内容：

- `ModelParameters`
- `DEFAULT_PARAMS`
- `price_schedule()`

### `components.py`

保存物理组件功率公式：

- `gpu_power_single(load)`：单块 GPU 功率，单位 W。
- `gpu_cluster_power(load)`：8 块 GPU 总功率，单位 kW。
- `transmission_power(rate)`：传输功率，单位 kW。
- `cooling_steady_power(load)`：冷却稳态功率，单位 kW。

### `task.py`

保存任务量和误差公式：

- `hourly_work(load, rate)`
- `total_work(loads, rates)`
- `analysis_error(load, rate)`
- `weighted_average_error(loads, rates)`

### `battery.py`

保存电池相关计算：

- `next_soc(soc, charge_power, discharge_power)`
- `grid_power(system_load_power, charge_power, discharge_power)`

### `constraints.py`

保存通用约束检查函数：

- `within_bounds(values, lower, upper)`
- `cyclic_deltas(values)`
- `max_abs_cyclic_delta(values)`
- `bounds_violation(values, lower, upper)`
- `cyclic_delta_violation(values, limit)`

### `objectives.py`

保存目标函数相关计算：

- `system_power(load, rate)`
- `daily_energy(loads, rates)`
- `daily_cost(loads, rates)`

## 快速示例

计算静态方案 `G=85%, R=1200Mbps`：

```python
import numpy as np

from models.objectives import daily_energy
from models.task import total_work, weighted_average_error

loads = np.full(24, 85.0)
rates = np.full(24, 1200.0)

print(daily_energy(loads, rates))
print(total_work(loads, rates))
print(weighted_average_error(loads, rates))
```

预期结果：

```text
158.64
1.275
5.0
```

## 测试

在项目根目录运行：

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- GPU、传输、冷却参考点。
- 基准处理量。
- 8-GPU 误差模型。
- 分时电价数量。
- 电池 SOC 递推。
- 日能耗和日电费计算。
- 问题一最优解、可行性和直接运行脚本。
- 问题一可视化网格、误差公式、输出图片和统一配色。

## 后续开发规则

1. `models/` 只放可复用公式，不放具体问题求解过程。
2. `question/` 负责问题一到问题四的优化建模和求解。
3. `visualization/` 负责论文图表、结果表和图片导出。
4. 不要在 `question/` 里重复写 GPU、传输、冷却、误差公式，应调用 `models/`。
5. 每次修改模型公式，都要同步更新测试。
6. `huo/` 中代码不直接作为正式版本修改，保留作早期思路参考。
