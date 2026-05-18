# AI 算力节能调度建模项目

本项目用于 2026 校选数学建模题“人工智能算力的电力消耗”。当前目标是把题目中的 GPU 功耗、数据传输功耗、冷却功耗、任务处理量、误差约束、电池储能和分时电价拆成可复用模块，后续再分别实现问题一到问题四的求解代码和论文图表。

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

  huo/                   # 队友早期草稿代码，保留作参考
    问题一.py
    问题二.py
```

计划后续新增：

```text
question/                # 问题一到问题四求解脚本
visualization/           # 论文图表和结果表生成模块
outputs/                 # 求解结果、图片、表格
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

## 后续开发规则

1. `models/` 只放可复用公式，不放具体问题求解过程。
2. `question/` 负责问题一到问题四的优化建模和求解。
3. `visualization/` 负责论文图表、结果表和图片导出。
4. 不要在 `question/` 里重复写 GPU、传输、冷却、误差公式，应调用 `models/`。
5. 每次修改模型公式，都要同步更新测试。
6. `huo/` 中代码不直接作为正式版本修改，保留作早期思路参考。
