# models 模型模块说明

`models/` 是正式建模代码的基础模块目录。它只负责保存题目参数、物理公式、任务量公式、误差公式、约束检查和目标函数，不直接求解某一道题。

后续 `question/` 目录中的问题一、问题二、问题三、问题四求解脚本，应优先调用这里的函数，避免重复写公式。

## 目录结构

```text
models/
  __init__.py        # 对外导出基础参数
  parameters.py      # 统一参数、电价表
  components.py      # GPU、传输、冷却功率模型
  task.py            # 处理量、误差、加权平均误差
  battery.py         # 电池 SOC、电网购电功率
  constraints.py     # 边界、周期变化率、违约量检查
  objectives.py      # 系统功率、日能耗、日电费
  _numeric.py        # 内部数值辅助函数
```

## 模块职责

### `parameters.py`

集中管理题目给定参数和建模默认参数。

主要内容：

- `ModelParameters`：参数数据类。
- `DEFAULT_PARAMS`：默认参数对象。
- `price_schedule()`：返回 24 小时分时电价数组。

重要约定：

- GPU 数量：8 块。
- GPU 负载范围：60% 到 100%。
- 传输速率范围：800 到 1200 Mbps。
- 误差上限：5%。
- 电池容量：40 kWh。
- 初始 SOC：16 kWh。
- 最低 SOC：8 kWh。

### `components.py`

计算物理组件功率。

主要函数：

- `gpu_power_single(load)`：单块 GPU 功率，单位 W。
- `gpu_cluster_power(load)`：8 块 GPU 集群功率，单位 kW。
- `transmission_power(rate)`：数据传输功率，单位 kW。
- `cooling_steady_power(load)`：冷却系统稳态功率，单位 kW。

示例：

```python
from models.components import gpu_cluster_power, transmission_power, cooling_steady_power

gpu_kw = gpu_cluster_power(85)
net_kw = transmission_power(1200)
cool_kw = cooling_steady_power(85)
```

### `task.py`

计算任务完成量和数据分析误差。

主要函数：

- `hourly_work(load, rate)`：某小时完成的日任务量比例。
- `total_work(loads, rates)`：24 小时总任务量比例。
- `analysis_error(load, rate)`：某小时误差。
- `weighted_average_error(loads, rates)`：全天处理量加权平均误差。

处理量公式：

```text
q_t = (G_t / 80) * (R_t / 1000) / 24
```

默认误差公式：

```text
E_t = 30 - 0.025 * 8 * (G_t - 60) + 0.05 * (800 - R_t)
```

这里采用 8 块 GPU 共同降低误差的建模口径。若严格按 PDF 单块 GPU 解释，可调用：

```python
analysis_error(100, 1200, aggregate_gpu_effect=False)
```

此时最小误差为 9%，无法满足 5% 约束。

### `battery.py`

计算电池状态和电网购电功率。

主要函数：

- `next_soc(soc, charge_power, discharge_power)`：下一小时电池电量，单位 kWh。
- `grid_power(system_load_power, charge_power, discharge_power)`：考虑电池后的电网购电功率，单位 kW。

SOC 递推公式：

```text
S_{t+1} = S_t + 0.9 * P_ch - P_dis / 0.9
```

### `constraints.py`

提供通用约束检查工具。

主要函数：

- `within_bounds(values, lower, upper)`：检查变量是否在上下界内。
- `cyclic_deltas(values)`：计算日周期相邻小时变化量。
- `max_abs_cyclic_delta(values)`：计算最大日周期变化幅度。
- `bounds_violation(values, lower, upper)`：计算上下界最大违约量。
- `cyclic_delta_violation(values, limit)`：计算变化率最大违约量。

日周期表示第 23 小时和第 0 小时也要衔接，用于问题三、问题四的冷却功率和调度平滑约束。

### `objectives.py`

计算目标函数相关数值。

主要函数：

- `system_power(load, rate)`：系统总功率，单位 kW。
- `daily_energy(loads, rates)`：一天总能耗，单位 kWh。
- `daily_cost(loads, rates)`：一天总电费，单位元。

系统总功率：

```text
P_system = P_gpu + P_transmission + P_cooling
```

若问题三、问题四中冷却功率是优化变量，可用：

```python
system_power(loads, rates, cooling_power=cooling_powers)
```

## 常用示例

计算问题一静态方案 `G=85, R=1200`：

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

## 后续开发规则

1. 不要在 `question/` 求解脚本里重复写 GPU、传输、冷却、误差公式。
2. 新问题需要新约束时，优先在 `constraints.py` 中加入通用检查函数。
3. 新目标函数如果可复用，放到 `objectives.py`。
4. 电池相关计算统一放在 `battery.py`。
5. 每次改动模型公式后，都要补充或更新 `tests/` 中的测试。

## 测试命令

在项目根目录运行：

```bash
python -m unittest discover -s tests -v
```

当前测试主要校验：

- GPU、传输、冷却参考点。
- 基准处理量。
- 8-GPU 误差口径。
- 分时电价数量。
- 电池 SOC 递推。
- 日能耗和日电费计算。
