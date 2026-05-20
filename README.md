# AI 算力节能调度建模项目

本项目用于 2026 校选数学建模题“人工智能算力的电力消耗”。目前四个问题的建模、求解代码、结果文档和可视化初版均已完成；图像可继续按论文排版需要微调。

## 当前进度

| 模块 | 状态 | 主要文件 |
| ---- | ---- | -------- |
| 基础模型模块 | 已完成 | `models/` |
| 问题一：最小能耗 | 已完成 | `question/question1.py`，`问题结果md/问题一.md` |
| 问题二：分时电价最小电费 | 已完成 | `question/question2.py`，`问题结果md/问题二.md` |
| 问题三：电池储能与冷却惯性 | 已完成 | `question/question3.py`，`问题结果md/问题三.md` |
| 问题四：负载与传输速率变化率约束 | 初版完成 | `question/question4.py`，`问题结果md/问题四.md` |
| 四问可视化 | 初版完成 | `visualization/`，`outputs/` |

## 目录说明

```text
建模/
  README.md                 # 项目总说明，给队友快速定位文件
  2026校选.md               # 题面整理
  2026国赛校选题目.pdf      # 原始题目 PDF
  模型说明.md               # 总体模型说明

  models/                   # 可复用模型模块，不写具体问题求解
    parameters.py           # 参数、电价、边界
    components.py           # GPU、传输、冷却功率
    task.py                 # 处理量、误差、加权平均误差
    battery.py              # 电池 SOC、电网购电功率
    constraints.py          # 边界、周期变化率、违约量检查
    objectives.py           # 系统功率、日能耗、日电费
    readme.md               # models 模块说明

  question/                 # 四个问题的设计说明与求解代码
    question1.py
    question2.py
    question3.py
    question4.py
    问题一设计.md
    问题二设计.md
    问题三设计.md
    问题四设计.md

  visualization/            # 可视化代码和图像说明
    style.py                # 中文字体、统一配色、图像参数
    question1_plots.py
    question2_plots.py
    question3_plots.py
    question4_plots.py
    问题一图像说明.md
    问题二图像说明.md
    问题三图像说明.md
    问题四图像说明.md

  outputs/                  # 已生成的图像和输出说明
    question1/
    question2/
    question3/
    question4/

  问题结果md/               # 论文文字结果，可直接取用和改写
    问题一.md
    问题二.md
    问题三.md
    问题四.md

  tests/                    # 单元测试

  huo/                      # 队友早期草稿代码，仅作参考
```

## 统一建模口径

处理量采用乘积正比：

```text
q_t = (G_t / 80) * (R_t / 1000) / 24
sum(q_t) >= 1
```

误差采用 8 块 GPU 共同降低误差的解释：

```text
E_t = 30 - 0.025 * 8 * (G_t - 60) + 0.05 * (800 - R_t)
```

动态调度问题采用全天处理量加权平均误差：

```text
sum(q_t * E_t) / sum(q_t) <= 5
```

说明：如果把题面“单块 GPU 每增加 1% 误差降低 0.025%”理解为不叠加，则最大配置下误差仍无法降到 5%，模型不可行。因此正式模型采用 8-GPU 叠加解释。

## 四问结果概览

### 问题一

目标：最小化系统总能耗。

结果：

```text
GPU负载 G = 85%
数据传输速率 R = 1200 Mbps
系统功率 = 6.61 kW
日总能耗 = 158.64 kWh
总处理量 = 1.275
误差率 = 5.0000%
```

结论：传输速率功耗代价较小但能显著降低误差，因此最优策略先把传输速率拉到上限，再选择刚好满足误差约束的最低 GPU 负载。

### 问题二

目标：加入峰、平、谷分时电价后，最小化日电费。

结果：

```text
谷时段：G = 100.000%，R = 1200 Mbps
平时段：G = 78.642%，R = 1200 Mbps
峰时段：G = 60.000%，R = 1200 Mbps
最小日电费 = 162.3362 元
总处理量 = 1.2398
加权平均误差 = 5.0000%
```

结论：低电价时段提高负载完成更多任务，高电价时段降低负载；传输速率仍保持上限，因为它的功耗增量较小，并且有利于降低误差。

### 问题三

目标：在问题二基础上加入电池储能和冷却惯性，继续最小化日电费。

推荐长期循环方案：

```text
循环保留电量 = 24 kWh
日电费 = 158.6374 元
系统总能耗 = 157.1937 kWh
电网购电量 = 163.9493 kWh
总处理量 = 1.2690
加权平均误差 = 5.0000%
总充电量 = 35.5556 kWh
总放电量 = 28.8000 kWh
峰时段放电量 = 24.8627 kWh
最大冷却功率变化 = 0.2000 kW
```

结论：电池在谷时段充电、峰时段放电，进一步降低电费；冷却惯性使 GPU 负载曲线更平滑，也让模型更接近实际运行。

### 问题四

目标：在问题三基础上加入 GPU 负载变化率和传输速率变化率约束。

结果：

```text
结果来源 = question3-feasible
日电费 = 158.6374 元
总处理量 = 1.2690
加权平均误差 = 5.0000%
最大 GPU 负载变化 = 1.6667% / 8%
最大传输速率变化 = 0.0000 Mbps / 300 Mbps
最大冷却功率变化 = 0.2000 kW / 0.2 kW
```

结论：问题三推荐方案已经满足问题四新增约束，因此问题四最优结果与问题三一致。真正限制调度平滑性的是冷却惯性约束，而不是 GPU 负载变化率或传输速率变化率。

## 常用运行命令

请在项目根目录运行命令。

求解四个问题：

```bash
python -m question.question1
python -m question.question2
python -m question.question3
python -m question.question4
```

生成四个问题的图像：

```bash
python -m visualization.question1_plots
python -m visualization.question2_plots
python -m visualization.question3_plots
python -m visualization.question4_plots
```

如果使用直接脚本运行，也可以：

```bash
python question/question1.py
python question/question2.py
python question/question3.py
python question/question4.py

python visualization/question1_plots.py
python visualization/question2_plots.py
python visualization/question3_plots.py
python visualization/question4_plots.py
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 图像输出

问题一图像：

- `outputs/question1/error_surface_3d.png`
- `outputs/question1/power_surface_3d.png`
- `outputs/question1/feasible_region_2d.png`
- `outputs/question1/power_breakdown_curve.png`

问题二图像：

- `outputs/question2/tariff_schedule.png`
- `outputs/question2/power_cost_schedule.png`
- `outputs/question2/work_error_contribution.png`
- `outputs/question2/rate_max_reason.png`
- `outputs/question2/q1_q2_comparison.png`

问题三图像：

- `outputs/question3/reserve_soc_comparison.png`
- `outputs/question3/schedule_profile.png`
- `outputs/question3/battery_schedule.png`
- `outputs/question3/cooling_inertia.png`
- `outputs/question3/grid_power_cost.png`
- `outputs/question3/q2_q3_comparison.png`

问题四图像：

- `outputs/question4/schedule_profile.png`
- `outputs/question4/change_rate_check.png`
- `outputs/question4/constraint_margin_summary.png`
- `outputs/question4/q3_q4_comparison.png`

所有图像说明文档在 `visualization/问题一图像说明.md` 到 `visualization/问题四图像说明.md`。

## 队友取用建议

1. 写论文模型部分：优先看 `模型说明.md` 和 `question/问题X设计.md`。
2. 写论文结果部分：优先看 `问题结果md/问题X.md`。
3. 插图：直接从 `outputs/questionX/` 取 PNG。
4. 查公式实现：看 `models/`，不要以 `huo/` 为正式代码依据。
5. 继续改图：只改 `visualization/`，不要把绘图逻辑写进 `question/`。
6. 继续改模型：先改 `models/` 或对应 `question/questionX.py`，再同步更新测试和结果文档。
