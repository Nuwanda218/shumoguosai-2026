# visualization 目录说明

`visualization/` 用来保存论文图表和结果表生成模块。

当前已完成问题一、问题二的可视化模块初版，后续继续补充问题三、问题四图像。

## 计划结构

```text
visualization/
  README.md
  style.py
  question1_plots.py
  question2_plots.py
  问题一图像说明.md
  问题二图像说明.md
```

## 设计原则

1. 可视化模块只接收求解结果，不负责优化求解。
2. 图片风格要适合论文，不做复杂交互图。
3. 所有图片默认保存到 `outputs/对应问题/`。
4. 图表标题、坐标轴、图例统一使用中文。
5. 后续 Q1-Q4 共用同一套绘图函数，避免每个问题重复写图表代码。

## 问题一图表计划

### 功率构成图

展示：

```text
GPU 功率
数据传输功率
冷却功率
```

用途：

```text
说明最优方案下能耗主要来自哪些部分。
```

### 结果汇总图

展示：

```text
GPU 负载
传输速率
日总能耗
误差
总处理量
```

用途：

```text
论文中快速展示问题一最优方案。
```

### 结果表

建议保存为：

```text
outputs/question1/question1_result.md
outputs/question1/question1_result.csv
```

Markdown 表适合直接复制到论文，CSV 表适合后续数据处理。

## 问题二图表计划

问题二图像说明见：

```text
visualization/问题二图像说明.md
```

绘图代码：

```text
visualization/question2_plots.py
```

输出目录：

```text
outputs/question2/
```

当前输出五张图：

```text
tariff_schedule.png              # 分时电价与调度曲线
power_cost_schedule.png          # 系统功率与小时电费
work_error_contribution.png      # 处理量与误差贡献
q1_q2_comparison.png             # 问题一与问题二对比
rate_max_reason.png              # 传输速率取上限原因
```


配色：
序号	HEX 色值	RGB 色值
1	#8074C8	128, 116, 200
2	#7895C1	120, 149, 193
3	#A8CBDF	168, 203, 223
4	#D6EFF4	214, 239, 244
5	#F2FAFC	242, 250, 252
6	#992224	153, 34, 36
7	#B54764	181, 71, 100
8	#E3625D	227, 98, 93
9	#EF8B67	239, 139, 103
10	#F0C284	240, 194, 132
11	#F5EBAE	245, 235, 174
12	#F7FBC9	247, 251, 201
