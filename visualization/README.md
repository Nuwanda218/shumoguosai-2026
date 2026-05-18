# visualization 目录说明

`visualization/` 用来保存论文图表和结果表生成模块。

当前阶段只讨论设计，不写绘图代码。等问题一输出结果结构确定后，再创建 Python 可视化模块。

## 计划结构

```text
visualization/
  README.md
```

后续达成共识后再新增：

```text
visualization/
  plots.py          # 通用绘图函数
  tables.py         # 可选：结果表导出函数
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
