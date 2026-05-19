# question 目录说明

`question/` 用来保存问题一到问题四的求解脚本和对应设计文档。

当前已完成问题一求解与可视化，问题二已完成求解代码初版。问题二图像与结果文件导出后续再讨论。

## 计划结构

```text
question/
  README.md
  问题一设计.md
  问题二设计.md
  question1.py      # 问题一求解代码
  question2.py      # 问题二求解代码
```

后续达成共识后再新增：

```text
question/
  question3.py      # 问题三求解代码
  question4.py      # 问题四求解代码
```

## 开发原则

1. `question/` 中的代码只负责组织某一道题的目标函数、约束、求解器调用和结果汇总。
2. GPU、传输、冷却、处理量、误差、电价、电池等公式必须复用 `models/`。
3. 图表和结果表导出必须调用 `visualization/`，不要混在求解脚本里。
4. 每个问题先写设计 Markdown，再写测试，再写代码。
