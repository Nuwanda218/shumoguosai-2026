"""AI 算力节能调度模型包。

包内各文件按“物理组件、任务误差、电池、目标函数、约束检查”拆分。
后续问题一到问题四的求解脚本应优先复用这里的函数，避免重复写公式。
"""

from .parameters import DEFAULT_PARAMS, ModelParameters, price_schedule

__all__ = ["DEFAULT_PARAMS", "ModelParameters", "price_schedule"]
