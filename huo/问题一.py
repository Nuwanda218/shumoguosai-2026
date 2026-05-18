import numpy as np
from scipy.optimize import minimize

# =====================================
# 1. GPU单块功率模型
# =====================================
def gpu_power(x):
    """
    单块GPU功率函数
    参数：
        x : GPU负载(%)
    返回：
        功率(W)
    """

    if 60 <= x <= 80:
        return 180 + 3 * (x - 60)

    elif 80 < x <= 100:
        return 240 + 2 * (x - 80)

    else:
        # 超出约束范围给予极大惩罚
        return 1e6


# =====================================
# 2. GPU集群总能耗
# =====================================
def gpu_energy(x):
    """
    GPU集群全天总能耗
    8块GPU同步运行
    单位：kWh
    """

    total_power = 8 * gpu_power(x)

    return 24 * total_power / 1000


# =====================================
# 3. 数据传输能耗模型
# =====================================
def transmission_energy(v):
    """
    数据传输全天总能耗
    参数：
        v : 数据传输速率(Mbps)

    返回：
        总能耗(kWh)
    """

    power = 0.25 + 0.0004 * (v - 800)

    return 24 * power


# =====================================
# 4. 冷却系统能耗模型
# =====================================
def cooling_energy(x):
    """
    冷却系统全天总能耗
    参数：
        x : GPU负载(%)

    返回：
        冷却系统能耗(kWh)
    """

    cooling_power = 1.2 + 0.12 * (x - 60)

    return 24 * cooling_power


# =====================================
# 5. 系统总能耗目标函数
# =====================================
def total_energy(vars):
    """
    系统总能耗目标函数

    vars[0] -> GPU负载 x
    vars[1] -> 传输速率 v
    """

    x, v = vars

    E_gpu = gpu_energy(x)

    E_trans = transmission_energy(v)

    E_cool = cooling_energy(x)

    return E_gpu + E_trans + E_cool


# =====================================
# 6. 数据误差模型
# =====================================
def error_rate(vars):
    """
    数据分析误差模型

    GPU集群共有8块GPU：
    单块GPU负载每增加1%
    总误差下降：
        8 × 0.025% = 0.2%

    返回：
        系统误差(%)
    """

    x, v = vars

    R = 30 - 0.2 * (x - 60) + 0.05 * (800 - v)

    return R


# =====================================
# 7. 非线性约束条件
# =====================================
# scipy.optimize.minimize 中：
# 不等式约束要求：
# constraint(vars) >= 0

constraints = [
    {
        'type': 'ineq',
        'fun': lambda vars: 5 - error_rate(vars)
    }
]


# =====================================
# 8. 变量边界
# =====================================
bounds = [
    (60, 100),      # GPU负载范围
    (800, 1200)     # 传输速率范围
]


# =====================================
# 9. 设置优化初始点
# =====================================
x0 = [80, 1000]


# =====================================
# 10. 调用SLSQP优化算法
# =====================================
result = minimize(
    fun=total_energy,
    x0=x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)


# =====================================
# 11. 输出优化结果
# =====================================
optimal_x = result.x[0]

optimal_v = result.x[1]

optimal_energy = result.fun

optimal_error = error_rate(result.x)

print("========== 问题一优化结果 ==========")

print(f"最优GPU负载：{optimal_x:.2f}%")

print(f"最优数据传输速率：{optimal_v:.2f} Mbps")

print(f"系统最小总能耗：{optimal_energy:.2f} kWh")

print(f"最终数据分析误差：{optimal_error:.2f}%")
