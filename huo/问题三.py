import numpy as np
from scipy.optimize import minimize

# ==================== 模型参数 ====================
hours = 24  # 时段数
Q0 = 24 * 80 * 1000  # 基准处理总量（% * Mbps * h）

# 分时电价（元/kWh），按小时索引0~23
price = np.zeros(24)
# 峰时段 9:00-11:00 (索引9,10) 和 18:00-20:00 (索引18,19)
for t in [9,10,18,19]:
    price[t] = 2.0
# 平时段 6:00-9:00 (6,7,8)，11:00-18:00 (11-17)，20:00-22:00 (20,21)
for t in [6,7,8] + list(range(11,18)) + [20,21]:
    price[t] = 1.2
# 谷时段 22:00-6:00 (22,23,0,1,2,3,4,5)
for t in [22,23] + list(range(0,6)):
    price[t] = 0.8

# ==================== 物理模型函数 ====================
def gpu_power_single(x):
    """单块GPU功率（瓦），x为负载百分比"""
    if x <= 80:
        return 180 + 3 * (x - 60)
    else:
        return 240 + 2 * (x - 80)

def total_power(x, v):
    """给定负载x（%）和速率v（Mbps），计算系统总功率（kW）"""
    # 8块GPU总功率，从瓦转为千瓦
    p_gpu = 8 * gpu_power_single(x) / 1000.0
    # 数据传输功率（kW）
    p_trans = 0.25 + 0.0004 * (v - 800)
    # 冷却功率（kW）
    p_cool = 1.2 + 0.12 * (x - 60)
    return p_gpu + p_trans + p_cool

# ==================== 优化问题的构建 ====================
# 决策变量：将24个x和24个v平铺为一个向量，长度为48
# 前24个为x[0]~x[23]，后24个为v[0]~v[23]
def obj_func(z):
    """目标函数：总电费（元）"""
    x = z[:24]
    v = z[24:48]
    cost = 0.0
    for t in range(24):
        cost += price[t] * total_power(x[t], v[t])
    return cost

def constraint_total(z):
    """总处理量约束：∑ x_t * v_t >= Q0，转化为不等式形式 Q0 - ∑ <= 0"""
    x = z[:24]
    v = z[24:48]
    total_work = np.sum(x * v)
    return Q0 - total_work  # 必须 <= 0

def constraint_error(z):
    """每个时段的误差约束：0.2*x + 0.05*v >= 77，转化为 77 - (0.2x+0.05v) <= 0"""
    x = z[:24]
    v = z[24:48]
    cons = []
    for t in range(24):
        cons.append(77.0 - (0.2 * x[t] + 0.05 * v[t]))
    return np.array(cons)

# 变量边界：x∈[60,100]，v∈[800,1200]
bounds = [(60, 100)] * 24 + [(800, 1200)] * 24

# 初始猜测：所有时段运行在基准点 x=80, v=1000
x0 = np.array([80.0] * 24)
v0 = np.array([1000.0] * 24)
z0 = np.hstack([x0, v0])

# 定义约束（scipy.minimize 要求每个约束为一个字典）
cons = []
# 总处理量约束为不等式（type='ineq' 要求约束函数 >=0，因此需要取相反数）
# 为了更直观，定义 constraint_total_geq(z) = ∑x*v - Q0 >=0
def total_work_geq(z):
    x = z[:24]
    v = z[24:48]
    return np.sum(x * v) - Q0
cons.append({'type': 'ineq', 'fun': total_work_geq})

# 误差约束：每个时段有 0.2x+0.05v - 77 >=0
for t in range(24):
    def error_geq(z, t=t):
        return 0.2 * z[t] + 0.05 * z[24+t] - 77.0
    cons.append({'type': 'ineq', 'fun': error_geq})

# ==================== 求解 ====================
# 使用 SLSQP 算法，适用于中小规模非线性约束问题
print("正在求解动态调度问题...")
res = minimize(obj_func, z0, method='SLSQP', bounds=bounds,
               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-6})

if res.success:
    print("优化成功完成")
    opt_x = res.x[:24]
    opt_v = res.x[24:48]
    total_cost = res.fun
    total_work = np.sum(opt_x * opt_v)
    print(f"最小总电费: {total_cost:.2f} 元")
    print(f"实际总处理量: {total_work:.0f} (基准: {Q0})")
    print("\n各时段最优调度方案:")
    print("时段\t负载(%)\t速率(Mbps)\t功率(kW)\t电价(元/kWh)\t电费(元)")
    for t in range(24):
        p = total_power(opt_x[t], opt_v[t])
        cost_t = price[t] * p
        print(f"{t+1:2d}\t{opt_x[t]:6.2f}\t{opt_v[t]:8.0f}\t{p:8.3f}\t{price[t]:8.2f}\t{cost_t:8.3f}")
else:
    print("优化失败:", res.message)

# ==================== 求解算法说明 ====================
# 本代码采用序列二次规划（SQP）算法，将24小时内的GPU负载x_t和传输速率v_t
# 作为连续决策变量，以分时电价下的总电费最小为目标。约束条件包括：
# 1) 每个时段负载和速率上下限；
# 2) 每个时段的数据分析误差不超过5%，该约束被线性化为0.2x_t+0.05v_t>=77；
# 3) 全天处理总量不低于基准值Q0。
# 目标函数和约束均为非线性（功率分段线性、乘积约束），但连续可微。
# SQP方法通过迭代求解二次规划子问题逼近原问题最优解，适合此类中小规模问题。
# 初始点取所有时段均为80%负载和1000 Mbps，该点满足误差和总量约束（实际总量为1.92e6），
# 但通过优化可以降低电费，例如在低电价时段增加处理量，高电价时段降低负载或速率。
# 功率函数中的GPU功耗采用分段线性形式，在代码中通过条件判断实现，
# 为保持梯度平滑，分段点处的导数略有变化，但SQP算法仍能有效处理。
# 运行结果将输出各时段的最优调度方案及总电费。
