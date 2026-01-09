# 回测模块：理论与方法论

## 概述

QuantArk 的回测模块是一个严谨的模拟框架，旨在评估对冲策略和投资组合管理算法的历史表现。与简单的盈亏计算器不同，它模拟了时间的推移、市场数据的更新、策略信号的生成以及具有现实约束的交易执行。

## 与其他风险模块的关系

回测模块是 QuantArk 中三个互补的风险分析框架之一，每个框架解决不同的风险维度：

| 模块 | 时间维度 | 数据来源 | 回答的风险问题 |
|--------|----------------|-------------|------------------------|
| **压力测试** | 瞬时 (t=0) | 假设冲击 | "如果 X 现在发生，我会损失多少？" |
| **动态情景** | 多日 (t=0→T) | 假设路径 | "在情景 X 下，风险如何随 T 天演化？" |
| **回测** | 历史 | 实际市场数据 | "该策略历史上表现如何？" |

回测模块的独特之处在于使用**实际历史市场数据**评估策略。这提供了实证验证，但受限于历史模式——而动态情景模块可探索假设的未来，压力测试模块则检验瞬时尾部风险。

## 核心框架

回测引擎基于时间步进模拟模型运行。在模拟周期的每个时间步中：

1.  **市场数据更新**：`PricingEnvironment`（定价环境）根据 `MarketDataAdapter` 提供的最新数据（现货价格、波动率、利率）进行更新。
2.  **投资组合重估**：使用更新后的环境对组合中的所有持仓进行逐日盯市（Mark-to-Market）。重新计算希腊值（Delta, Gamma, Vega 等）。
3.  **策略评估**：活跃策略（如 Delta 中性、DV01 中性）根据其定义的目标和约束条件分析当前的投资组合状态。
4.  **信号生成**：如果满足再平衡触发条件（例如 Delta 超过阈值），策略将生成所需的交易列表。
5.  **执行与成本建模**：交易在市场中执行。交易成本根据选定的模型（如滑点、佣金）计算，并在净 P&L（扣成本口径）中扣除。
6.  **状态记录**：系统记录完整的状态（盈亏、风险敞口、交易记录）以供事后分析。

**[图片占位符]**
> **Prompt for Nanobanana**: /diagram prompt: "A professional technical flowchart of a quantitative backtesting engine. Left to right flow: 'Market Data Feed' -> 'Portfolio Re-valuation' -> 'Strategy Logic' -> 'Risk Check' -> 'Trade Execution' -> 'Performance Logging'. Use a clean, modern style with a professional blue and grey color scheme."

## 离散时间对冲理论

### 展开式与剩余盈亏

回测中的 Delta/DV01 对冲本质上是一个**离散时间**的复制实验：我们只能在一系列离散时刻进行再平衡，因此剩余盈亏来自更高阶风险项、跳空、以及模型偏差与交易成本。

以单标的期权组合价值 \(V(S, \sigma, t)\) 为例，局部近似可写为：

$$
\Delta V \approx \Delta \,\Delta S + \frac{1}{2}\Gamma (\Delta S)^2 + \Theta \,\Delta t + \text{Vega}\,\Delta \sigma + \cdots
$$

若使用 Delta-one 工具（现货/期货，\(\Delta_{\text{hedge}} \approx 1\)）并设对冲头寸 \(h \approx -\Delta\)，一阶项 \(\Delta \,\Delta S\) 会被显著削弱，但仍会留下：

$$
\Delta V_{\text{hedged}} \approx \frac{1}{2}\Gamma (\Delta S)^2 + \Theta \,\Delta t + \text{Vega}\,\Delta \sigma + \cdots \;-\; \text{交易成本}
$$

因此，"Gamma 损耗（频繁对冲成本）"、Theta 衰减、波动率变化、跳空，以及成本拖累都会体现在回测结果中。

### GBM 下的预期对冲误差

在 Black-Scholes 假设和几何布朗运动下，离散对冲误差可以解析表征。对于以 \(\Delta t\) 间隔再平衡的 Delta 对冲期权：

$$
\mathbb{E}[\text{P\&L}_{\text{hedged}}] \approx \frac{1}{2}\Gamma S^2 \sigma^2 \Delta t + \Theta \Delta t
$$

根据 Black-Scholes 偏微分方程，\(\Theta + \frac{1}{2}\Gamma S^2 \sigma^2 = rV\)，因此在连续极限下预期 P&L 等于无风险收益。然而，**离散再平衡引入了方差**：

$$
\text{Var}(\text{P\&L}_{\text{hedged}}) \approx \frac{1}{2}\Gamma^2 S^4 \sigma^4 (\Delta t)^2 + \cdots
$$

该方差与 \((\Delta t)^2\) 成正比，这意味着更频繁的对冲可以成比例地减少复制误差的平方——但受交易成本制约。

### 最优对冲间隔问题

交易成本创造了一个基本的权衡：更频繁的对冲降低了 Gamma 风险，但增加了成本。**Whalley-Wilmott** 框架将其表征为目标 Delta 周围的"不交易区域"：

$$
|\Delta_{\text{net}} - \Delta_{\text{target}}| < \text{阈值} \implies \text{不对冲}
$$

最优阈值平衡了再平衡的预期边际收益与边际交易成本。对于比例成本 \(c\)（每单位名义），最优半宽度 \(h^*\) 的标度为：

$$
h^* \sim \left(\frac{3c S}{2 |\Gamma| \sigma^2 \Delta t}\right)^{1/3}
$$

这预测了更大的成本、更高的波动率和更长的间隔都会扩大最优不交易区域。

## 对冲策略

### 权益类：Delta 中性对冲
Delta 中性策略的主要目标是使投资组合免受标的资产小幅方向性变动的影响。

*   **理论**：策略监控投资组合的净 Delta ($\Delta_{net}$)。
    $$ \Delta_{net} = \sum_{i} \Delta_i \times Q_i $$
    其中 $\Delta_i$ 是工具 $i$ 的 Delta，$Q_i$ 是数量。
*   **再平衡逻辑**：当 $|\Delta_{net}| > \text{阈值}$ 时触发对冲交易。计算对冲数量 $Q_{hedge}$ 以将 $\Delta_{net}$ 带回目标值（通常为 0）。

    一般形式：
    $$
    Q_{hedge} = - \frac{\Delta_{net} - \Delta_{target}}{\Delta_{hedge\_instrument}} \times \text{hedge\_ratio}
    $$

    对于现货/期货对冲的常见近似（\(\Delta_{hedge\_instrument} \approx 1\)）：
    $$
    Q_{hedge} \approx -(\Delta_{net} - \Delta_{target}) \times \text{hedge\_ratio}
    $$

### 固定收益：DV01 中性对冲
对于固定收益投资组合，主要的风险衡量指标是 DV01（基点价值），代表收益率曲线平行移动 1 个基点时的盈亏变化。

*   **理论**：该策略通过中和投资组合的 DV01 来防止平行利率变动带来的风险。
    $$ \text{DV01}_{net} = \sum_{j} \text{DV01}_j $$
*   **合约数量（期货）**：若对冲期货每张合约的 DV01 为 \(\text{DV01}_{fut}\)，则对冲张数为：
    $$
    N_{contracts} = -\frac{\text{DV01}_{net} - \text{DV01}_{target}}{\text{DV01}_{fut}} \times \text{hedge\_ratio}
    $$
*   **执行**：对冲通常使用国债期货（如 TU, FV, TY, US）进行。合约数量由期货的具体 DV01 和投资组合的敞口决定。

**[图片占位符]**
> **Prompt for Nanobanana**: /diagram prompt: "A financial line chart comparing two equity curves on a dark background. Line 1: 'Unhedged Portfolio' (volatile, jagged, red). Line 2: 'Delta-Neutral Strategy' (smooth, stable, green). Style: Professional financial terminal UI, dark mode, clear grid lines."

## 交易成本模型

### 成本组成

现实的回测需要对摩擦成本进行准确建模。QuantArk 支持全面的成本模型：

1.  **固定佣金**：每笔交易的固定费用。
2.  **比例佣金**：基于名义价值的费用（基点 bps）。
3.  **买卖价差**：跨越价差的成本，实际上在进入和退出时支付一半的价差。
4.  **市场冲击（滑点）**：建模为交易规模相对于市场流动性的函数。大额交易会产生更高的冲击成本。

概念上：
$$
\text{总成本} = \text{固定费用} + (\text{费率} \times |\text{名义价值}|) + \text{价差成本} + \text{滑点成本}
$$

### 成本与流动性的权衡

交易成本从根本上改变了最优对冲策略。在无摩擦的 Black-Scholes 世界中，连续对冲可以消除所有方向性风险。在有成本的情况下，最优策略变为**基于频带**：仅在 Delta 偏离阈值时才对冲。

经济学直觉来自边际分析：
- **对冲的边际收益**：通过减少 Gamma P&L 方差 \(\sim \Gamma^2 S^4 \sigma^4 \Delta t\) 来获得收益
- **对冲的边际成本**：与交易规模线性相关（对于比例成本）

当两者相等时，我们得到最优不交易区域的宽度。Whalley-Wilmott 结果表明阈值标度为 \(c^{1/3}\)——这意味着成本对最优对冲宽度具有**次线性影响**，但效应显著。

### 对夏普比率优化的影响

交易成本通过两种机制降低策略的夏普比率：

1. **直接拖累**：预期 P&L 被每笔交易的平均成本乘以对冲频率所减少
2. **间接方差增加**：为避免成本而设置的更宽的不交易区域增加了 Gamma 风险敞口

对于在周期 \(T\) 内进行 \(N\) 次对冲且平均成本为 \(C\) 的策略：
$$
\text{Sharpe}_{\text{net}} \approx \frac{\mu - \frac{N}{T}C}{\sigma}
$$

其中 \(\mu\) 是总预期收益，\(\sigma\) 是波动率。这突显了权衡：更激进的对冲（更高的 \(N/T\)）降低了 \(\sigma\)，但增加了成本拖累。

## 绩效指标

该模块使用机构级指标评估策略：

### 风险调整收益

**夏普比率** 衡量每单位风险的超额收益：

$$
\text{Sharpe} = \frac{\mathbb{E}[R - R_f]}{\sigma_{R-R_f}} = \frac{\bar{R} - R_f}{\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(R_t - \bar{R})^2}}
$$

其中 \(R_t\) 是时刻 \(t\) 的收益，\(R_f\) 是无风险利率，\(\bar{R}\) 是平均收益。

**索提诺比率** 仅惩罚下行波动：

$$
\text{Sortino} = \frac{\mathbb{E}[R - R_f]}{\sigma_{\text{downside}}}, \quad \sigma_{\text{downside}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\min(R_t - R_f, 0)^2}
$$

### 回撤分析

**最大回撤** 衡量最大的峰谷下跌：

$$
\text{MDD} = \max_{\tau_1 < \tau_2} \left( \frac{V_{\tau_1} - V_{\tau_2}}{V_{\tau_1}} \right)
$$

其中 \(V_t\) 是时刻 \(t\) 的投资组合价值。

**恢复因子** 将总收益与最大回撤关联：

$$
\text{Recovery Factor} = \frac{\text{总收益}}{\text{MDD}}
$$

数值越高表示风险调整后的表现越好。

### 对冲效率

**跟踪误差** 衡量对冲维持其目标的程度：

$$
\text{TE}_{\Delta} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(\Delta_t - \Delta_{\text{target}})^2}
$$

对于 DV01 对冲，类似的指标使用 DV01 偏差。

**对冲频率** 捕获交易强度：

$$
\text{对冲频率} = \frac{N_{\text{hedges}}}{T_{\text{days}}}
$$

这对于成本管理至关重要——更高的频率通常会增加交易成本拖累。

**对冲有效性** 量化方差减少：

$$
\text{HE} = 1 - \frac{\sigma^2_{\text{hedged}}}{\sigma^2_{\text{unhedged}}}
$$

接近 1 的值表示更有效的风险降低。

**[图片占位符]**
> **Prompt for Nanobanana**: /diagram prompt: "A financial dashboard UI wireframe showing four modules: 1. 'Sharpe Ratio' (gauge widget), 2. 'Max Drawdown' (bar chart), 3. 'Win Rate' (pie chart), 4. 'Total Return' (large distinct text). Style: Dark mode professional trading interface, high contrast."

## 实现细节与注意事项 (QuantArk 实现)

回测模块实现了基于协议的架构，通过通用接口支持多种资产类别：

### 支持的策略

| 策略 | 资产类别 | 目标风险 | 对冲工具 |
|----------|-------------|-------------|------------------|
| `DeltaNeutralStrategy` | 权益 | Delta → 0 | 现货或期货 |
| `DV01NeutralStrategy` | 固定收益 | DV01 → 0 | 国债期货 |
| `ConvexityNeutralStrategy` | 固定收益 | 凸性 → 0 | 债券/期货组合 |

### 架构设计

模块使用 Python 协议（`backtest/base.py`）定义以下契约：
- **`BaseBacktestEngine`**：主模拟循环（时间步进、状态管理）
- **`BaseStrategy`**：策略接口（触发评估、对冲规模计算）
- **`BaseHedgeExecutor`**：交易执行和成本计算
- **`BaseBacktestResults`**：结果访问和指标计算

这种基于协议的设计使相同的策略类可以在权益和固定收益投资组合中工作。

### 配置选项

影响计算的关键配置选项：
- **`hedge_frequency`**：控制再平衡节奏（阈值、每日、连续）
- **`hedge_threshold`**：设置不交易区域的宽度
- **`transaction_costs`**：选择成本模型（Zero、Fixed、Proportional、Slippage、Spread、Complete）
- **`save_state`**：启用详细状态历史以供事后分析

### 参考文献

- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*.
- Whalley, A. E., & Wilmott, P. (1997). "An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs." *Mathematical Finance*.
