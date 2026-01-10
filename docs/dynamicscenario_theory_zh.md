# 动态情景模块：理论与方法论

## 概述

动态情景模块将风险分析扩展到了时间维度。与瞬时的静态压力测试不同，动态情景模拟了市场因子在多日跨度内的**演变**。这使得分析路径依赖风险、动态对冲策略的有效性以及时间衰减（Theta）的影响成为可能。

## 与其他风险模块的关系

动态情景模块是 QuantArk 中三个互补的风险分析框架之一，每个框架解决不同的风险维度：

| 模块 | 时间维度 | 数据来源 | 回答的风险问题 |
|--------|----------------|-------------|------------------------|
| **压力测试** | 瞬时 (t=0) | 假设冲击 | "如果 X 现在发生，我会损失多少？" |
| **动态情景** | 多日 (t=0→T) | 假设路径 | "在情景 X 下，风险如何随 T 天演化？" |
| **回测** | 历史 | 实际市场数据 | "该策略历史上表现如何？" |

动态情景模块的独特之处在于其**前瞻性的路径依赖性**：它模拟投资组合风险如何在假设情景下随时间演化。与压力测试（瞬时快照）不同，它可以探索假设的未来并分析 Gamma 损耗和 Theta 损耗等路径依赖效应；与回测（历史重演）不同，它不受历史数据限制。

## 路径生成理论

动态情景由 $T$ 天内的市场状态序列定义。该模块主要使用两种方法构建这些路径：

### 1. 参数化路径

这些是为了测试特定结构性风险而构建的合成路径。

#### 几何布朗运动 (GBM) 路径

对于带趋势的权益现货动态，参数化形式遵循 GBM 解：

$$
S_t = S_0 \exp\left(\mu t + \sigma W_t\right)
$$

其中：
- \(\mu\) 是漂移（预期收益）
- \(\sigma\) 是波动率
- \(W_t\) 是维纳过程（布朗运动）

在日步长 \(\Delta t = 1\) 的离散时间中：

$$
S_{t+1} = S_t \exp\left(\mu - \frac{1}{2}\sigma^2 + \sigma \sqrt{\Delta t} \, Z_t\right)
$$

其中 \(Z_t \sim \mathcal{N}(0,1)\) 是独立的标准正态分布。

#### 均值回归路径 (Ornstein-Uhlenbeck)

对于波动率和利率，均值回归往往更合适：

$$
dX_t = \kappa(\theta - X_t)dt + \sigma dW_t
$$

离散形式（欧拉离散化）：

$$
X_{t+1} = X_t + \kappa(\theta - X_t)\Delta t + \sigma\sqrt{\Delta t}\,Z_t
$$

其中：
- \(\theta\) 是长期均值
- \(\kappa\) 是均值回归速度
- \(\sigma\) 是过程波动率

#### 波动率机制转换

对于建模波动率波动或突然的机制转换，动态可建模为：

$$
\sigma_t = \sigma_0 \cdot \exp\left(\alpha \cdot \mathbb{I}_{t > t_{\text{switch}}}\right)
$$

其中 \(\mathbb{I}\) 是机制转换时间的指示函数。

### 2. 固定收益路径动态

对于收益率曲线演变，**Heath-Jarrow-Morton (HJM)** 框架提供了理论基础：

$$
df(t,T) = \alpha(t,T)dt + \sigma(t,T)dW_t
$$

其中 \(f(t,T)\) 是时刻 \(t\) 对到期日 \(T\) 的瞬时远期利率。

在实践中，情景通常聚焦于收益率曲线的前三个**主成分**：

1. **水平 (Level)**：平行移动（解释约 70-80% 的方差）
2. **斜率 (Slope)**：变陡/变平（解释约 10-15% 的方差）
3. **曲率 (Curvature)**：扭曲/驼峰（解释约 5-10% 的方差）

### 3. 历史自举（未来规划）

通过从历史回报中采样来构建路径，以保留资产和波动率之间的经验相关性结构。

## 建模风险因子
动态情景允许复杂市场因子的演变：

*   **现货动态 (Spot Dynamics)**：建模资产价格的漂移（趋势）和扩散（波动）。
*   **波动率曲面 (Volatility Surface)**：隐含波动率曲面随时间的演变。
    *   *粘性行权价 (Sticky Strike) vs. 粘性 Delta (Sticky Delta)*：处理偏度如何相对于现货移动。
    *   *波动率的波动 (Vol-of-Vol)*：波动率水平本身的波动。
*   **收益率曲线主成分 (Yield Curve Principal Components)**：建模利率的水平、斜率和曲率的独立运动。
*   **期限结构 (Term Structure)**：远期利率的演变以及债券价格的“滚落”效应。

**[图片占位符]**
> **Prompt for Nanobanana**: /diagram prompt: "A 3D scientific surface plot of a 'Volatility Surface'. Use visual indicators like semi-transparent 'ghost' surfaces or directional arrows to depict the surface rippling and changing shape over time (Vol-of-Vol). Style: High-tech 3D data visualization, blue and purple gradient."

## QuantArk 中的路径表达（DayPath / DayStep）

在 QuantArk 中，动态情景由 **DayPath** 表示：它是按天顺序排列的 **DayStep** 列表。每个 DayStep 包含一个或多个 **ParameterChange**（现货/波动率/利率/股息等），并指定冲击类型：

* **百分比（复利）**：\(X_t = X_{t-1}(1+\epsilon_t)\)
* **绝对值（加法）**：\(X_t = X_{t-1}+\delta_t\)
* **覆盖值（设定）**：\(X_t = \bar{X}_t\)

这种表示方式让情景在“每日粒度”上可审计：每一天发生了哪些变化、为什么发生、影响哪些对象（组合/标的/持仓）。

## 动态对冲模拟

该模块的一个关键特性是能够模拟路径上的**主动投资组合管理**。

对于情景中的每一天 $t$：
1.  **推进时间**：将估值日期移动到 $t$。到期时间减少 ($T-t$)。
2.  **更新市场**：应用情景定义的第 $t$ 天市场参数。
3.  **评估投资组合**：计算盈亏和风险指标（希腊值）。
4.  **对冲逻辑**：
    *   检查对冲触发条件（例如，Delta 偏离是否 > 限额？）。
    *   如果触发，执行交易以再平衡至目标。
    *   记录交易成本和交易细节。

概念上的引擎流程为：

1. 对克隆后的投资组合定价环境施加当日 DayStep 变化
2. 推进估值日期（使到期时间递减，从而体现 Theta/滚降等效应）
3. 全组合重估并计算风险度量（Greeks / DV01）
4. 若启用对冲：运行触发与头寸计算、执行对冲交易、必要时再重估
5. 记录当日结果（价值、P&L、风险、交易与成本）

**[图片占位符]**
> **Prompt for Nanobanana**: /diagram prompt: "A flowchart diagram of the Dynamic Scenario Simulation Loop. Visual flow: Start -> [Update Market Data (t)] -> [Reprice Portfolio] -> [Check Hedge Triggers] -> [Execute Trades] -> [Record Results] -> [Advance Time (t+1)] -> Loop back. Style: Technical process diagram, circular cycle, modern UI elements."

### 路径依赖分析

此框架捕捉了静态分析遗漏的风险。以下是关键路径依赖风险的正式定义：

#### Gamma 损耗 (Gamma P&L)

**Gamma 损耗**是指当标的沿实现路径移动时，Gamma（凸性）产生的累积盈亏。对于 Delta 对冲持仓：

$$
\text{Gamma P\&L} = \sum_{t=1}^{T} \frac{1}{2} \Gamma_{t-1} (\Delta S_t)^2
$$

在**震荡市场**（高已实现波动率）中，Gamma 为正并产生收益。在**趋势市场**中，对冲无法捕获完全的方向性移动。

**预期 Gamma P&L**（基于隐含波动率）与**已实现 Gamma P&L**（基于实际路径）的区别是对冲盈亏的关键来源：

$$
\mathbb{E}[\text{Gamma P\&L}] \approx \frac{1}{2}\Gamma S^2 \sigma_{\text{imp}}^2 \Delta t
$$

$$
\text{已实现 Gamma P\&L} = \frac{1}{2}\Gamma S^2 \sigma_{\text{realized}}^2 \Delta t
$$

#### Theta 损耗（时间衰减）

**Theta 损耗**是由于时间流逝导致的期权价值确定性侵蚀，独立于市场变动：

$$
\Theta_t = \frac{\partial V}{\partial t}\bigg|_{S_t, \sigma_t}
$$

对于 Black-Scholes 下的欧式看涨期权：

$$
\Theta = -\frac{S \phi(d_1)\sigma}{2\sqrt{T-t}} - rK e^{-r(T-t)} \Phi(d_2)
$$

情景跨度内的累积 Theta 损耗：

$$
\text{总 Theta} = \sum_{t=1}^{T} \Theta_t \cdot \Delta t
$$

Theta 通常对**多头期权为负**（时间对您不利），对**空头期权为正**（时间对您有利）。

#### 流动性缺失

当市场变动速度快于再平衡频率时，会发生**流动性缺失**。如果标的价格跳空幅度超过对冲阈值：

$$
|\Delta S_t| > \text{阈值} \implies \text{未对冲敞口}
$$

流动性缺失的预期损失标度为：

$$
\mathbb{E}[\text{跳空损失}] \approx \frac{1}{2}\Gamma \cdot \mathbb{E}[(\Delta S)^2 \cdot \mathbb{I}_{|\Delta S| > \text{阈值}}]
$$

此风险在以下情况下尤为严重：
- 空头 Gamma 持仓（卖出期权）
- 短期期权（临近到期时高 Gamma）
- 流动性差的标的（可能出现大跳空）

## 固定收益动态

对于固定收益投资组合，动态情景涉及整个收益率曲线和关键利率的演变。

*   **平行移动**：整个曲线在此期间上下移动（例如，美联储加息周期）。
*   **扭曲 (Twists)**：短端移动快于长端（熊市变平 / 牛市变陡）。
*   **滚落 (Roll-down)**：随着时间推移，债券沿收益率曲线“滚落”，即使曲线是静态的，其收益率也会变化。模块考虑了这一老化过程。

**滚落收益**是指债券在情景跨度内沿收益率曲线滚落所产生的收益。它计算为债券最终价格与初始价格之差，并调整期间收到的利息。

**[图片占位符]**
> **Prompt for Nanobanana**: /diagram prompt: "A 3D wireframe surface plot representing a Yield Curve evolving over time. Axes: X='Tenor' (1M-30Y), Y='Time' (Day 1-30), Z='Interest Rate'. The mesh surface should visibly twist and shift along the Time axis to demonstrate curve evolution. Style: Technical scientific plot, clean wireframe lines."

## 模型覆盖（QuantArk 实现）

### 引擎架构

动态情景模块分别为**权益**和**固定收益**投资组合实现了引擎：

* **权益引擎** (`DynamicScenarioEngine`)：支持现货、平坦波动率、平坦利率与股息率的逐日变化；可选地接入回测模块中的对冲策略（例如 Delta 中性）并计入交易成本。
* **固收引擎** (`FIDynamicScenarioEngine`)：支持逐日的利率曲线变化（包括平移与简单扭曲路径）；跟踪 DV01/久期/凸性，并可选执行基于 DV01 的对冲。

### 情景构建

模块提供了两个主要接口来构建情景：

* **PathBuilder**：用于构建自定义参数化路径的流式 API。方法包括 `spot_trend()`、`vol_decay()`、`rate_parallel_shift()`、`rate_steepener()` 等。
* **PathLibrary / FIPathLibrary**：预定义情景工厂，包括：
  - 权益：`consecutive_rally()`、`v_shaped_recovery()`、`volatility_spike_decay()`、`gradual_crash()`
  - 固定收益：`parallel_shift()`、`steepener()`、`flattener()`、`rate_hike_cycle()`

### 对冲集成

模块与回测模块的策略框架集成：
- 可使用 `backtest/strategy/` 中的策略进行动态情景
- 引擎调用策略生命周期方法：`on_step()`、`should_hedge()`、`calculate_hedge_size()`、`on_hedge_executed()`
- 交易成本被跟踪并从 P&L 中扣除

### 路径表示

在 QuantArk 中，动态情景由 **DayPath** 表示：它是按天顺序排列的 **DayStep** 列表。每个 DayStep 包含一个或多个 **ParameterChange**（现货/波动率/利率/股息等），并指定冲击类型：

* **百分比（复利）**：\(X_t = X_{t-1}(1+\epsilon_t)\)
* **绝对值（加法）**：\(X_t = X_{t-1}+\delta_t\)
* **覆盖值（设定）**：\(X_t = \bar{X}_t\)

这种表示方式让情景定义具有确定性且易于审计："每一天发生了什么变化，为什么？"

### 扩展点

诸如历史自举路径、完整波动率曲面动力学、粘性 Delta vs. 粘性行权价规则等属于扩展方向：当前实现是"显式参数更新 + 每日全重估"的确定性情景运行器。

## 应用

1.  **前瞻性风险**：估算下季度在"软着陆"与"衰退"情景下的盈亏。
2.  **策略验证**：证明对冲算法不仅在瞬时有效，而且能在动荡的一周内维持表现。
3.  **保证金模拟**：基于投资组合演变估算未来的追加保证金通知。

## 参考文献

*   Heath, D., Jarrow, R., & Morton, A. (1992). "Bond Pricing and the Term Structure of Interest Rates: A New Methodology for Contingent Claims Valuation." *Econometrica*.
*   Hull, J., & White, A. (1990). "Pricing Interest Rate Derivative Securities." *The Review of Financial Studies*.
*   Litterman, R., & Scheinkman, J. (1991). "Common Factors Affecting Bond Returns." *Journal of Fixed Income*.
