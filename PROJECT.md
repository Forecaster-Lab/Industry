# Industry Analysis Platform — 项目技术文档

> 版本：0.3.0 · Fama-French 五因子集成版  
> 仓库：[Forecaster-Lab/Industry](https://github.com/Forecaster-Lab/Industry)

---

## 一、项目逻辑：从源头到输出的完整链路

### 1.1 总览

```
数据获取 → 特征工程 → 因子面板合并 → 模型训练 → 组合构建 → 回测 → 结果展示
```

### 1.2 第一层：数据获取

系统支持两种数据源：

| 模式 | 来源 | 说明 |
|------|------|------|
| `alpha_vantage` | Alpha Vantage API（真实行情） | 8 只默认股票，免费 Key 每天 25 次调用 |
| `synthetic` | 合成数据 | 确定性随机数生成，固定种子，可复现 |

**数据提供者（Provider）各司其职：**

| Provider | 产出 | 核心字段 |
|----------|------|---------|
| `OHLCVProvider` / `AlphaVantageProvider` | 日线行情 | date, ticker, close, volume |
| `FundamentalsProvider` | 基本面 | book_equity, operating_profit, total_assets, market_cap, pe_fwd, gross_margin_ttm, fcf_yield 等 16 个字段 |
| `MacroProvider` | 宏观指标 | gdp_growth, cpi_yoy, interest_rate, vix, oil_price 等 |
| `UniverseProvider` | 行业分类 | ticker → industry（ai_hardware / energy / photonics / quantum） |

### 1.3 第二层：特征工程

**价格因子**（`make_price_features`）：

原始 OHLCV → 收益率 + 波动率：

| 因子 | 计算方式 |
|------|---------|
| `ret_1m / ret_3m / ret_6m / ret_12m` | 收盘价的 N 期百分比变化（pct_change） |
| `volatility_1m / volatility_3m` | 月收益率的滚动标准差 |

**基本面因子**（`make_fundamental_features`）：

除了直接使用 provider 的原始字段外，还生成两个复合因子：

| 复合因子 | 公式 |
|---------|------|
| `quality_value_blend` | gross_margin_ttm + fcf_yield − 0.3 × debt_to_equity |
| `growth_capex_cycle` | rev_growth_yoy + 0.5 × capex_ratio − inventory_growth |

**Fama-French 五因子**（`FF5Engine`）：

基于账面权益、营业利润率、总资产增长率，在行业内做横截面分位分组，生成每个股票的因子暴露得分：

| 因子 | 含义 | 分组逻辑 |
|------|------|---------|
| `mkt_exposure` | 市场因子 | 市值的分位排名 |
| `smb_exposure` | 规模因子 | 三重排序（B/M、OP、Inv）平均 |
| `hml_exposure` | 价值因子 | 高账面市值比（前 30%）− 低（后 30%） |
| `rmw_exposure` | 盈利因子 | 高营业利润率（前 30%）− 低（后 30%） |
| `cma_exposure` | 投资因子 | 保守型（低资产增长，后 30%）− 激进型（前 30%） |

**量子行业因子**（`QuantumIndustryBusinessAnalysisEngine`）：

针对 quantum 行业的 13 个定制因子：`upstream_exposure`, `platform_exposure`, `application_exposure`, `pqc_exposure`, `partnership_count`, `contract_score`, `government_dependency_score`, `commercialization_stage_score`, `technology_bottleneck_score`, `capex_cycle_score` 等。当前用规则引擎生成，LLM 工作流接口已预留。

### 1.4 第三层：面板合并

```
price_features + fundamental_features + macro + universe + quantum_panel + ff5_panel
                        ↓
              merge_feature_panels()
                        ↓
              panel（所有因子 + industry + future_return）
```

合并后每行是一个（日期, 股票）的完整因子画像，附带了行业标签和未来一期收益率（`future_return`）作为训练目标。

### 1.5 第四层：模型训练

`train_industry_model` 按行业过滤数据 → 构建行业专属模型 → 训练 → 预测：

```
panel[industry == "ai_hardware"]
       ↓
BaseLowFrequencyTradeModel
  ├── prepare_features()  → Winsorize(1%-99%) → 行业中性化 → Z-Score 标准化
  ├── build_model()       → Ridge / RandomForest / LightGBM / XGBoost / Ensemble
  ├── fit(train_df)       → 用 future_return 作为目标训练
  └── predict(train_df)   → 生成 prediction（预期收益）
```

### 1.6 第五层：组合构建

```
prediction
    ↓
construct_portfolio()
    ├── 按日期分组
    ├── 每日期内：按 prediction 降序排列
    ├── Top 20%（前 20%）→ 多头，等权重 1/n_top
    ├── Bottom 20%（后 20%）→ 空头，等权重 -1/n_bottom
    ├── 其余 60% → 平仓（权重 0）
    └── apply_industry_neutral() → 行业内部去均值，消除行业偏差
```

### 1.7 第六层：回测

```
portfolio（每日期 × 每股票的权重）
    ↓
backtest()
    ├── turnover_component = |本期权重 − 上期权重|
    ├── gross_pnl = 权重 × 未来实际收益
    ├── trading_cost = turnover × (交易成本 + 滑点) / 10000
    ├── net_pnl = gross_pnl − trading_cost
    ├── excess_return = net_return − benchmark_return
    └── cum_net_return = (1 + net_return) 的累积乘积 − 1
```

### 1.8 最终输出

| 输出项 | 说明 |
|--------|------|
| `metrics` | last_cum_return、last_cum_excess_return、avg_turnover、avg_holding_continuity |
| `model_graph` | 逐期 cum_net_return 和 cum_excess_return 序列（用于前端绘制 SVG 折线图） |
| `backtest` | 逐期明细：date / net_return / excess_return / turnover / cum_net_return / cum_excess_return |
| `feature_columns` | 该模型使用的所有特征列名 |
| `config` | 模型配置的完整快照 |

---

## 二、前端页面 UI 说明

页面采用**深色金融控制台风格**，左右分栏布局。

### 2.1 顶栏

| 元素 | 作用 |
|------|------|
| **Industry Analysis Platform** | 应用标题 |
| **Runtime 徽章** | 运行状态：Idle / Running... / Completed / Error |
| **DB Interface 徽章** | 数据库连接状态，当前显示 Reserved（预留接口） |

### 2.2 左侧配置面板

| UI 组件 | 作用 |
|---------|------|
| **Industry 下拉框** | 选择要分析的行业：ai_hardware / energy / photonics / quantum。切换时下方因子表自动刷新 |
| **Model Type 下拉框** | 选择模型算法：ridge / random_forest / lightgbm / xgboost / ranker / ensemble |
| **Data Source 下拉框** | alpha_vantage（真实数据）或 synthetic（合成数据） |
| **Top Quantile / Bottom Quantile** | 组合构建参数：做多前 N%、做空后 N% 的股票，默认各 20% |
| **Tickers 文本框** | 手动输入股票代码（逗号分隔）。留空则用默认池 |
| **Factor Table（因子表）** | 三列：Use（勾选启用/禁用）、Factor（因子名）、Weight（权重）。每个因子可独立控制。五因子（mkt/smb/hml/rmw/cma）已集成在此 |
| **Feature Columns 文本框** | 当前行业的所有特征列，自动同步因子勾选状态 |
| **Execute 按钮** | 点击后触发全流程（数据→特征→训练→回测），下方出现进度条 |
| **Reset Defaults** | 恢复该行业的默认因子选择和权重 |

### 2.3 右侧结果展示

| 区域 | 展示内容 |
|------|---------|
| **Performance Snapshot** | 4 张 KPI 卡片：Last Cum Net（累计净收益）、Last Cum Excess（累计超额收益）、Avg Turnover（平均换手率）、Hold Continuity（持仓连续性） |
| **Model Graph** | SVG 双线图：蓝线 = 累计净收益，橙线 = 累计超额收益 |
| **Backtest Timeline** | 最近 20 期逐期回测明细表 |
| **Raw API Result** | 后端返回的完整 JSON 原始数据 |

### 2.4 进度条

点击 Execute 后按钮下方出现进度条，分阶段动画推进：

> Initializing → Fetching data → Computing features & FF5 factors → Training model → Constructing portfolio → Running backtest → Done

成功时绿色 100%，失败时红色并显示错误信息。

---

## 三、数学模型框架详解

### 3.1 输入数据

模型接收一个**面板数据矩阵**（Panel Data），结构如下：

```
每行 = (日期, 股票) 的完整因子画像
```

| 维度 | 典型值 |
|------|--------|
| 股票数量 | 8 只（默认，可扩展） |
| 时间跨度 | 24 个月 |
| 特征列数 | 10–19 列（因行业而异） |
| 总样本量 | ~192 行（8 × 24） |

**训练目标**：`future_return` = 该股票下个月的收益率（`ret_1m` 向前平移一期）。

### 3.2 数据预处理：三个标准化步骤

按日期分组做截面处理，每个步骤消除不同维度的噪声：

#### 步骤一：Winsorize（缩尾）

```
对每个因子列，按每日横截面：
    下界 = 该日第 1 分位数
    上界 = 该日第 99 分位数
    超过界限的值 → 截断到边界值
```

**目的**：消除极端离群值对模型的影响。

#### 步骤二：行业中性化（Industry Neutralization）

```
对每个因子列 × 每个日期：
    原始值 − 同行业同日均值
```

**目的**：剔除行业层面的系统性偏差（例如 AI 硬件行业的 PE 天然比能源行业高），使因子反映的是**行业内个股之间的相对差异**，而非行业间的差异。

#### 步骤三：Z-Score 标准化

```
对每个因子列 × 每个日期：
    z = (x − μ) / σ
    其中 μ = 该日截面均值，σ = 该日截面标准差
```

**目的**：将不同量纲的因子统一到同一尺度（均值为 0、标准差为 1），使 Ridge 回归的正则化惩罚对各因子公平。

### 3.3 模型构建：可选 6 种算法

所有算法均通过 scikit-learn 的 `Pipeline` 封装，统一流程：

```
SimpleImputer（中位数填补缺失值）→ StandardScaler（再次标准化）→ 模型
```

| model_type | 底层算法 | 关键参数 |
|-----------|---------|---------|
| `ridge` | Ridge 回归 | α = 1.0（L2 正则化） |
| `random_forest` | 随机森林回归 | 300 棵树，max_depth=6，min_samples_leaf=8 |
| `lightgbm` | LightGBM 回归 | 300 轮，learning_rate=0.05，num_leaves=31 |
| `xgboost` | XGBoost 回归 | 300 轮，max_depth=6，learning_rate=0.05 |
| `ranker` | Ridge 回归 | α = 0.6（更弱正则化，侧重排序） |
| `ensemble` | 随机森林回归 | 150 棵树，max_depth=5 |

### 3.4 核心数学思想

这个项目使用的是**截面多因子选股模型**，属于量化金融中最经典的方法论。其数学本质是：

**假设 1：因子暴露驱动收益**

每只股票的预期收益可以分解为多个因子暴露的线性组合：

```
E[r_i] = Σ w_k · f_{i,k}
```

其中 `f_{i,k}` 是股票 i 在第 k 个因子上的暴露（标准化后的值），`w_k` 是因子权重（由模型学习）。

**假设 2：历史因子暴露能预测未来收益**

模型用 `t` 时刻的因子值去预测 `t+1` 时刻的收益率，即：

```
r_{i, t+1} = g( f_{i,1,t}, f_{i,2,t}, ..., f_{i,K,t} ) + ε
```

其中 `g` 是模型学习到的映射函数（Ridge 是线性，RandomForest 是非线性集成）。

**假设 3：横截面相对排序比绝对预测重要**

模型训练的目标不是精确预测每只股票的绝对收益，而是在每期内**正确排序**股票。因为组合构建是按排序取 Top/Bottom 分位，排序质量直接决定策略表现。

**Fama-French 五因子的贡献：** 在传统价量/基本面因子之外，引入学术上被广泛验证的系统性风险因子（规模 SMB、价值 HML、盈利 RMW、投资 CMA），让模型能捕捉市场定价的结构性规律。

### 3.5 组合构建：量化多空策略

```
每期 t：
    1. 取 prediction 降序排列
    2. 前 P%  → 多头，权重 1/n_top（等权重）
    3. 后 Q%  → 空头，权重 -1/n_bottom
    4. 其余   → 不持仓（权重 0）
    5. 行业中性化 → 每组内部去均值，消除行业偏移
```

默认 P = Q = 20%，即做多前 20% 股票、做空后 20% 股票。

### 3.6 回测绩效计算

逐日计算：

```
gross_pnl_t = Σ_i weight_{i,t} × future_return_{i,t}        （总盈亏）
trading_cost_t = Σ_i |weight_{i,t} − weight_{i,t-1}| × cost_rate  （交易成本）
net_pnl_t = gross_pnl_t − trading_cost_t                     （净盈亏）
excess_return_t = net_pnl_t − benchmark_return_t             （超额收益）
```

累积收益采用几何累乘：

```
cum_net_return_T = ∏_{t=1}^{T} (1 + net_pnl_t) − 1
```

交易成本 `cost_rate = (交易成本 5bps + 滑点 3bps) / 10000`，考虑实际交易的摩擦成本。

---

## 附录：默认股票池与行业分布

| 代码 | 行业 | 说明 |
|------|------|------|
| NVDA | ai_hardware | NVIDIA |
| AMD | ai_hardware | AMD |
| AVGO | photonics | Broadcom（光电子） |
| MSFT | ai_hardware | Microsoft |
| XOM | energy | ExxonMobil |
| SHEL | energy | Shell |
| IONQ | quantum | IonQ |
| RGTI | quantum | Rigetti Computing |
