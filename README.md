# Industry Analysis Platform

> 版本 0.3.0 · Fama-French 五因子集成

一个基于 Web 的**行业量化分析与低频交易研究平台**，支持多因子选股、机器学习建模、组合构建与回测评估。

- [完整更新日志](CHANGELOG.md)
- 运行测试：`python -m pytest tests/ -v`（18 个测试用例）

---

## 一、快速开始

```bash
# 安装依赖
pip install fastapi uvicorn pandas numpy scikit-learn lightgbm xgboost

# 启动 Web 服务
uvicorn src.web.app:app --reload --port 8000
```

打开 `http://127.0.0.1:8000`，选择行业和模型类型，点击 Execute 即可运行全流程。

### 数据源

| 模式 | 来源 | 适用场景 |
|------|------|---------|
| `synthetic` | 确定性合成数据（固定随机种子） | 开发、调试、框架验证 |
| `alpha_vantage` | Alpha Vantage API（免费端点） | 真实数据验证 |
| `massive` | Massivest API（财报+OHLCV） | 真实数据验证 |

Alpha Vantage 模式需要你自己的 API Key（免费申请：alphavantage.co）。将 Key 写入 `src/data/alpha_vantage_provider.py` 第 5 行的 `DEFAULT_ALPHA_VANTAGE_API_KEY` 即可。

Massive 模式需要 API Key，写入 `src/config.py` 第 33 行的 `massive_api_key`（或设置环境变量 `MASSIVE_API_KEY`）。当 massive 财报数据不完整时，系统自动用合成基本面数据回填，确保训练流程不中断。

---

## 二、项目架构

```text
数据获取 → 特征工程 → 因子面板合并 → 模型训练 → 组合构建 → 回测 → Web 展示
```

```
src/
├── config.py                         # 全局配置
├── data/                             # 数据提供层
│   ├── base_provider.py              # QueryContext 基类
│   ├── ohlcv_provider.py             # 合成行情
│   ├── alpha_vantage_provider.py     # Alpha Vantage 真实行情
│   ├── massive_provider.py            # Massivest 真实行情+财报
│   ├── fundamentals_provider.py      # 合成基本面（含 FF5 专用字段）
│   ├── macro_provider.py             # 宏观指标
│   └── universe_provider.py          # 行业分类
├── features/                         # 特征工程层
│   ├── price_features.py             # 价格因子（收益、波动）
│   ├── fundamental_features.py       # 基本面因子（质量、增长）
│   ├── ff5_engine.py                 # ★ 五因子引擎（SMB/HML/RMW/CMA/MKT）
│   ├── quantum_business_engine.py    # 量子行业因子（LLM 预留）
│   └── merge.py                      # 面板合并
├── models/                           # 模型层
│   └── industry_low_frequency_models.py  # 6 种模型 + 预处理 + 组合构建
├── backtest/                         # 回测层
│   ├── portfolio.py                  # 组合构建与行业中性化
│   └── simulator.py                  # 绩效计算（含交易成本、滑点）
├── pipelines/                        # 流水线层
│   ├── build_dataset.py              # 数据集构建
│   └── train_industry_model.py       # 完整训练流程
├── web/                              # Web 接口层
│   ├── app.py                        # FastAPI 后端（异步训练 + 进度轮询）
│   └── static/index.html             # 交互式前端
└── tests/                            # 单元测试
    ├── test_ff5_engine.py            # FF5 引擎 7 个测试
    └── test_preprocessing.py         # 预处理 + 流水线 11 个测试
```

---

## 三、数学模型

### 核心理念：截面多因子选股

```
股票预期收益 = Σ w_k × 因子暴露_{k}
```

模型用 `t` 时刻的因子值预测 `t+1` 时刻的收益率，按预测排序取 Top/Bottom 分位构建多空组合。

### 数据预处理

| 步骤 | 方法 | 目的 |
|------|------|------|
| Winsorize 缩尾 | 1%–99% 分位截断 | 消除极端离群值 |
| 行业中性化 | 原始值 − 同行业均值 | 剔除行业系统性偏差 |
| Z-Score 标准化 | z = (x − μ) / σ | 统一量纲，公平惩罚 |

### 可选模型

| 类型 | 算法 | 特点 |
|------|------|------|
| `ridge` | Ridge 回归 (α=1.0) | 线性、保守、可解释 |
| `random_forest` | 随机森林 (300 树, depth=6) | 非线性、灵活 |
| `lightgbm` | LightGBM (300 轮, lr=0.05) | 梯度提升、快速 |
| `xgboost` | XGBoost (300 轮, depth=6) | 梯度提升、精细 |
| `ranker` | LightGBM (300 轮, lr=0.03, num_leaves=41) | 侧重排序区分度 |
| `ensemble` | VotingRegressor (Ridge+RF+LGBM) | 三方集成投票 |

### Fama-French 五因子

基于 Fama & French (2015)，在横截面内按行业做 2×3 分组：

| 因子 | 含义 | 构造方式 |
|------|------|---------|
| MKT | 市场因子 | 市值分位排名 |
| SMB | 规模因子 | 三重排序（B/M、OP、Inv）平均 |
| HML | 价值因子 | 高 B/M（前 30%）− 低 B/M（后 30%） |
| RMW | 盈利因子 | 高营业利润率−低 |
| CMA | 投资因子 | 保守型（低资产增长）− 激进型 |

每年 6 月底重新做分组，其余月份沿用 6 月的分组值（June-end rebalance）。

### 组合构建

每期按预测排序 → 做多前 20% → 做空后 20% → 等权重 → 行业中性化。

### 回测绩效

```
净收益 = 持仓盈亏 − 交易成本 − 滑点
累计净收益 = Π(1 + 净收益) − 1
超额收益 = 净收益 − 基准收益
```

交易成本 5bps + 滑点 3bps，模拟实际交易摩擦。

---

## 四、前端界面

深色金融控制台风格，左右分栏布局。

**左侧配置面板**：行业选择、模型类型、数据源切换、因子表（可勾选启用/禁用、调节权重）、Execute 按钮（点击后出现实时进度条）

**右侧结果展示**：KPI 卡片（累计收益、超额收益、换手率、持仓连续性）、SVG 双线收益图、逐期回测明细表、完整 JSON 原始结果

**进度条**：后端异步训练 + 前端每 600ms 轮询真实进度，显示当前执行阶段。

---

## 五、默认股票池

| 代码 | 行业 |
|------|------|
| NVDA | ai_hardware |
| AMD | ai_hardware |
| MSFT | ai_hardware |
| AVGO | photonics |
| XOM | energy |
| SHEL | energy |
| IONQ | quantum |
| RGTI | quantum |

---

## 六、数据库与 LLM 预留

- `DatabaseConfig` 接口已预留，Provider 的 `fetch()` 实现可替换为 SQL 后端
- `QuantumIndustryBusinessAnalysisEngine` 的 LLM 工作流已预留，当前使用规则引擎

---

## 七、部署

支持 Render 一键部署（`render.yaml` + `Procfile`）：

```yaml
# Build
pip install -r requirements.txt

# Start
uvicorn src.web.app:app --host 0.0.0.0 --port $PORT
```
