# T06 模块说明：同池等权买入持有基线（Equal-Weight Buy-and-Hold）

> 所属任务：**T06 非 DRL 核心基线（P3 / P4 实现，P2 口径核验）**  
> 源码路径：[src/baselines/equal_weight.py](../../src/baselines/equal_weight.py)  
> 入口脚本：[scripts/run_equal_weight_baseline.py](../../scripts/run_equal_weight_baseline.py)  
> 依据：[小组项目计划](../小组项目计划.md) 第 2.6 节，[基础联调交接单 v1](../handoffs/handoff_s0_baseline_integration_v1.md)，[统一回测框架说明](module_backtest.md)

---

## 1. 设计目标与 P2 口径对齐

根据《小组项目计划》第 2.6 节要求，上证 50 价格指数（`sh.000016`）不含分红再投资、不扣交易费用，且与固定 48 股池存在成分差异；因此必须提供严格在同一股票池、同一价格口径、同一交易成本和可交易约束下运行的**同池等权买入持有（Equal-Weight Buy-and-Hold）核心基线**，并直接无缝接入 `scripts/backtest.py run --baseline equal_weight=<CSV>`。

本实现逐项对齐 `src/backtest/runner.py` 与 `StockTradingEnv` 的会计与时间约定：

| 口径维度 | 冻结约定 |
| --- | --- |
| **股票池与价格** | 输入 CSV 中的有序同池股票（如 `sse48_baostock_v2` 固定 48 股），统一使用前复权模拟价格 `close`（`adjusted_virtual_units`）成交与估值 |
| **首行建仓前快照** | 首日 `dates[0]` 记录建仓前初始资金行 `account_value = initial_amount`（`1,000,000`），与 `backtest.data.align_reference` 校验规则完全一致 |
| **成交与估值时序** | 与 FinRL 回测相同：第 $t$ 日收盘价成交，第 $t+1$ 日收盘价开始逐日盯市估值（`dates[0] ... dates[-1]` 全日期无缺失） |
| **资金切分** | 初始资金 $V_0$ 按股票数 $N$ 等分为独立目标额度 $B_i = V_0 / N$（48 股下每只额度约 $20,833.33$） |
| **单位取整（`unit_mode`）** | 默认 `integer_virtual_units`：$q_i = \lfloor B_i / (p_{t,i}(1 + c_{\text{buy}})) \rfloor$，与 `StockTradingEnv` 整数复权模拟单位一致；另支持 `fractional_units` 供理论无取整对照 |
| **首日停牌处理（`suspension_policy`）** | 默认 `defer_reserved_cash`：若股票 $i$ 在首日 `tradable=False`，其额度 $B_i$ 专款保留在现金中，于后续首个 `tradable=True` 交易日（$t \le T-2$）按当日收盘价建仓；若全区间停牌则始终保留为现金。另支持 `keep_cash`（仅首日建仓，停牌份额永久留作现金） |
| **未成交与零头现金** | 整数取整零头现金及未建仓停牌资金留存于账户现金（计 0% 无风险收益），不做跨股挪用、不做中途再平衡、期末不收清仓手续费 |
| **统一评价指标** | 直接复用 `backtest.metrics.curve_metrics` 与 `regime_metrics` 输出期末净值、总收益、年化收益、年化波动、最大回撤、Sharpe，以及成交笔数、累计手续费、`gross_turnover` 与期末现金占比 |

---

## 2. 命令行运行与接入统一回测

### 2.1 生成验证集或测试集等权基线产物

```powershell
conda activate finrl
cd "C:\Users\Haoyu Fan\PycharmProjects\FinRL\RL_Agent_Stock_Trade"

# 生成验证集同池等权基线（可同时传入 val_index.csv 对照上证50指数与市场阶段）
python scripts/run_equal_weight_baseline.py `
  --data datasets/sse48_baostock_v2/val_data.csv `
  --benchmark datasets/sse48_baostock_v2/val_index.csv `
  --train-config config/train_model_p4_sac_tqc.json `
  --split validation `
  --output-name sse48_val_equal_weight
```

运行后在 `results/baselines/sse48_val_equal_weight/` 下生成：
- `account_value.csv`（含首日建仓前初始净值行，可直接传给 `scripts/backtest.py`）
- `holdings.csv`、`trades.csv`、`normalized_curves.csv`、`regime_metrics.csv`
- `metrics.json`、`run_manifest.json`（完整记录输入数据与源码 SHA256 指纹）

### 2.2 在 DRL 模型回测中直接挂载等权基线同图对比

```powershell
python scripts/backtest.py run `
  --model trained_models/sse48_baostock_v2/sac/<YOUR_MODEL>.zip `
  --data datasets/sse48_baostock_v2/val_data.csv `
  --benchmark datasets/sse48_baostock_v2/val_index.csv `
  --baseline equal_weight=results/baselines/sse48_val_equal_weight/account_value.csv `
  --split validation
```

回测生成的 `metrics.json`、`normalized_curves.csv` 与 `equity_curve.png` 会自动将 `strategy`、`equal_weight` 与 `market_index` 放在同一时间轴和指标口径下对比。
