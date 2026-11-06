# P2统一回测框架：复权价格简化实验

当前数据版本为`sse48_baostock_v2`，48只股票的行情、状态和因子均来自BaoStock。交接说明见`docs/README_sse48_baostock_v2.md`，当前名单与指纹见`config/sse48_baostock_v2_universe.json`。


基于P1的`train_module`提交`19b9fc71c3d99c6b48853dea4100395410d54a24`开发。本模块使用同一套复权价格观察、成交和估值，持仓数量是模拟单位；不在此基础上再次派发现金分红、送股或换股权益。不是实盘交易系统。

## 1. 本机入口

```powershell
conda activate stock_agent
cd 'D:\VScode Project\RL_Agent_Stock_Trade_backtest'

# 无需真实行情，验证训练、模型保存、回测和重复运行
python scripts/smoke_backtest.py

# 回测单元测试
python -m unittest discover -s unit_tests -p 'test_backtest.py' -v
python -m unittest discover -s unit_tests -p 'test_data_cache.py' -v
```

依赖以requirements-lock.txt为准，数据获取使用baostock 0.9.4。须先激活Conda；只调用绝对python.exe而未激活时，Windows可能找不到OpenSSL DLL。也可用`conda run -n stock_agent --no-capture-output python ...`。

合成数据测试生成的模型和报告位于被Git忽略的`datasets/backtest_synthetic_*`、`trained_models/backtest_synthetic_*`和`results/backtest_smoke/`。它们用于接口验证，不是正式模型或收益证据。

## 2. 数据检查

```powershell
python scripts/backtest.py check-data `
  --data datasets/sse48_baostock_v2/trade_data.csv `
  --output results/backtest_verification/sse48_test_data_check.json
```

检查失败返回退出码2；执行错误返回1；通过返回0。不会自动删股票、补零或截短区间。

输入每行对应date/tic，至少包含：

`date,tic,open,high,low,close,volume,day`，训练元数据列出的八项指标，以及对应风险变量（默认turbulence）。

- 自动去除历史CSV序号列，按date/tic排序并生成FinRL日索引。
- 同日同股不能重复，每个交易日须有相同股票名单。
- 所需数值必须有限，OHLC必须为正且范围一致，成交量非负，day须与日期星期一致。
- 可提供tradable/is_suspended/is_delisted/feature_available；相互矛盾的标记报错。
- 若含raw_close和corporate_action_factor，检查close是否等于两者乘积。本回测只使用close，raw_*不参与成交。
- CSV用float_precision='round_trip'读取，保持浮点往返精度。

当前48只三份输入的必需数值有限，日期与指数对齐。股票池和价格口径记录于config/sse48_baostock_v2_universe.json。

## 3. 接入P1模型

直接读取SB3 `.zip`内的`metadata.txt`，兼容P1现有“标题+JSON”格式，也可通过`--metadata`指定训练run_manifest/effective_config。

新增训练元数据包含：

- 有序股票名单与指标名单；
- 状态、动作维度；
- 价格口径adjusted_virtual_units、成交时间finrl_same_close；
- 训练与验证日期边界；
- 共享环境及原FinRL环境源代码SHA256。

模型的观察/动作空间、股票身份和顺序必须与回测数据相同；相同维数但股票不同也拒绝。旧模型元数据没有股票名单时，必须能找到原训练CSV并核对哈希；移动过文件可用`--train-data`、`--validation-data`指定原文件新位置。

验证模式只接受训练记录中的原验证CSV哈希。测试模式要求日期晚于验证期，且不能把训练/验证文件换个名字充作测试集。回测只predict，不learn，不根据测试结果选模型。

## 4. 训练和回测共享环境

`src/trading_environment.py`提供两种明确策略：

- `legacy_finrl`：保留原FinRL交易规则。若数据存在tradable=False，拒绝忽略这些标记运行。
- `enforce_column`：增加tradable过滤，买卖都遵守，包括风险触发的强制卖出；其他执行规则沿用FinRL。

训练端`train_model.runner.build_environment`已改为导入此共享构造器；回测调用同一函数实现。P1原调用接口和模型ZIP结构保留。使用有停牌标记的数据训练时，可复制`config/train_model_adjusted.example.json`并冻结为正式配置。

不能把旧的legacy模型只在回测时换成enforce_column，然后宣称训练与测试环境一致。新模型记录环境源文件哈希，代码变化时明确拒绝静默回放。

P1训练入口还修复了policy_kwargs的独立透传。DDPG/TD3/SAC验证与回测加载时只创建1格经验缓存，避免模型加载就按训练buffer_size申请大量内存；不会修改网络权重，且不进行学习。

## 5. 单模型回测

以下YOUR_*是待替换路径，不能直接照抄为正式实验：

```powershell
python scripts/backtest.py run `
  --model trained_models/YOUR_DATASET/ppo/YOUR_MODEL.zip `
  --data datasets/YOUR_DATASET/trade_data.csv `
  --benchmark datasets/YOUR_DATASET/test_index.csv `
  --split test `
  --output-root results/backtest
```

验证期使用`--split validation --data .../val_data.csv`。默认CPU、deterministic=True、单线程；初始资金、费用、最大交易单位、指标顺序和风险阈值从模型元数据恢复，不在回测端重新手填。

每次生成独立目录，不覆盖已有回测。每个模型从同一份零持仓、初始现金开始。

## 6. 基线接入与批量运行

基线实现由P3/P4负责，本模块提供统一评价接口，不另造一套同池等权算法。输入基线CSV支持：

- `date,account_value`：首行必须为交易前初始资本；
- `date,nav`：首行必须为1；
- `date,close`：指数点位，归一到同一初始资本。

评估区间内的日期必须完整且与策略逐日相同，不用inner join偷偷丢掉缺失日期。可以提供更长历史，但明确裁切到本次评估起止日期。指数可附regime列供阶段统计。

```powershell
python scripts/backtest.py run `
  --model trained_models/YOUR_DATASET/ppo/YOUR_MODEL.zip `
  --data datasets/YOUR_DATASET/trade_data.csv `
  --benchmark datasets/YOUR_DATASET/test_index.csv `
  --baseline equal_weight=results/baselines/equal_weight.csv

python scripts/backtest.py batch --config config/backtest.example.json

python scripts/backtest.py summarize `
  --manifests results/backtest/RUN1/run_manifest.json results/backtest/RUN2/run_manifest.json `
  --output results/backtest/summary.csv
```

batch按显式任务表串行执行。先将example配置中的YOUR_*替换为真实路径。汇总检查数据、环境、阶段、smoke/main类型、年化参数一致；同算法/种子的重复运行拒绝直接混进统计。输出逐次结果及各算法count/mean/std，样本量1时不伪造标准差。

## 7. 结果文件

| 文件 | 内容 |
|---|---|
| account_value.csv | 每日现金、持仓价值、总资产、累计费用、净值、日收益、回撤、风险暴露 |
| actions.csv | 模型动作、截整后的请求单位、实际执行单位、执行日期、下一估值日期、差异原因 |
| holdings.csv | 每日每股的模拟单位、复权计价、持仓价值 |
| trades.csv | 实际成交的顺序、方向、单位、价格、费用、成交前后现金与持仓 |
| metrics.json | 策略及各基线共用的评价指标 |
| regime_metrics.csv | 按指数阶段归属的日收益统计 |
| normalized_curves.csv | 策略与各基线按日期对齐的净值 |
| equity_curve.png / drawdown.png | 净值与回撤图 |
| model_metadata.json / run_manifest.json | 模型元数据、输入输出哈希、版本、参数、限制和完成/失败状态 |

所有shares字段均表示复权模拟单位，不是交易所的真实股数。策略初始行在首日交易前，最后一天仅估值，不额外做一次无意义的终止交易。每步核对现金变化、费用、持仓、资产变化和reward。

## 8. 指标定义与限制

- 收益按含费用后的账户曲线计算；累计收益=末值/初值-1。
- 年化收益使用实际收益区间数N-1及252日假设；年化波动使用日收益样本标准差ddof=1。
- Sharpe使用年化无风险利率转换后的日无风险收益；默认0。标准差为0或收益区间不足时返回null。
- 最大回撤为account_value/运行最高值-1的最小值，报告为非正数。
- gross_turnover=累计买卖成交金额/初始资本，不能与按平均资产计算的换手指标混用。
- 阶段统计先算完整日收益，再按当日指数regime归属。conditional_compounded_return是在不连续阶段日上归属收益的连乘，不是重新运行的连续策略；不计算混拼阶段的最大回撤或CAGR。
- 沿用FinRL“t日复权收盘价成交、t+1日估值”的简化；观察中包含t日收盘信息，不能声称为严格可执行的次日开盘回测。
- 未模拟真实T+1持仓批次、涨跌停排队、整手、滑点及公司行为现金/股份结算。指数为不含分红再投资、也不扣交易费的外部价格指数参照。

## 9. 验证记录

- 14项单元测试通过：现金/费用/持仓/reward对账、停牌过滤及风险强平过滤、不能卖空、相同维数不同股票拒绝、环境版本检查、验证/测试边界、旧元数据哈希、非有限值拒绝、基线日期、已知指标、阶段收益归属、P1网络参数透传。
- 合成数据PPO训练64步，经P1保存/加载后回测；与P1原验证流程的逐日账户曲线最大差为0。
- 同一个模型重复回测，账户、动作、持仓、成交和指标文件字节一致。
- TD3模型即使声明训练缓存1,000,000格，回测只分配1格缓存。
- batch CLI、外部现金基线、模型汇总及图片输出已实际运行检查。
- 实际48只股票数据完成四算法各128步训练及485日验证回放，账户对账通过；正式测试集用于结构校验。

详见`results/backtest_smoke/latest_smoke_report.json`和`results/backtest_verification/`。

## 10. 从仓库生成数据并交接

在项目根目录运行。prepare_sse48.py与src/data_pipeline包含下载和处理的完整实现，输入配置均位于本仓库。

```powershell
python -m pip install -r requirements-lock.txt
python scripts/prepare_sse48.py --fetch
python scripts/backtest.py check-data --data datasets/sse48_baostock_v2/train_data.csv
python scripts/backtest.py check-data --data datasets/sse48_baostock_v2/val_data.csv
python scripts/backtest.py check-data --data datasets/sse48_baostock_v2/trade_data.csv
python scripts/validate_prepared48.py --dataset sse48_baostock_v2
```

下载缓存使用相对路径清单，已完成文件通过SHA256复用，连接失败后重新运行会继续缺少的请求。数据生成写入独立暂存目录，通过检查后发布。再次构建时用--dataset指定新的版本名。

config/sse48_data_pipeline.json固定原始区间、指标预热、划分和阶段规则。config/sse48_baostock_v2_universe.json提供48股名单及已验证快照的CSV指纹。新下载结果与参考快照的指纹比较写入dataset_manifest.json，便于识别供应商历史数据修订。

训练端使用scripts/train_template.py，选择--config config/train_model_adjusted.example.json与--dataset sse48_baostock_v2，并按实验协议填写算法、种子和正式训练预算。P1/P2共同使用src/trading_environment.py和模型元数据契约，P3/P4通过统一指标接口交接基线。

源代码、配置、字段说明和测试用于Git交接。datasets、results、trained_models和tensorboard_log保存本地行情、模型与运行产物，已由.gitignore管理。
