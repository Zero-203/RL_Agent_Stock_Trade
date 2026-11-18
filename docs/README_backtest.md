# P2统一回测框架：复权价格简化实验

新增算法加载、循环状态和门限调用的约定见 [P1 与 P2 回测交接契约 v1](P1_P2_回测交接契约_v1.md)；训练侧接口见 [P1 接口契约 v1](P1_接口契约_v1.md)。

P2 本次需求、接入约定和联调结果见 [P2 回测更新交付](P2_回测更新交付_v1.md)。

当前数据版本为`sse48_baostock_v2`，48只股票的行情、状态和因子均来自BaoStock。交接说明见`docs/README_sse48_baostock_v2.md`，当前名单与指纹见`config/sse48_baostock_v2_universe.json`。


当前更新基于 P1 的 `train_module` 提交 `d51a0771906de109fc95ac48adda463e9aaa5d0a`，按项目计划 v2.0 接入六算法白名单、循环状态及推理门限。本模块使用同一套复权价格观察、成交和估值，持仓数量是模拟单位；不在此基础上再次派发现金分红、送股或换股权益。不是实盘交易系统。

## 1. 本机入口

```powershell
conda activate stock_agent
cd 'D:\VScode Project\RL_Agent_Stock_Trade_backtest'

# 无需真实行情，验证训练、模型保存、回测和重复运行
python scripts/smoke_backtest.py

# 回测单元测试
python -m unittest discover -s unit_tests -p 'test_backtest.py' -v
python -m unittest discover -s unit_tests -p 'test_data_cache.py' -v
python -m unittest discover -s unit_tests -p 'test_interface.py' -v
python -m unittest discover -s unit_tests -p 'test_backtest_thresholds.py' -v

# 四算法真实短训练及合成验证集双路径对账
python scripts/smoke_backtest_interfaces.py
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

P1训练入口还修复了policy_kwargs的独立透传。DDPG/TD3/SAC/TQC 验证与回测加载时只创建1格经验缓存，避免模型加载就按训练buffer_size申请大量内存；不会修改网络权重，且不进行学习。PPO/RecurrentPPO 加载不传 buffer_size；RecurrentPPO 在每次独立回放从空隐藏状态开始，逐日传递 state 和 episode_start。

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

### 5.1 推理门限入口（P2 v1）

| 入口 | 默认值 | 含义 |
|---|---|---|
| `--threshold-mode` / `threshold_mode` | `none` | 仅本次推理使用 `none`、`simple` 或 `critic` |
| `--threshold-config JSON_FILE` | 无 | CLI 从 UTF-8 JSON 对象文件读取参数，支持 BOM |
| `run_backtest(..., threshold_config={...})` | `{}` | Python 接口和 batch 任务直接传 JSON 对象 |

门限完全由回测入参决定，不从训练元数据自动启用。`none` 不允许带非空参数，未知模式、非对象参数和 NaN/Infinity 均报错。目前 `simple`/`critic` 的正式实现由 P3/P4 提供，本构建明确抛 `NotImplementedError`，并在已建立的运行目录记录 failed；不能将其当作有效门限结果。

以下是 P3 交付实现后使用的接口示例，`q_min=5` 只是示例，正式值须按验证集选择协议冻结：

```powershell
python scripts/backtest.py run `
  --model trained_models/YOUR_DATASET/sac/YOUR_MODEL.zip `
  --data datasets/YOUR_DATASET/val_data.csv --split validation `
  --threshold-mode simple --threshold-config config/YOUR_THRESHOLD.json
```

其中 JSON 文件内容为 `{"q_min": 5}`。批量配置的 `defaults` / `jobs` 使用 `"threshold_mode": "simple", "threshold_config": {"q_min": 5}`；任务字段覆盖默认字段，参数对象整体替换。三条件验证模板见 `config/backtest_thresholds.example.json`，切勿把尚未交付的门限当作已通过联调。

每次 `apply_threshold` 调用接收一维 float32 原动作，`obs`、`model`、`critic`、`hmax`、`ordered_tickers`、`algorithm` 作为上下文；返回 `(action, info)`。返回动作必须为 `(n,)`、有限、严格位于 `[-1,1]`。简单门限的 `info.accepted` 必须为 `(n,)` 布尔数组；Critic 门限必须为标量布尔值。拒绝位置只能归零，放行位置必须保留原动作。回测再次校验，不静默 reshape 或 clip；模型原动作仅额外接受单批维 `(1,n)`。门限前后保持 float32，使乘 hmax 和向零取整与训练环境一致。

输出目录格式为 `{algorithm}_s{seed}_{split}_{mode}-{condition_hash前12位}_{随机6位}`。清单记录 `threshold_mode`、`threshold_config`、`threshold_contract_version`、`threshold_implementation_sha256`、完整 `threshold_condition_sha256` 和便于阅读的 `threshold_condition_id`。条件哈希由契约版本、模式、规范化参数与实现哈希生成，不包含模型或种子；当前实现哈希覆盖整个 `src/backtest/runner.py`，有门限模式在运行结束时复核该文件未变化。门限拆成独立模块后须把其实际源码加入实现指纹，不得只记录调用壳的哈希。

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

batch按显式任务表串行执行。先将example配置中的YOUR_*替换为真实路径。汇总检查 split、数据哈希、环境配置及源码哈希、外部参照哈希、smoke/main 类型、年化参数一致；不允许混入其他口径。

逐次表保留模型哈希、清单路径及全部门限条件字段。去重键为 `(algorithm, seed, threshold_condition_sha256)`；分组键为 `(algorithm, threshold_mode, threshold_condition_sha256)`，CSV 同时展示条件短标识。允许同一模型的 none/simple/critic 及不同候选参数，重复条件仍报错。同算法同种子跨条件必须复用同一模型 SHA256，否则拒绝作为配对实验混入。旧无门限清单补推 `none/{}`，有门限清单必须具有实现哈希并通过条件身份复核。

输出 `summary.csv` 和 `summary_by_algorithm.csv`；后者保留原文件名，但现在按算法及门限条件分别统计 count/mean/std。每项指标只统计非空值，Sharpe 的 null 不当作零，样本量1时标准差为空。该汇总提供各条件描述统计，不自动完成验证集门限选择或 P6 的逐模型配对差值分析。

## 7. 结果文件

| 文件 | 内容 |
|---|---|
| account_value.csv | 每日现金、持仓价值、总资产、累计费用、净值、日收益、回撤、风险暴露 |
| actions.csv | 模型原始动作、门限后动作、门限模式、截整后的请求单位、实际执行单位、执行日期、下一估值日期、差异原因 |
| holdings.csv | 每日每股的模拟单位、复权计价、持仓价值 |
| trades.csv | 实际成交的顺序、方向、单位、价格、费用、成交前后现金与持仓 |
| metrics.json | 策略及各基线共用的评价指标 |
| regime_metrics.csv | 按指数阶段归属的日收益统计 |
| normalized_curves.csv | 策略与各基线按日期对齐的净值 |
| equity_curve.png / drawdown.png | 净值与回撤图 |
| model_metadata.json / run_manifest.json | 模型元数据、输入输出哈希、版本、参数、限制和完成/失败状态 |

`actions.csv` 固定包含 `threshold_accepted`：simple 为逐资产是否放行，critic 为整向量决定重复到各资产行，none 恒为 true。原动作本来为零仍可能被门限拒绝；放行也不代表一定成交。可选 `critic_policy_value`、`critic_zero_value`、`critic_delta` 为 P4 提供的有限标量，未提供时留空，不在 P2 另算价值。

`metrics.json` 额外记录 `threshold_asset_rejection_rate`（所有日期×资产行中拒绝的比例）与 `threshold_vector_rejection_rate`（只对 Critic，按决策日统计；其他模式为 null）。拒绝比例包含原本零动作，不能与减少成交次数等同。环境请求仍为门限后动作乘 hmax 后向零取整，风险强平可以覆盖该请求产生卖出。

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

- 23项单元测试通过（`test_backtest.py`、`test_data_cache.py`、`test_interface.py`）：现金/费用/持仓/reward对账、停牌过滤及风险强平过滤、不能卖空、相同维数不同股票拒绝、环境版本检查、验证/测试边界、旧元数据哈希、非有限值拒绝、基线日期、已知指标、阶段收益归属、网络参数透传，以及新增算法注册、按算法配置校验、on/off-policy加载分类、LSTM状态传递/重置与门限未实现报错。
- 合成数据PPO训练64步，经P1保存/加载后回测；与P1原验证流程的逐日账户曲线最大差为0。
- 同一个模型重复回测，账户、动作、持仓、成交和指标文件字节一致。
- TD3模型即使声明训练缓存1,000,000格，回测只分配1格缓存。
- batch CLI、外部现金基线、模型汇总及图片输出已实际运行检查。
- 实际48只股票数据完成四算法各128步训练及485日验证回放，账户对账通过；正式测试集用于结构校验。

2026-11-16 更新：上述 23 项原测试及新增 12 项门限接口回归共 35 项通过。`scripts/smoke_backtest_interfaces.py` 对 PPO/RecurrentPPO/SAC/TQC 各完成 16 步真实合成数据训练，SAC/TQC 各发生 12 次梯度更新；与 P1 的逐日验证资产最大差为 0，成交单位一致；五类核心结果重复回放字节一致。本次不读取正式测试集绩效，不代表正式门限策略有效性。

P1 当前保存 `validation_actions.csv` 时丢弃了日期索引。合成联调以已核对的账户日期和动作行顺序比对，并在报告标记；建议 P1 保存时显式保留 date。详见交付文档中的反馈。

历史结果见 `results/backtest_smoke/latest_smoke_report.json` 和 `results/backtest_verification/`；本次四算法报告见 `results/interface_smoke/latest_report.json`，均为本地产物。

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
