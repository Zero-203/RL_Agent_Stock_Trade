# 48只股票 BaoStock统一来源版本

2026年11月5日用户确认将已选定的48只股票统一使用BaoStock。当前数据集名称为sse48_baostock_v2，模型输入为481维，动作为48维。

## 数据位置与口径

开发目录为D盘VScode Project下的RL_Agent_Stock_Trade_backtest。

- datasets/sse48_baostock_v2保存股票数据、配套指数、数据清单和核验报告。
- config/sse48_baostock_v2_universe.json保存当前48只名单、价格口径、划分与CSV指纹。
- config/backtest_sse48.example.json指向当前数据，模型文件名由P1正式训练后填写。
- scripts/prepare_sse48.py下载并处理本版本数据，src/data_pipeline提供仓库内的处理实现。
- scripts/validate_prepared48.py执行短步数训练和验证集回放。

48只股票固定为2016年初上证50名单排除600837.SS、601989.SS后的集合。价格、成交量、交易状态和复权因子均来自BaoStock。八项技术指标由本地stockstats 0.5.4基于这些价格重新计算，turbulence按48只收益重新计算。

tic沿用600000.SS形式作为模型资产标识。baostock_code记录sh.600000形式的供应商代码。benchmark_code记录上证50代码sh.000016。

## 价格字段

| 字段 | 含义 |
|---|---|
| raw_open、raw_high、raw_low、raw_close | BaoStock不复权日线价格 |
| raw_price_basis | as_traded_unadjusted，描述raw价格 |
| open、high、low、close | 用BaoStock累计后复权因子按2016年起点归一后的价格 |
| price_basis | adjusted_virtual_units，描述模型使用的复权模拟价格 |
| corporate_action_factor | 当日及此前已生效的累计后复权因子，相对研究起点归一 |
| source、price_source、volume_source、factor_source | 供应商与字段来源记录 |
| tradable | 共享训练/回测环境使用的可交易标记 |

当前模拟统一使用close观察、成交和估值。除权除息导致的价格单位变化由BaoStock涨跌幅复权因子处理。实际现金红利、配股认购资金和真实股份结算属于实际账户记账口径。

## 时间划分

| 子集 | 文件 | 日期 | 交易日 | 行数 |
|---|---|---|---:|---:|
| 训练 | train_data.csv | 2017-01-16至2020-12-31 | 965 | 46320 |
| 验证 | val_data.csv | 2021-01-04至2022-12-30 | 485 | 23280 |
| 测试 | trade_data.csv | 2023-01-03至2026-11-03 | 927 | 44496 |

完整日线特征包含2016年开始的预热历史，共126240行。对应指数文件为train_index.csv、val_index.csv、test_index.csv。各子集均覆盖上涨、横盘和下跌三类，沿用60日收益正负5%的标签与连续20交易日规则。

## 已完成的核验

1. 对照下载清单验证48只日线和因子文件SHA256，检查每日证券身份、重复、价格范围及交易日历。
2. 检查所有必需模型数值有限、每个交易日有48只、股票与指数日期一致。
3. 重新计算八项指标和48股turbulence，并做截至2020年末的真实数据前缀一致性检查。
4. 抽查三只股票的CSV浮点往返与指标复算一致性。
5. 对浦发银行、中信证券、贵州茅台各2630个交易日，与BaoStock直接后复权接口归一后的四价逐日核对。最大差异低于0.000000000001。
6. PPO、DDPG、TD3、SAC各128步短训练，并完整回放485个验证交易日。P1/P2账户、现金、费用、持仓与reward对账通过。
7. 首个模型重复验证回放的账户、动作、持仓、成交、指标和阶段文件逐字节一致。

报告位置为数据集内verification_report.json、provider_checks/adjustment_crosscheck.json，以及results/dataset_validation/latest_report.json。source_migration_comparison.json记录从v1到v2的数值差异。模型联调使用单种子7、小网络和小缓存，报告类型为smoke。

测试集校验范围为字段、数值、日期和环境维度。正式训练与最终绩效测试按照P1/P2后续确定的实验预算推进。

## 复现

在项目根目录使用Python 3.9环境运行。

```powershell
python -m pip install -r requirements-lock.txt
python scripts/prepare_sse48.py --fetch
python scripts/backtest.py check-data --data datasets/sse48_baostock_v2/train_data.csv
python scripts/validate_prepared48.py --dataset sse48_baostock_v2
```

下载、清洗、复权、指标、风险变量、指数处理与时间划分均由本仓库代码完成。配置在config/sse48_data_pipeline.json。默认缓存目录为datasets/baostock_cache/sse48_20261103，默认数据集为datasets/sse48_baostock_v2。

缓存可以通过--cache指定其他目录。下载支持已完成文件复用与中断后重试。构建目标已存在时，用--dataset指定新版本名。缓存清单、原始文件和生成数据在本地保存。

数据下载时间和原始文件指纹写入cache_manifest.json，生成数据的指纹写入dataset_manifest.json。供应商可能修订历史数据，程序将新生成CSV与已验证快照的指纹比较结果一并保存。

正式训练使用--dataset sse48_baostock_v2，预算、算法配置和三种子依P1/P2实验协议。模型记录本机数据哈希和有序股票名单。研究报告保留事后选股偏差、复权模拟单位、同日收盘撮合及价格指数基线等设定说明。
