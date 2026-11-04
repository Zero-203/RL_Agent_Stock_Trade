# 2016 年冻结上证50：Yahoo 数据下载与指标说明

## 股票池和获取方式

股票池固定为 2016-01-04 的上证50成分，具体代码、名称、数据来源及核验限制见 [成分股名单](sse50_constituents_2016.md)。Yahoo 的 `.SS` 后缀对应上海证券交易所；这不保证每只股票在 Yahoo 的整个历史期间都有数据。退市、合并、更名、停牌或数据源缺失可能影响结果。

在项目根目录运行：

```powershell
python examples/FinRL_StockTrading_SSE50_2016_data.py --ticker 600000.SS
python examples/FinRL_StockTrading_SSE50_2016_data.py --all
```

依赖 `pandas` 和 `yfinance`，项目已将其列入 `requirements.txt`。股票池、起始日、请求间隔和重试次数在 `config/sse50_2016.json` 中配置；可通过环境变量 `PROXY_SERVER` 指定代理。默认从 2016-01-01（含）开始，结束参数为运行时的上海日期的次日（不含），因此可包含运行当日已有的行情，盘中当日数据可能尚未完整。

脚本串行请求，默认每只间隔 5 秒；失败最多尝试 2 次，两次之间等待 30 秒。若某只股票仍失败，脚本停止并返回非零状态；已经写出的单股 CSV 保留，不会生成本次完整汇总文件。再次运行会重新下载并覆盖已成功的单股文件，不会自动跳过已下载股票。旧汇总文件如已存在，不会因本次失败自动删除或更新，需根据运行日志确认其是否为本次结果。

输出在被 Git 忽略的 `datasets/sse50_2016/`：

- 单股：`<Yahoo代码>.csv`，例如 `600000.SS.csv`。
- 汇总：只有 `--all` 的 50 次请求均成功才写出 `sse50_2016_raw.csv`。请求成功只代表返回非空行，不保证每只覆盖全部交易日或所有字段非空。

### 本次下载测试

完整模式在 `600837.SS` 两次请求失败后停止，前 19 只已保存；其余代码随后分别按相同的间隔和重试策略测试。最终 50 只中 48 只返回非空数据，合计 126,192 行，返回日期范围为 2016-01-04 至 2026-11-04；该范围不保证各股票在其中每个交易日均有有效交易。单股文件位于上述目录。

`600837.SS` 和 `601989.SS` 均在重试后返回无时区、可能已退市的提示；`601989.SS` 还出现 HTTP 404。此结果不足以单独确定原因，不能将这两只自动替换或排除；50 只的完整汇总未生成。测试过程中还出现网络超时，部分请求在重试后成功。数据源和网络状态可能变化，后续运行结果可能不同。

### 两只失败股票的进一步核实

- 直接访问 Yahoo 的 `v8/finance/chart` 日线接口，使用 2016-01-01（含）至 2018-01-01（不含）的相同时间范围：`600000.SS` 返回 HTTP 200；`600837.SS` 和 `601989.SS` 均返回 HTTP 404，信息为 `No data found, symbol may be delisted`。这证实本次查询的历史区间不可用，并不能证明 Yahoo 永久不提供其他区间或其他接口的数据。
- [海通证券退市公告报道](https://www.163.com/dy/article/JP8T5NI00514R9P4.html)：600837 于 2025-03-04 终止上市，股份转换为国泰君安股票；[中国重工退市和合并报道](https://www.163.com/dy/article/K8DD1U6905569L3G.html)：601989 于 2025-09-05 终止上市，由中国船舶换股吸收合并。接收方 `601211` 和 `600150` 都已在冻结池中，不应直接替换原代码，否则会重复标的并改变样本含义。
- [BaoStock](https://www.baostock.com/) 的 `query_history_k_data_plus` 对原证券代码、日线、`adjustflag="3"`（不复权）返回了数据：`sh.600837` 有 2,225 行，范围 2016-01-04 至 2025-03-04，最后一条 `tradestatus=1` 为 2025-02-05；`sh.601989` 有 2,352 行，范围 2016-01-04 至 2025-09-04，最后一条 `tradestatus=1` 为 2025-08-12。范围包含停牌记录，最后记录不等于最后交易日，退市后并无原代码日线。
- 口径不可直接混用：2016-01-04 的 `600000` 对照样本，现存 Yahoo CSV 的 `Close` 为 12.447552、`Volume` 为 60,404,072；BaoStock 不复权结果分别为 17.8000、42,240,610。数据源的价格和成交量可能有不同调整规则，不能将 BaoStock 的字段直接当成 Yahoo 的 `Close` 或 `Adj Close`；BaoStock 日线也未验证能提供等价的 Yahoo 分红和拆股列。

已按选择保留现有 Yahoo 文件不变，并将两只 BaoStock 原代码历史分别另存；如需合并成模型数据，还须先单独确定统一价格口径、公司行为和退市后处理规则。不能通过补零、延长最后价格或复制吸收方行情伪造退市后的原股票交易数据。原始 Yahoo 完整汇总仍未生成。

### BaoStock 独立补充文件

在已安装项目依赖的环境中，按需安装单独的可选依赖并运行：

```powershell
python -m pip install baostock==0.9.4
python examples/FinRL_StockTrading_SSE50_2016_baostock.py
```

补充标的、`adjustflag` 和输出目录在 `config/sse50_2016.json` 的 `baostock_supplement` 中配置。脚本使用 BaoStock 日线，起始日期为 `start_date`，结束日期为运行时的上海日期（含）。结果保存在 Git 忽略的独立目录：

| 文件 | 本次结果 | 含义 |
| --- | --- | --- |
| `datasets/sse50_2016/baostock/sh.600837.csv` | 2,225 行 | 2016-01-04 至 2025-03-04；含 36 条停牌记录。 |
| `datasets/sse50_2016/baostock/sh.601989.csv` | 2,352 行 | 2016-01-04 至 2025-09-04；含 156 条停牌记录。 |

CSV 原样保存接口返回的 `date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST` 字段；其中 `adjustflag=3` 表示不复权，`tradestatus` 标记交易或停牌，空字段保留为空。脚本不转换为 Yahoo 列名、不计算 `Adj Close`、不补造分红和拆股列，也不合并或覆盖 Yahoo 数据。查询结果为空或明确报告分页错误时，不会替换原有对应文件；源接口不保证提供完整数据，使用前仍应核验覆盖范围。

## 本脚本的字段

本脚本调用 `yf.Ticker(...).history()`，使用日线、`auto_adjust=False`、`actions=True`、`repair=False` 和 `keepna=True`。它不计算技术指标、不做 FinRL 的价格复权、不填补缺失值、不划分训练集或交易集。下面的“源字段”指 **yfinance 返回的字段**，并非保证未经 Yahoo 或 yfinance 内部处理的交易所逐笔原始数据；Yahoo 的价格历史仍可能受拆股等供应商规则影响。

| 字段 | 类别 | 说明 |
| --- | --- | --- |
| `tic` | 脚本添加 | Yahoo 股票代码，标识每行所属股票。 |
| `Date` | 源字段 | 日线日期索引转为列；通常带上海时区。 |
| `Open`、`High`、`Low`、`Close` | 源字段 | Yahoo/yfinance 返回的开盘、最高、最低和收盘价；本脚本未使用 `Adj Close` 再次复权。 |
| `Adj Close` | 源字段 | Yahoo 返回的调整后收盘价，与 `Close` 分列保留。 |
| `Volume` | 源字段 | Yahoo/yfinance 返回的成交量。 |
| `Dividends`、`Stock Splits` | 源字段 | yfinance 返回的分红、拆股事件；具体列依数据源和版本而定。 |

`keepna=True` 表示请求保留 yfinance 返回的空记录，本脚本也不调用 `dropna` 或 `fillna`；这不保证上游未填充数据，也不保证停牌日或缺失交易日一定存在记录。CSV 输出保留空值和已有行，不构造完整交易日历。

## 与原 DOW30 脚本的区别

原脚本是 [`FinRL_StockTrading_2026_1_data.py`](../examples/FinRL_StockTrading_2026_1_data.py)，下载 DOW30 并使用 `YahooDownloader` 和 `FeatureEngineer`；本脚本仅保存冻结上证50的日线，不直接输出与原脚本相同的模型输入。

| 内容 | 上证50下载脚本 | 原脚本处理后 |
| --- | --- | --- |
| 价格 | 保留 `Open`、`High`、`Low`、`Close` 和 `Adj Close` | `YahooDownloader` 将列名转成小写，按 `Adj Close / Close` 同比例调整 `open`、`high`、`low`、`close`，然后移除 `adjcp`。 |
| 日期、代码、成交量 | `Date`、`tic`、`Volume` | `date` 为 `YYYY-MM-DD` 字符串，`tic` 为代码，`volume` 为成交量；另从日期派生 `day`，周一为 0。 |
| 公司行为 | 保留 yfinance 返回的分红、拆股列 | 原脚本的 `YahooDownloader` 默认调用不要求这些事件列。 |
| 缺失值和股票 | 不主动清洗或构造缺失交易日 | 下载器先 `dropna`；`clean_data` 根据收盘价透视表去掉存在缺失日期的股票；特征工程执行前向及后向填充。之后原脚本构造已有日期与股票的组合并用 0 填充剩余缺失。 |
| 划分 | 不划分 | `data_split` 采用左闭右开区间：训练 `[2014-01-06, 2025-12-31)`，交易 `[2026-01-01, 2026-03-20)`；结果保存为 `train_data.csv` 和 `trade_data.csv`。中间的日期不属于这两份输出。 |

原脚本使用的其他指标均**不是本脚本下载 CSV 中现成的列**：

| 指标 | 来源和含义 |
| --- | --- |
| `macd` | `stockstats` 基于价格计算的指数平滑异同移动平均指标。 |
| `boll_ub`、`boll_lb` | `stockstats` 计算的布林带上轨和下轨。 |
| `rsi_30` | `stockstats` 计算的 30 周期相对强弱指标。 |
| `cci_30` | `stockstats` 计算的 30 周期商品通道指标。 |
| `dx_30` | `stockstats` 计算的 30 周期趋向指标。 |
| `close_30_sma`、`close_60_sma` | `stockstats` 计算的 30、60 周期收盘价简单移动平均。 |
| `vix` | 原脚本另从 Yahoo 下载 `^VIX` 并按日期内连接，取其收盘价；它是外部市场指标，不是上证50价格计算结果。 |
| `turbulence` | FinRL 根据股票收益率、前 252 个交易日的均值和协方差计算的市场动荡指标，初始阶段置零。 |

技术指标计算细节由安装的 `stockstats` 版本决定。原脚本未启用 `user_defined_feature`，因此不会生成该选项对应的 `daily_return`。若后续要将本 CSV 用作 FinRL 输入，需先明确字段映射、复权、缺失数据政策、适合中国市场的风险指标和时间划分，再单独执行特征工程；不能直接把当前 CSV 当作原脚本的处理结果。