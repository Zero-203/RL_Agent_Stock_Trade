# 基于 FinRL 的强化学习股票交易 Agent

深度强化学习课程小组项目：基于 [FinRL](https://github.com/AI4Finance-Foundation/FinRL) v0.3.8 开发的股票交易 DRL Agent。

> **致谢与许可**：本项目基于 AI4Finance Foundation 开源的 FinRL（MIT License，© 2024 AI4Finance Foundation Inc.）二次开发，上游版权与许可声明完整保留于 [LICENSE](LICENSE)。上游原始 README 归档于 [docs/upstream/README_FinRL.md](docs/upstream/README_FinRL.md)，修改说明见 [NOTICE.md](NOTICE.md)。

## 环境搭建

要求：**Python 3.9.x**（已在 3.9.13 + Windows 上验证）。

一键搭建（推荐）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

脚本自动完成：定位 Python 3.9 → 创建 `.venv` → 按 `requirements-lock.txt` 安装锁定依赖 → 以可编辑模式安装本仓库 `finrl` 包 → 从 GitHub 安装 ElegantRL → 冒烟测试。

手动搭建：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip install git+https://github.com/AI4Finance-Foundation/ElegantRL.git
```

说明：

- `requirements-lock.txt` 是本项目的**权威环境**（pip 全量锁定）；上游的 `requirements.txt` / `pyproject.toml` 仅供参考。
- 默认流程使用 stockstats 计算技术指标，**不需要 TA-Lib**；如确需使用请自行安装。
- ElegantRL 从 GitHub 源码安装，网络不可达时请先配置代理或使用镜像。

### Yahoo Finance 代理

教程脚本通过 `finrl.meta.yahoo_utils.get_yahoo_proxy()` 读取环境变量 `PROXY_SERVER` 作为 Yahoo Finance 访问代理。网络受限环境下先设置再运行：

```powershell
$env:PROXY_SERVER = "http://127.0.0.1:7890"   # 按本机实际代理端口修改
```

## 快速开始

`examples/` 下提供三连教程脚本（数据 → 训练 → 回测），按顺序运行：

```powershell
.\.venv\Scripts\Activate.ps1
python examples\FinRL_StockTrading_2026_1_data.py       # 下载 DOW30 数据、计算技术指标、划分数据集
python examples\FinRL_StockTrading_2026_2_train.py      # 训练并保存智能体
python examples\FinRL_StockTrading_2026_3_Backtest.py   # 回测并与基线比较
```

更多示例见 [examples/README.md](examples/README.md)。

## 训练与回测脚本

### T04 统一训练模板

训练入口为 `scripts/train_template.py`，实验参数由 `config/train_model.json` 提供，算法与种子通过命令行指定。数据集按名称放在 `datasets/<DATASETNAME>/` 下，默认输入文件名由 JSON 中的 `data` 配置指定：`train_data.csv`、`val_data.csv` 和 `trade_data.csv`。训练模块只读取训练集和验证集，不读取测试集；如数据集使用其他测试文件名，需在配置中更新 `data.test_file`。

```powershell
python scripts\train_template.py --dataset DOW30_example --algorithm ppo --seed 42
```

支持的算法为 `ppo`、`ddpg`、`td3` 和 `sac`。试运行可通过 `--timesteps` 覆盖总步数，并将 `--run-type` 设为 `pilot` 或 `smoke`；覆盖仅改变总步数，不会自动缩小算法的 rollout/batch 等参数。特别是 PPO，若要缩短实际冒烟时间，应在临时 JSON 配置中同时调小 `n_steps` 等参数。

- **模型**：保存为 SB3 原生 `.zip`，路径为 `trained_models/<DATASETNAME>/<algorithm>/`。文件名简要包含算法、种子、训练步数和主要超参数；完整有效配置、训练/验证数据哈希、环境及运行信息写入 zip 内的 `metadata.txt`。同名模型不会被覆盖。
- **训练日志**：写入 `tensorboard_log/<DATASETNAME>/<algorithm>/<run_id>/`，包含 SB3 CSV/TensorBoard 日志。
- **运行结果**：写入 `results/train_model/<DATASETNAME>/<run_id>/`，包括运行清单、有效配置及验证集账户价值和动作轨迹。当前验证复用 FinRL 的 `DRLAgent.DRL_prediction`；统一指标和正式样本外回测仍属于 T05，不由本训练模块另行定义。

### 后续 T05 回测模型读取约定

T05 回测入口应接收训练完成的 SB3 模型 `.zip` 路径和 `DATASETNAME`，不从文件名推断完整配置。模型 zip 内含 `metadata.txt`：文件以说明行开头，空行后为 JSON；`effective_config` 保存算法、数据集名、种子、训练步数、策略、算法参数、环境参数和数据路径，顶层 `runtime` 记录软件版本等运行信息。回测脚本应读取并校验其中的算法和数据集与本次请求一致。

模型加载复用 FinRL 的 SB3 算法映射与预测接口：`MODELS[algorithm].load(model_path, device=...)` 加载模型，再将配置中 `data.test_file` 指向的测试 CSV 构造成同规格 `StockTradingEnv`，并调用 `DRLAgent.DRL_prediction(model, environment, deterministic=True)` 获取账户价值和动作轨迹。测试集只用于最终样本外回测，不用于训练、调参或选模；验证集选模及统一指标、结果登记由 T05 按项目协议实现。

## 项目结构

> **注意**：本节随项目开发持续修订；最终目录与产物以 T04/T05 联调及项目计划为准。

| 路径 | 说明 |
| --- | --- |
| `finrl/` | FinRL 库源码（上游代码，含 agents / meta / applications 三层） |
| `examples/` | 上游教程脚本与 Notebook（快速开始入口） |
| `docs/` | 上游 Sphinx 文档源；`docs/upstream/` 归档上游 README；`docs/小组项目计划v1.2.md` 为本组完整项目计划 |
| `config/train_model.json` | T04 JSON 实验配置（数据文件名、环境、算法超参数及输出目录） |
| `scripts/` | 小组脚本入口，包括 `setup_env.ps1` 和 `train_template.py` |
| `src/train_model/` | T04 数据加载、环境构造、训练、验证及运行追踪模块 |
| `unit_tests/` | 上游单元测试 |
| `local_test/` | 本地私有测试代码；已忽略，不纳入版本控制 |
| `requirements-lock.txt` | 权威锁定环境（Python 3.9.13 验证） |
| `NOTICE.md` / `LICENSE` | 上游致谢与 MIT 许可证 |
| `results/` `trained_models/` `tensorboard_log/` `datasets/` | 运行产物（不入库，见下节） |

## 复现与数据政策

> **注意**：本节为当前约定，随项目开发（尤其是 T12 独立复现验证的反馈）可能有修订与调整。

- **环境复现**：以 `requirements-lock.txt` + `scripts/setup_env.ps1` 为准，固定 Python 3.9。
- **运行产物不入库**：`results/`、`trained_models/`、`tensorboard_log/`、`temp/`、`datasets/`、`local_test/` 及散落的 `*.csv`/`*.png` 均已加入 `.gitignore`，日常训练产物和本地测试不进版本库。
- **最终定稿例外**：`final_deliverables/` 为最终定稿追踪文件夹，计划仅在最终定稿时一次性提交入库，中途不做细节追踪。
- **数据政策**：使用公开历史数据，本地缓存并可复现；缓存文件不入库，数据处理与版本说明随实验进展补充。
- **敏感信息**：API 密钥等写入 `finrl/config_private.py`（已忽略），不得提交。

## 项目文档

- 完整项目计划：[docs/小组项目计划v1.2.md](docs/小组项目计划v1.2.md)
- 上游 FinRL 文档：[docs/upstream/README_FinRL.md](docs/upstream/README_FinRL.md)、[官方文档站点](https://finrl.readthedocs.io/)

## 免责声明

本项目为课程教学项目，仅供学习与研究使用，不构成任何投资建议。历史回测表现不代表未来收益。


## P2 BaoStock数据与统一回测

当前采用48只固定股票池和复权模拟单位。安装requirements-lock.txt后，在项目根目录运行以下命令生成本地数据。

```powershell
python scripts/prepare_sse48.py --fetch
python scripts/backtest.py check-data --data datasets/sse48_baostock_v2/train_data.csv
```

数据准备说明见[48股BaoStock数据](docs/README_sse48_baostock_v2.md)，模型与回测接口见[统一回测](docs/README_backtest.md)。原始行情、生成CSV、模型、日志及图表存放在Git忽略目录。
