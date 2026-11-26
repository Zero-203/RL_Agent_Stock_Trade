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
- 锁文件固定 Windows、Python 3.9 的 `torch==2.8.0+cu126`，并在 pip 配置的默认索引之外增加 PyTorch 官方 CUDA 12.6 索引。PyTorch 安装包约 2.9 GB，需要网络和足够磁盘空间。该版本也支持 CPU 运算；使用 GPU 时需兼容的 NVIDIA 驱动。
- 安装后可运行 `python -m pip check`，并通过 `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` 检查版本和 GPU 可用性。CUDA 可用性取决于本机环境，不代表训练链路已验证。
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

支持的算法为 `ppo`、`ddpg`、`td3`、`sac`、`recurrentppo` 和 `tqc`。配置按当次 `--algorithm` 校验，可在对应算法段通过 `policy` 覆盖默认策略（如 `recurrentppo` 使用 `MlpLstmPolicy`）。试运行可通过 `--timesteps` 覆盖总步数，并将 `--run-type` 设为 `pilot` 或 `smoke`；覆盖仅改变总步数，不会自动缩小算法的 rollout/batch 等参数。特别是 PPO/RecurrentPPO，若要缩短实际冒烟时间，应在临时 JSON 配置中同时调小 `n_steps` 等参数。

- **模型**：保存为 SB3 原生 `.zip`，路径为 `trained_models/<DATASETNAME>/<algorithm>/`。文件名简要包含算法、种子、训练步数和主要超参数；完整有效配置、训练/验证数据哈希、环境及运行信息写入 zip 内的 `metadata.txt`。同名模型不会被覆盖。
- **训练日志**：写入 `tensorboard_log/<DATASETNAME>/<algorithm>/<run_id>/`，包含 SB3 CSV/TensorBoard 日志。
- **运行结果**：写入 `results/train_model/<DATASETNAME>/<run_id>/`，包括运行清单、有效配置及验证集账户价值和动作轨迹。当前验证复用 FinRL 的 `DRLAgent.DRL_prediction`；统一指标和正式样本外回测仍属于 T05，不由本训练模块另行定义。

### 后续 T05 回测模型读取约定

T05 回测入口应接收训练完成的 SB3 模型 `.zip` 路径和 `DATASETNAME`，不从文件名推断完整配置。模型 zip 内含 `metadata.txt`：文件以说明行开头，空行后为 JSON；`effective_config` 保存算法、数据集名、种子、训练步数、策略、算法参数、环境参数和数据路径，顶层 `runtime` 记录软件版本等运行信息。回测脚本应读取并校验其中的算法和数据集与本次请求一致。

模型加载通过 `train_model.runner.load_model_for_inference(algorithm, model_path, device=...)` 自动区分 on-policy（`ppo`、`recurrentppo` 不传 `buffer_size`）与 off-policy（`buffer_size=1`），再将配置中 `data.test_file` 指向的测试 CSV 构造成共享交易环境，调用 `DRLAgent.DRL_prediction(model, environment, deterministic=True)` 或统一回测入口获取账户价值和动作轨迹。测试集只用于最终样本外回测，不用于训练、调参或选模；验证集选模及统一指标、结果登记由 T05 按项目协议实现。

## 项目结构

> **注意**：本节随项目开发持续更新；新增模块、脚本或文档时须同步维护本表。

| 路径 | 说明 |
| --- | --- |
| `finrl/` | FinRL 库源码（上游代码，含 agents / meta / applications 三层） |
| `examples/` | 上游教程脚本与 Notebook（快速开始入口） |
| `src/` | 小组自研核心源码（按模块建子目录：`data_pipeline/`、`train_model/`、`backtest/`、`sac/`、`tqc/`、`thresholds/`、`baselines/`、`ensemble/`、`evaluation/`；共享环境 `trading_environment.py` 暂留根目录） |
| `scripts/` | 可直接运行的小组入口脚本（`setup_env.ps1`、`prepare_sse48.py`、`validate_prepared48.py`、`train_template.py`、`backtest.py`、`run_equal_weight_baseline.py`、`p4_critic_diagnostics.py`、冒烟测试等；不存放文档） |
| `config/` | 实验、数据与门限 JSON 配置 |
| `docs/` | 根目录仅保留现行唯一计划 `docs/小组项目计划.md`（旧版通过 Git 追溯）；下设 `modules/`（模块说明）、`contracts/`（接口契约）、`handoffs/`（交接说明）及 `upstream/`（上游归档） |
| `unit_tests/` | 单元测试与接口/门限/回测回归测试 |
| `local_test/` | 本地私有测试代码；已忽略，不纳入版本控制 |
| `requirements-lock.txt` | 权威锁定环境（Python 3.9.13 验证） |
| `NOTICE.md` / `LICENSE` | 上游致谢与 MIT 许可证 |
| `results/` `trained_models/` `tensorboard_log/` `datasets/` | 运行产物（不入库，见下节） |

### 目录与文档规范

- **源码与脚本分离**：模块业务代码在 `src/` 下按模块建子目录存放；`scripts/` 下只放置可作为入口运行的脚本，不放置 Markdown 说明文档。
- **文档分类与命名**：`docs/` 下仅保留最新一版 `小组项目计划.md`；三类小组文档分别归档至 `docs/modules/`（`module_<模块名>.md`）、`docs/contracts/`（`contract_<主题>_v<N>.md`）和 `docs/handoffs/`（`handoff_<主题>_v<N>.md`）。

## 复现与数据政策

> **注意**：本节为当前约定，随项目开发（尤其是 T12 独立复现验证的反馈）可能有修订与调整。

- **环境复现**：以 `requirements-lock.txt` + `scripts/setup_env.ps1` 为准，固定 Python 3.9。
- **运行产物不入库**：`results/`、`trained_models/`、`tensorboard_log/`、`temp/`、`datasets/`、`local_test/` 及散落的 `*.csv`/`*.png` 均已加入 `.gitignore`，日常训练产物和本地测试不进版本库。
- **最终定稿例外**：`final_deliverables/` 为最终定稿追踪文件夹，计划仅在最终定稿时一次性提交入库，中途不做细节追踪。
- **数据政策**：使用公开历史数据，本地缓存并可复现；缓存文件不入库，数据处理与版本说明随实验进展补充。
- **敏感信息**：API 密钥等写入 `finrl/config_private.py`（已忽略），不得提交。

## 项目文档

- 现行项目计划：[docs/小组项目计划.md](docs/小组项目计划.md)（历史版本通过 Git 追溯）
- 模块说明：[数据准备](docs/modules/module_data_sse48.md)、[训练模板](docs/modules/module_train_model.md)、[统一回测](docs/modules/module_backtest.md)、[等权基线](docs/modules/module_baselines_equal_weight.md)、[P4 实验方案](docs/modules/module_p4_sac_tqc.md)、[SAC 模块](docs/modules/module_p4_sac.md)、[TQC 模块](docs/modules/module_p4_tqc.md)、[P4 参数优化消融](docs/modules/module_p4_ablation_optimization.md)
- 接口契约与交接：[P1 接口契约](docs/contracts/contract_p1_train_interface_v1.md)、[P1 与 P2 回测交接契约](docs/contracts/contract_p1_p2_backtest_v1.md)、[S0 基础联调交接单](docs/handoffs/handoff_s0_baseline_integration_v1.md)、[P4 训练与门限冻结交接单](docs/handoffs/handoff_p4_sac_tqc_validation_v1.md)
- 上游 FinRL 文档：[docs/upstream/README_FinRL.md](docs/upstream/README_FinRL.md)、[官方文档站点](https://finrl.readthedocs.io/)

## 免责声明

本项目为课程教学项目，仅供学习与研究使用，不构成任何投资建议。历史回测表现不代表未来收益。


## P2 BaoStock数据与统一回测

当前采用48只固定股票池和复权模拟单位。安装requirements-lock.txt后，在项目根目录运行以下命令生成本地数据。

```powershell
python scripts/prepare_sse48.py --fetch
python scripts/backtest.py check-data --data datasets/sse48_baostock_v2/train_data.csv
```

数据准备说明见[48股BaoStock数据](docs/modules/module_data_sse48.md)，模型与回测接口见[统一回测](docs/modules/module_backtest.md)。原始行情、生成CSV、模型、日志及图表存放在Git忽略目录。
