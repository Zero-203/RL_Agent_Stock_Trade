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

> **占位**：本项目自研的统一训练脚本与回测脚本正在开发中（对应项目任务 T04 / T05）。开发完成后，本节将说明脚本入口、配置方式、命令行参数与结果产物位置。

## 项目结构

> **注意**：本节随项目开发持续修订，小组新增模块（训练/回测脚本、配置、实验登记等）落位后将及时更新。

| 路径 | 说明 |
| --- | --- |
| `finrl/` | FinRL 库源码（上游代码，含 agents / meta / applications 三层） |
| `examples/` | 上游教程脚本与 Notebook（快速开始入口） |
| `docs/` | 上游 Sphinx 文档源；`docs/upstream/` 归档上游 README；`docs/小组项目计划v1.2.md` 为本组完整项目计划 |
| `scripts/` | 小组工具脚本（目前含 `setup_env.ps1` 环境搭建） |
| `unit_tests/` | 上游单元测试 |
| `requirements-lock.txt` | 权威锁定环境（Python 3.9.13 验证） |
| `NOTICE.md` / `LICENSE` | 上游致谢与 MIT 许可证 |
| `results/` `trained_models/` `tensorboard_log/` `datasets/` | 运行产物（不入库，见下节） |

## 复现与数据政策

> **注意**：本节为当前约定，随项目开发（尤其是 T12 独立复现验证的反馈）可能有修订与调整。

- **环境复现**：以 `requirements-lock.txt` + `scripts/setup_env.ps1` 为准，固定 Python 3.9。
- **运行产物不入库**：`results/`、`trained_models/`、`tensorboard_log/`、`temp/`、`datasets/` 及散落的 `*.csv`/`*.png` 均已加入 `.gitignore`，日常训练产物不进版本库。
- **最终定稿例外**：`final_deliverables/` 为最终定稿追踪文件夹，计划仅在最终定稿时一次性提交入库，中途不做细节追踪。
- **数据政策**：使用公开历史数据，本地缓存并可复现；缓存文件不入库，数据处理与版本说明随实验进展补充。
- **敏感信息**：API 密钥等写入 `finrl/config_private.py`（已忽略），不得提交。

## 项目文档

- 完整项目计划：[docs/小组项目计划v1.2.md](docs/小组项目计划v1.2.md)
- 上游 FinRL 文档：[docs/upstream/README_FinRL.md](docs/upstream/README_FinRL.md)、[官方文档站点](https://finrl.readthedocs.io/)

## 免责声明

本项目为课程教学项目，仅供学习与研究使用，不构成任何投资建议。历史回测表现不代表未来收益。
