# 基础联调交接单 v1

**交接编号**：S0（联调启动关口）  
**文档版本**：v1  
**交接范围**：P1 / P2 → P3 / P4（第一阶段实验，2 周，**12 月 6 日截止**）、P5（固定集成选优输入规范）  
**依据文档**：[小组项目计划（v2.1）](../小组项目计划.md)、[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md)、[P1 与 P2 回测交接契约 v1](../contracts/contract_p1_p2_backtest_v1.md)、[P2 回测更新交付 v1](handoff_p2_backtest_update_v1.md)

---

## 1. 交接概述与适用范围

本交接单标志着 **P1（统一训练接口与算法/门限接入口）** 与 **P2（SSE48 数据工程与统一回测框架）** 已完成编码、自测与本地端到端联调，正式移交 P3、P4 启动第一阶段实验（12 月 6 日截止），并同步明确 P5 固定集成选优的输入规范。

目前已完成全部基础联调核验任务：
- `.venv` 已与 `requirements-lock.txt` 对齐并通过 `pip check` 及 CUDA 可用性检查；
- 全量 36 项单元测试全部通过（含接口校验、门限协议、回测对账、数据缓存及 `validation_actions.csv` 显式保留 `date` 列修复）；
- 四算法（`ppo`、`recurrentppo`、`sac`、`tqc`）合成数据训练—回测接口联调通过，验证账户曲线逐日差值为 `0.0`，成交单位与重复回放产物字节完全一致；
- `sse48_baostock_v2` 正式数据完整性校验及 485 个验证交易日回放核验通过，未读取测试集绩效。

---

## 2. 已冻结基线与交付清单（含调用模块参考与用法）

| 类别 | 交付内容 / 路径 | 说明、使用方法与参考文档 |
| --- | --- | --- |
| **权威依赖** | `requirements-lock.txt`、`scripts/setup_env.ps1` | 固定 Windows / Python 3.9.13、`torch==2.8.0+cu126`、`stable_baselines3==2.7.1`、`sb3_contrib==2.7.1`。云端或新环境使用 `pip install -r requirements-lock.txt` 安装对齐。 |
| **数据基线** | `datasets/sse48_baostock_v2/`、`config/sse48_baostock_v2_universe.json` | 48 股固定池、8 项指标、481 维观测 / 48 维动作；训练（2017-01-16–2020-12-31）、验证（2021-01-04–2022-12-30）、测试（2023-01-03–2026-11-03）；前复权模拟单位与 `tradable` 规则。<br>**使用方法**：训练传 `--dataset sse48_baostock_v2`；回测传 `val_data.csv`/`trade_data.csv` 与对应 `*_index.csv`，可用 `python scripts/backtest.py check-data --data <CSV>` 校验。<br>**参考文档**：[SSE48 数据说明](../modules/module_data_sse48.md)。 |
| **共享环境** | `src/trading_environment.py` | 训练与回测共用同一环境源码与哈希校验（**禁止 P3/P4 修改**）。<br>**使用方法**：通过配置文件 `environment` 段传入 `tradability_policy: "enforce_column"` 与 `price_basis: "adjusted_virtual_units"`（模板见 `config/train_model_adjusted.example.json`）。<br>**参考文档**：[回测模块说明](../modules/module_backtest.md)（第 4 节）、[P1 与 P2 回测交接契约 v1](../contracts/contract_p1_p2_backtest_v1.md)（第 5 节）。 |
| **训练入口** | `scripts/train_template.py`、`src/train_model/` | 支持 `ppo`、`ddpg`、`td3`、`sac`、`recurrentppo`、`tqc`，按当次 `--algorithm` 校验配置。<br>**使用方法**：CLI 运行 `python scripts/train_template.py --config <JSON> --dataset sse48_baostock_v2 --algorithm <ALGO> --seed <SEED> --run-type <smoke\|pilot\|main>`（`smoke`/`pilot` 可加 `--timesteps N`）；推理加载调用 `train_model.runner.load_model_for_inference(algorithm, model_path, device=device)`。<br>**参考文档**：[训练模块说明](../modules/module_train_model.md)、[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md)（第 2–4 节）。 |
| **回测入口** | `scripts/backtest.py`、`src/backtest/` | 支持条件身份指纹、门限参数透传、配对分组、统一指标与对账。<br>**使用方法**：单次运行 `python scripts/backtest.py run --model <ZIP> --data <CSV> --benchmark <INDEX_CSV> --split <validation\|test> [--threshold-mode <none\|simple\|critic> --threshold-config <JSON>]`；批量与汇总分别使用 `batch` 与 `summarize` 子命令。<br>**参考文档**：[回测模块说明](../modules/module_backtest.md)、[P2 回测更新交付 v1](handoff_p2_backtest_update_v1.md)。 |
| **扩展目录** | `src/thresholds/`、`src/baselines/`、`src/ensemble/`、`src/evaluation/` | 已创建占位包：`src/thresholds/`（P3/P4 门限实现接入）、`src/baselines/`（P3/P4 等权基线接入）、`src/ensemble/`（P5 集成选优）、`src/evaluation/`（P6 评估）。 |

---

## 3. P3、P4 开始前需共同确定的实验约定

在启动正式训练前，P3 与 P4 须先在配置与协议层面共同对齐并冻结以下控制量与接入约定；各项参数按来源划分为**从已有框架中选取值**、**可参考经验规定的值**与**自行设计调整的参数**三类：

1. **共同控制量与预算冻结**：
   - **从已有框架中选取值**：数据集 `sse48_baostock_v2` 及既定训练/验证/测试划分；共享环境固定字段（`price_basis="adjusted_virtual_units"`、`tradability_policy="enforce_column"`、`tech_indicator_list` 8 项指标、`risk_indicator_col="turbulence"`，见 `config/train_model_adjusted.example.json`）。
   - **可参考经验规定的值**：**3 个预先冻结的随机种子**（如常用固定整数种子组合）；`initial_amount=1000000`、`buy_cost_pct=0.001`、`sell_cost_pct=0.001`、`hmax=100`、`reward_scaling=0.0001`、`turbulence_threshold=null` 等 FinRL 常用基线环境参数（可沿用 `config/train_model_adjusted.example.json` 默认值后冻结）。
   - **自行设计调整的参数**：**总交互步数预算（`total_timesteps`）**。在正式训练前须先以 `--run-type pilot` 实测步数、耗时、内存与 GPU 机时，确认预算后再冻结主配置；`ppo`/`recurrentppo` 实际步数按 rollout 上取整，`sac`/`tqc` 记录实际梯度更新数。

2. **算法超参、门限候选与选择规则**：
   - **从已有框架中选取值**：算法注册名（`ppo`、`recurrentppo`、`sac`、`tqc`）、策略基类框架（`ppo` 默认 `MlpPolicy`、`recurrentppo` 使用 `MlpLstmPolicy`、`sac`/`tqc` 使用对应 off-policy 策略）、回测门限模式枚举（`--threshold-mode` ∈ `{none, simple, critic}`）及输出指标口径（`metrics.json` 中的 `sharpe_ratio` 等）。
   - **可参考经验规定的值**：`config/train_model_adjusted.example.json` 与 `config/backtest_thresholds.example.json` 给出的通用超参初值（如 `learning_rate`、`batch_size`、`n_steps`、`buffer_size`、`ent_coef`、`learning_starts`）；简单动作幅度门限的经验候选档位（如 `q_min ∈ {2, 5, 10}`，对应 `hmax=100`）；验证期主选优指标（`sharpe_ratio`）。
   - **自行设计调整的参数**：
     - P3 自行设计并冻结 `ppo`、`recurrentppo` 的正式超参（含 LSTM `policy_kwargs`、`n_steps` 等）及简单门限最终候选集合与临界处理；
     - P4 自行设计并冻结 `sac`、`tqc` 的正式超参（含 `top_quantiles_to_drop_per_net`、`buffer_size`、`learning_starts` 等）及 Critic 门限的**摘要定义（SAC 标量 Critic 摘要 / TQC 分位数均值摘要）、动作输入坐标、阈值候选范围、相等边界与无效值处理**；
     - P3、P4 须在正式训练前**共同设计、确定并版本化统一的验证选择细则（主指标、`null`/非有限值处理、并列平局规则）**，并确认 SAC/TQC 共用的简单门限选择规则。

3. **公共接口与运行入口约定**：
   - **训练入口**（参考 [训练模块说明](../modules/module_train_model.md)、[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md)）：统一通过 `scripts/train_template.py --config <JSON> --dataset sse48_baostock_v2 --algorithm <ALGO> --seed <SEED> --run-type <smoke|pilot|main>` 运行，允许 P3/P4 各自交付独立的 JSON 配置文件，禁止修改共享环境源码或绕过公共入口自行训练。
   - **回测入口**（参考 [回测模块说明](../modules/module_backtest.md)、[P2 回测更新交付 v1](handoff_p2_backtest_update_v1.md)）：统一通过 `scripts/backtest.py` 执行验证与测试回测；`--threshold-mode` 支持 `none`、`simple`、`critic`（当前 `simple` 与 `critic` 在 P3/P4 正式实现接入前仍抛出 `NotImplementedError`）。
   - **门限返回契约**：`apply_threshold` 返回 `(action, info)`，其中 `action` 为形状 `(n,)`、`float32`、范围 `[-1, 1]` 且无非有限值的数组；`simple` 的 `accepted` 为 `(n,)` 布尔数组，`critic` 为布尔标量；不得在门限内部修改模型权重或绕过环境停牌/风险强平规则。

4. **交付依赖与隔离约束**：
   - **本地与云端依赖**：本地 `.venv`（含 CUDA）已按 `requirements-lock.txt` 验证；**使用云算力者须在启动正式训练前自行按锁文件完成云端环境适配与基础 smoke 核验**，不得静默升级或替换核心依赖。
   - **测试集严格隔离**：正式超参、门限阈值与 P5 选中模型全部冻结前，严禁读取或利用 `trade_data.csv`（测试集）的绩效指标进行任何调参、门限选择或模型筛选。

---

## 4. 下游接入边界、待办与 P5 输入说明

### 4.1 P3 待办（记忆组 & 简单门限，12 月 6 日前）
- **配置与训练验证**：基于 `config/train_model_adjusted.example.json` 复制并冻结 `ppo`、`recurrentppo` 正式配置 JSON，调用 `scripts/train_template.py` 在 SSE48 数据上执行 `smoke` 与 `pilot`，验证真实多步 LSTM 隐藏状态传递与回合/跨数据集重置一致性（参考 [训练模块说明](../modules/module_train_model.md)、[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md) 第 3 节）。
- **应修改/新增源码（简单门限）**：
  - 在 `src/thresholds/` 新增简单动作幅度门限实现，并修改 `src/backtest/runner.py` 中的 `apply_threshold` 接入 `simple` 分支（若拆分独立模块须同步扩展 `run_manifest.json` 记录的实现源码 SHA256 哈希）；
  - 在 `unit_tests/` 补充简单门限临界值、负动作向零取整及 `q_min=1` 等价性单测（可参照 `unit_tests/test_backtest_thresholds.py`）。
  - **参考文档**：[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md)（第 5 节）、[P2 回测更新交付 v1](handoff_p2_backtest_update_v1.md)（第 2–3 节）。
- **应修改/新增源码（等权基线）**：与 P4 共同在 `src/baselines/` 实现同池等权买入持有基线（参考 [SSE48 数据说明](../modules/module_data_sse48.md)、[回测模块说明](../modules/module_backtest.md)，由 P2 核验口径）。
- **正式训练与回测**：调用 `scripts/train_template.py` 与 `scripts/backtest.py` 完成 `ppo`、`recurrentppo` 各 3 种子共 6 次正式训练、验证与测试回测，输出记忆对照初步分析。

### 4.2 P4 待办（门限组 & Critic 门限，12 月 6 日前）
- **配置与训练验证**：基于 `config/train_model_adjusted.example.json` 复制并冻结 `sac`、`tqc` 正式配置 JSON，并参考 `config/backtest_thresholds.example.json` 准备门限配置 JSON；调用 `scripts/train_template.py` 完成 `sac`、`tqc` 的 `smoke` 与 `pilot`（参考 [训练模块说明](../modules/module_train_model.md)、[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md)）。
- **应修改/新增源码（Critic 门限）**：
  - 在 `src/thresholds/` 新增 SAC 与 TQC 的推理期 Critic 门限实现，并修改 `src/backtest/runner.py` 中的 `apply_threshold` 接入 `critic` 分支（按契约记录 `critic_policy_value`、`critic_zero_value`、`critic_delta` 与 `accepted`，若拆分独立模块须同步扩展实现源码哈希）；
  - 在 `unit_tests/` 补充 Critic 门限整向量拒绝、相等边界及无效值处理单测（可参照 `unit_tests/test_backtest_thresholds.py`）。
  - **参考文档**：[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md)（第 5 节）、[P2 回测更新交付 v1](handoff_p2_backtest_update_v1.md)（第 2–3 节）。
- **应修改/新增源码（等权基线）**：与 P3 共同在 `src/baselines/` 交付同池等权买入持有基线（参考 [SSE48 数据说明](../modules/module_data_sse48.md)、[回测模块说明](../modules/module_backtest.md)）。
- **正式训练与配对回测**：调用 `scripts/train_template.py` 完成 `sac`、`tqc` 各 3 种子共 6 次正式训练，调用 `scripts/backtest.py` 在验证集选定并冻结简单门限与 Critic 门限参数；在选中模型与门限均冻结后，使用 `scripts/backtest.py run`/`batch` 与 `summarize` 完成 `S0/S1/S2`、`T0/T1/T2` 共 18 次配对测试回测与门限行为初步分析（参考 [回测模块说明](../modules/module_backtest.md)）。

### 4.3 P5 输入说明（固定验证集集成选优，12 月 20 日前）

1. **现 P1、P2 联调已完成项（可立即消费）**：
   - **统一模型元数据与运行清单**：训练产物统一保存于 `trained_models/sse48_baostock_v2/<algorithm>/`，zip 内含 `metadata.txt`（记录算法、种子、`requested_timesteps`、`actual_timesteps`、`actual_policy`、数据与环境源码 SHA256 哈希等）；`results/train_model/sse48_baostock_v2/<run_id>/run_manifest.json` 关联模型、配置、`validation_account_value.csv` 与 `validation_actions.csv`（均显式含 `date` 列）。
   - **统一回测与 Sharpe 消费口径**：P2 已在 `src/backtest/` 提供统一的验证期回放、指标计算（`metrics.json` 中的 `sharpe_ratio` 等）、多条件判重与汇总接口；P5 可基于此在 `src/ensemble/` 提前开发选优与交叉核验脚本。

2. **P3、P4 应交付 P5 项（12 月 6 日前训练完成后移交）**：
   - **12 个无门限正式模型**：`ppo`、`recurrentppo`、`sac`、`tqc` 各 3 个预冻结种子的正式模型 `.zip` 文件及其 `run_manifest.json`、`effective_config.json`（不含门限变体）。
   - **验证集结果与规则记录**：12 个候选模型的验证期回放指标、已登记的门限候选与验证选择记录、数据/环境哈希核验状态，供 P5 执行一次固定验证集选优、冻结选中模型并复核 24 次正式测试回测与独立抽查。