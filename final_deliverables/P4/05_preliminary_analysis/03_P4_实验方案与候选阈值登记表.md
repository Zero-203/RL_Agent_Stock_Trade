# P4 SAC / TQC 训练配置与 Critic 门限方案 v1

> 状态：**草案（待与 P3 共同冻结控制量；Critic 候选阈值待 pilot 后登记）**  
> 分支：`feat/p4-sac-tqc`  
> 算法与代码实现文档：[SAC 模块说明](module_p4_sac.md)（`src/sac/`）、[TQC 模块说明](module_p4_tqc.md)（`src/tqc/`）、[P4 补充诊断与参数优化消融](module_p4_ablation_optimization.md)、诊断脚本 [p4_critic_diagnostics.py](../../scripts/p4_critic_diagnostics.py)  
> 依据：[小组项目计划](../小组项目计划.md) 第 2.1、2.3、2.4 节，[P1 接口契约 v1](../contracts/contract_p1_train_interface_v1.md)，[P2 回测更新交付 v1](../handoffs/handoff_p2_backtest_update_v1.md)，[基础联调交接单 v1](../handoffs/handoff_s0_baseline_integration_v1.md)。

## 1. 范围

| 条件 | 模型 | 推理规则 | 负责 |
| --- | --- | --- | --- |
| S0 / T0 | SAC / TQC 正式模型 | 无门限 | P4 |
| S1 / T1 | 与 S0 / T0 **同一模型文件** | 简单门限 `simple` | 实现由 P3 提供，P4 运行 |
| S2 / T2 | 与 S0 / T0 **同一模型文件** | Critic 门限 `critic_delta_v1` | P4（本文件） |

正式规模：SAC、TQC 各 3 种子 = **6 次正式训练**；测试期 6 条件 × 3 种子 = 18 次配对回测（在 P5 冻结选中模型、门限全部冻结之后才执行）。

## 2. 训练配置

配置文件：[train_model_p4_sac_tqc.json](../../config/train_model_p4_sac_tqc.json)。`environment` 段与 [train_model_adjusted.example.json](../../config/train_model_adjusted.example.json) 逐项一致（共享环境冻结字段，**不得修改**）。

| 参数 | SAC | TQC | 取值理由 |
| --- | --- | --- | --- |
| `learning_rate` | 3e-4 | 3e-4 | SB3 / TQC 论文默认；pilot 可对比 1e-4 |
| `buffer_size` | 100000 | 100000 | ≥ 正式步数，保留全部转移；481 维观测约 0.4 GB 内存 |
| `learning_starts` | 1000 | 1000 | 约 1 个训练回合（964 步）随机探索后再更新 |
| `batch_size` | 256 | 256 | SB3 默认，481 维输入下梯度更稳定 |
| `tau` / `gamma` | 0.005 / 0.99 | 0.005 / 0.99 | 默认目标网络软更新与折扣 |
| `train_freq` / `gradient_steps` | 1 / 1 | 1 / 1 | 每步 1 次梯度更新，`actual_gradient_updates ≈ 步数 − learning_starts` |
| `ent_coef` | `auto_0.1` | `auto_0.1` | 自动温度，初值沿用 FinRL 基线 |
| `net_arch` | [256, 256] | [256, 256] | 默认结构；SAC 与 TQC 一致，使差异只来自 Critic 形式 |
| `n_critics` | 2 | 2 | SAC 双 Q；TQC 两个分位数网络 |
| `n_quantiles` | — | 25 | TQC 默认 |
| `top_quantiles_to_drop_per_net` | — | 2 | TQC 默认截断（共丢弃 4/50 个分位数） |
| `total_timesteps` | 100000 | 100000 | **占位建议值**（约 104 个训练回合），须 pilot 后与 P3 共同冻结 |
| `device` | cuda | cuda | 本机 RTX 3060 Laptop，CUDA 12.6 已验证可用 |

设计原则：SAC 与 TQC 除 Critic 类型（双 Q vs 截断分位数）外**全部超参相同**，保证门限对照与算法对照的可解释性。

## 3. 需与 P3 共同冻结的控制量

| 项目 | P4 建议 | 状态 |
| --- | --- | --- |
| 3 个随机种子 | `42, 2024, 2026` | 待 P3 确认 |
| `total_timesteps` | 100000（pilot 实测后校准） | 待 pilot |
| 数据 / 环境 / 成本 | `sse48_baostock_v2` + 上表共享环境字段 | 已冻结（P2） |
| 验证主指标 | `metrics.json → strategy.sharpe`（P2 口径） | 待 P3 确认 |
| `null` / 非有限 Sharpe | 视为无效，不当作 0 | 待 P3 确认 |
| 并列规则 | 差值 < 1e-9 视为并列，取干预更小（阈值更低）的候选 | 待 P3 确认 |

## 4. Critic 门限登记：`critic_delta_v1`

实现：[src/thresholds/critic.py](../../src/thresholds/critic.py)，由 [src/backtest/runner.py](../../src/backtest/runner.py) 的 `apply_threshold` 在 `threshold_mode == "critic"` 时调用。

| 登记项 | 冻结内容 |
| --- | --- |
| 比较对象 | 同一观测 `s` 下，模型确定性动作 `a` 与零动作 `0`（全部持有不动） |
| 判定量 | `delta = summary(Q(s, a)) − summary(Q(s, 0))` |
| SAC 摘要 | `min_q`：双 Q 取最小（与 SAC TD 目标一致）；备选 `mean_q` |
| TQC 摘要 | `truncated_quantile_mean`：两网络 50 个分位数合并排序，丢弃最高 `2×2=4` 个后取均值（与 TQC TD 目标一致）；备选 `quantile_mean` |
| 动作输入坐标 | 环境坐标 `[-1, 1]` 经 `policy.scale_action` 映射为 Critic 输入；零动作同样映射（对称区间下仍为 0） |
| 放行规则 | `delta > delta_threshold` 时放行整向量；**相等视为拒绝** |
| 拒绝行为 | 输出全零向量，`accepted=false`；资金、停牌、风险强平仍由环境最后执行 |
| 无效值 | Critic 输出非有限值时抛 `FloatingPointError`，回测记为 failed，不猜测决策 |
| 配置格式 | `{"rule": "critic_delta_v1", "summary": "...", "delta_threshold": <有限数>}`，三键必填，未知键拒绝 |
| 诊断字段 | `actions.csv` 记录 `critic_policy_value`、`critic_zero_value`、`critic_delta`；`metrics.json` 记录 `threshold_vector_rejection_rate` |
| 实现指纹 | `threshold_implementation_sha256` 覆盖 `backtest/runner.py` 与 `src/thresholds/*.py`；清单另存逐文件 `threshold_code_sha256` |

> [!NOTE]
> `delta_threshold` 是 Critic 价值单位（受 `reward_scaling=1e-4` 与 `gamma` 影响），SAC 与 TQC 的阈值数值不要求相同。

## 5. 候选阈值登记流程（不使用正式模型、不读取测试集）

1. 用 **pilot 模型**（非正式模型）在验证集运行一次“全放行诊断”：`delta_threshold = -1e6`，结果与无门限逐日一致，但记录每日 `critic_delta`。
2. 每个算法分别取候选集合 `{0.0, P25, P50, P75}`（`critic_delta` 分布分位数，保留 2 位有效数字），最多 4 个候选。
3. 将候选集合写入本文件第 8 节并提交 Git，作为正式训练前的登记版本。
4. 正式模型训练完成后，在验证集对每个候选运行 S2 / T2；每个算法按“3 种子验证 Sharpe 均值最高”选出唯一阈值；任一种子 Sharpe 无效的候选整体剔除；全部无效则登记失败，不使用测试集补选。

批量回测模板：[backtest_p4_critic_validation.example.json](../../config/backtest_p4_critic_validation.example.json)（`REPLACE_WITH_REGISTERED_CANDIDATE` 未替换时会因配置校验失败而拒绝运行）。

## 6. 执行步骤

```powershell
conda activate finrl
cd "C:\Users\Haoyu Fan\PycharmProjects\FinRL\RL_Agent_Stock_Trade"

# 0) 数据
python scripts/prepare_sse48.py --fetch
python scripts/backtest.py check-data --data datasets/sse48_baostock_v2/train_data.csv

# 1) smoke：确认真实梯度更新（learning_starts=1000 < 2000）
python scripts/train_template.py --config config/train_model_p4_sac_tqc.json --dataset sse48_baostock_v2 --algorithm sac --seed 42 --timesteps 2000 --run-type smoke
python scripts/train_template.py --config config/train_model_p4_sac_tqc.json --dataset sse48_baostock_v2 --algorithm tqc --seed 42 --timesteps 2000 --run-type smoke

# 2) pilot：实测耗时 / 显存，并用于第 5 节候选登记
python scripts/train_template.py --config config/train_model_p4_sac_tqc.json --dataset sse48_baostock_v2 --algorithm sac --seed 42 --timesteps 20000 --run-type pilot
python scripts/train_template.py --config config/train_model_p4_sac_tqc.json --dataset sse48_baostock_v2 --algorithm tqc --seed 42 --timesteps 20000 --run-type pilot

# 3) 正式训练（控制量冻结后；6 次）
foreach ($algo in "sac","tqc") { foreach ($seed in 42,2024,2026) {
  python scripts/train_template.py --config config/train_model_p4_sac_tqc.json --dataset sse48_baostock_v2 --algorithm $algo --seed $seed --run-type main
} }

# 4) 验证集门限选择与冻结留痕（填入登记候选与正式模型路径后）
python scripts/backtest.py batch --config config/backtest_p4_critic_validation.json
python scripts/p4_critic_diagnostics.py select --algorithm sac --seeds 42 2024 2026 --manifests results/backtest/p4_critic_validation/*/run_manifest.json --output results/backtest/p4_critic_validation/sac_selection_record.json
python scripts/p4_critic_diagnostics.py select --algorithm tqc --seeds 42 2024 2026 --manifests results/backtest/p4_critic_validation/*/run_manifest.json --output results/backtest/p4_critic_validation/tqc_selection_record.json
```

## 7. 验收清单

- [x] Critic 门限实现与 P2 返回契约对齐（形状、`float32`、整向量拒绝、诊断有限值）
- [x] 单元测试：摘要与独立计算一致、严格边界、配置校验、非有限值报错、全放行等价于无门限、全拒绝不下单、实现指纹覆盖门限模块、P4 配置可构建 SAC / TQC（全量 60 项单测通过）
- [x] 数据集与基线：`sse48_baostock_v2` 本地构建完成，6 份 CSV SHA256 与 P2 发布快照 100% 一致；验证集同池等权基线已生成（总收益 `+2.54%`，Sharpe `0.1618`）
- [x] smoke（2,000 步，RTX 3060 Laptop CUDA）：SAC 耗时 `44.5s`、TQC 耗时 `44.1s`，`actual_gradient_updates = 1000 > 0`，模型重载与回放验证通过
- [x] pilot（20,000 步，RTX 3060 Laptop CUDA）：SAC 耗时 `516.7s`（约 `8.6` 分钟）、TQC 耗时 `516.2s`（约 `8.6` 分钟），`actual_gradient_updates = 19000`；推算 100,000 步单次正式训练约 `43` 分钟，6 次合计约 `4.3` GPU 小时（待与 P3 对齐确认最终 `total_timesteps` 与 3 个种子）
- [x] 第 8 节候选阈值登记并提交（严守测试集隔离，未读取 `trade_data.csv` 策略表现）
- [x] 6 次正式训练完成（SAC/TQC × 种子 `42, 2024, 2026`，各 `100,000` 步），验证集选定并冻结 SAC / TQC Critic 阈值均为 `delta_threshold = 0.0`（详见 [P4 验证与冻结交接单](../handoffs/handoff_p4_sac_tqc_validation_v1.md)；简单门限待 P3 合入）
- [ ] P5 冻结选中模型且 P3 合入简单门限后，执行 S0/S1/S2、T0/T1/T2 共 18 次测试回测

## 8. 候选阈值登记表（已按 20k Pilot 验证集 484 个决策日标定）

| 算法 | 摘要 | Pilot 模型（`seed=42, t=20k`） | 验证集 `critic_delta` 分位数（`P25 / P50 / P75`） | 登记候选 `delta_threshold`（`{0.0, P25, P50, P75}` 两位有效数字） | 登记日期 |
| --- | --- | --- | --- | --- | --- |
| SAC | `min_q` | `sac_s42_t20k_lr0p0003_b256_buf100k_start1000_entauto0p1.zip` | `0.1065 / 0.2630 / 0.3397` | `[0.0, 0.11, 0.26, 0.34]` | 2026-11-26 |
| TQC | `truncated_quantile_mean` | `tqc_s42_t20k_lr0p0003_b256_buf100k_start1000_entauto0p1.zip` | `1.4053 / 1.8686 / 2.1022` | `[0.0, 1.4, 1.9, 2.1]` | 2026-11-26 |
