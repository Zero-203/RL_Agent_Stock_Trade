# T04 训练模板

算法注册、参数透传与循环预测约定见 [P1 接口契约 v1](../docs/P1_接口契约_v1.md)；回测侧消费约定见 [P1 与 P2 回测交接契约](../docs/P1_P2_回测交接契约_v1.md)。

数据集文件放在 `datasets/<DATASETNAME>/` 下，默认包含配置指定的训练、验证和测试 CSV。训练入口只读取训练集和验证集，不读取测试 CSV。`DATASETNAME` 作为参数传入：

```powershell
python scripts\train_template.py --dataset DOW30 --algorithm ppo --seed 42
```

训练配置位于 `config/train_model.json`。训练步数覆盖仅用于试运行，并须标记类型。覆盖总步数不会自动缩小算法的 rollout/batch 参数；PPO 冒烟时应先复制配置到临时 JSON，并调小 `n_steps` 和相应 batch 参数：

```powershell
python scripts\train_template.py --config path\to\smoke.json --dataset DOW30 --algorithm ppo --seed 42 --timesteps 8 --run-type smoke
```

模型以 SB3 原生 `.zip` 保存到 `trained_models/<DATASETNAME>/<algorithm>/`，短文件名包括算法、种子、训练步数和主要超参数；完整实验配置与数据版本写入 zip 内的 `metadata.txt`。训练日志写入 `tensorboard_log/<DATASETNAME>/<algorithm>/<run_id>/`；运行清单及验证集账户价值/动作记录写入 `results/train_model/<DATASETNAME>/<run_id>/`。同名模型不会被覆盖。

## 后续 T05 回测读取约定

支持的算法为 `ppo`、`ddpg`、`td3`、`sac`、`recurrentppo` 和 `tqc`；配置按当次 `--algorithm` 校验，可在对应算法段通过 `policy` 覆盖默认策略（如 `recurrentppo` 使用 `MlpLstmPolicy`）。推理加载请使用 `train_model.runner.load_model_for_inference(algorithm, model_path, device=device)`，自动区分 on-policy（不传 `buffer_size`）与 off-policy（`buffer_size=1`）。

```python
from train_model.runner import load_model_for_inference

model = load_model_for_inference(algorithm, model_path, device=device)
account_values, actions = DRLAgent.DRL_prediction(
	model=model,
	environment=test_environment,
	deterministic=True,
)
```

`test_environment` 应使用 `datasets/<DATASETNAME>/<data.test_file>` 和元数据中的环境规格构造。测试集只用于最终样本外回测；验证集用于协议允许的选模。回测指标、结果登记和产物格式由 T05 统一实现，不在训练模块或各算法脚本中另设口径。
