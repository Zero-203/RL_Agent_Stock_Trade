# T04 训练模板

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

回测入口接收模型 zip 路径和 `DATASETNAME`。从 zip 的 `metadata.txt` 读取空行后的 JSON `effective_config`，校验算法和数据集名；不要仅依赖短文件名还原参数。加载方式采用原 FinRL/SB3 接口：

```python
model = MODELS[algorithm].load(model_path, device=device)
account_values, actions = DRLAgent.DRL_prediction(
	model=model,
	environment=test_environment,
	deterministic=True,
)
```

`test_environment` 应使用 `datasets/<DATASETNAME>/<data.test_file>` 和元数据中的环境规格构造。测试集只用于最终样本外回测；验证集用于协议允许的选模。回测指标、结果登记和产物格式由 T05 统一实现，不在训练模块或各算法脚本中另设口径。
