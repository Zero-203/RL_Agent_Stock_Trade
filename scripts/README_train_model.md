# T04 训练模板

数据集文件放在 `datasets/<DATASETNAME>/` 下，包含配置指定的训练、验证和测试 CSV。训练入口不会读取测试 CSV。`DATASETNAME` 作为参数传入：

```powershell
python scripts\train_template.py --dataset DOW30 --algorithm ppo --seed 42
```

训练配置位于 `config/train_model.json`。训练步数覆盖仅用于试运行，并须标记类型：

```powershell
python scripts\train_template.py --dataset DOW30 --algorithm ppo --seed 42 --timesteps 8 --run-type smoke
```

模型以 SB3 原生 `.zip` 保存到 `trained_models/<DATASETNAME>/<algorithm>/`，短文件名包括算法、种子、训练步数和主要超参数；完整实验配置与数据版本写入 zip 内的 `metadata.txt`。训练日志写入 `tensorboard_log/<DATASETNAME>/<algorithm>/<run_id>/`；运行清单及验证集账户价值/动作记录写入 `results/train_model/<DATASETNAME>/<run_id>/`。同名模型不会被覆盖。
