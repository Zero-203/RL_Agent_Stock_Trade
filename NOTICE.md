# NOTICE

本项目基于 **FinRL** 二次开发：

- **上游项目**：[AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)，版本 v0.3.8
- **上游版权**：Copyright (c) 2024 AI4Finance Foundation Inc.
- **上游许可证**：MIT License（全文见根目录 [LICENSE](LICENSE)，其中包含上游版权与许可声明）

## 许可证合规说明

依据 MIT License 条款，本项目在所有副本及实质部分中完整保留上游的版权声明与许可声明（即根目录 `LICENSE` 文件）。本小组在 FinRL 0.3.8 基础上的修改与新增内容，同样以 MIT License 发布。

## 相对上游的主要修改

- 根目录 README 替换为本小组项目说明；上游原始 README 归档至 [docs/upstream/README_FinRL.md](docs/upstream/README_FinRL.md)。
- 新增 `requirements-lock.txt`（经验证环境的 pip 全量锁定）与 `scripts/setup_env.ps1`（一键环境搭建脚本）。
- 扩充 `.gitignore`（运行产物、IDE 目录、最终交付文件夹的入库策略）。
- 清理上游 `.github/` 中的募捐信息（FUNDING.yml、sponsor.md）。
- 后续开发中的代码修改以 Git 提交历史为准；基准代码以 "Import FinRL 0.3.8 baseline" 提交为分界。

## 致谢

感谢 AI4Finance Foundation 开源 FinRL 框架。上游文档详见 `docs/upstream/` 与 `docs/` 目录。

---

*This project is a derivative work of FinRL (MIT License, © AI4Finance Foundation Inc.). The original copyright and permission notices are preserved in LICENSE. This file itself does not modify or replace the terms of the MIT License.*
