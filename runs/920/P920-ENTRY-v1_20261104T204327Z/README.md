# 官方 SFT 入口的配对数据 dry-run

实际从固定 Scene-Designer 的 `src/train_sft.py` 应用可复核补丁后执行，原运行环境源码不变。两个条件均经该入口选取 DecisionUnitDataset，并由 DataLoader 调用 DecisionUnitCollator。26.007 秒退出 0，14 项检查通过。

| 项目 | 配对 | 分开 |
| --- | --- | --- |
| 决定数／批次例数 | 1／2 | 1／2 |
| 理由／动作目标 token | 32／51 | 32／51 |
| 总目标 token | 83 | 83 |
| 非填充输入 token | 2630 | 2601 |
| 图像实例 | 10 | 10 |
| 文本批形状 | 2×8192 | 2×8192 |

两组目标 token ID 完全相同；动作样本只在配对组含理由前缀，该前缀不受监督。图像张量相同，评分专用哨兵未进入输入。普通 Stepwise 数据 dry-run 亦通过。缺少 `--dry-run` 的决定调用，以及缺少 manifest 的 arm 调用，均在输出目录创建和 processor/model 加载前按预期拒绝（子进程 exit 1）。这些拒绝是事前测试项，整个审计 exit 0。

[计划](PLAN.json) · [完整结果](RESULT.json) · [实际执行](EXECUTION.json) · [补丁](official_entry_decision_dry_run.patch) · [小产物与哈希](ARTIFACTS.json)

仅使用旧工程 fixture；没有新专业样本、场景回放、权重加载、模型前向、Trainer、优化器或训练。实际损失归一化、LoRA、优化步与计算公平性尚未验收，不能从 dry-run 推出训练已接入可用。命令、本地算法模块及小日志留在 recorded；四个上游源文件只保留固定哈希和服务器副本，用所存补丁恢复；不得用它绕过训练单独审定。
