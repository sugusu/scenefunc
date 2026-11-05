# 固定官方训练入口的数据接线

[ENTRY-v1](../../../runs/920/P920-ENTRY-v1_20261104T204327Z/README.md)已实际运行修改后的官方 `src/train_sft.py`。补丁只新增显式决定 manifest/arm 选项和 dry-run 批次检查；默认数据路径仍使用上游 StepwiseSFTDataset。原已验收运行环境不改，在独立 overlay 中复制四个固定上游文件和两个本地数据模块，再应用补丁。

新增决定模式要求 `--dry-run --dataloader-workers 0`。每个数据项保留同一决定的理由样本与动作样本，DataLoader 以决定为批次单位，整理器展开成两个例子。批次一决定时，两组83目标 token、10图像实例，目标 ID 相同；配对动作含已屏蔽损失的理由前缀。完整真实 CLI 命令、文件哈希和拒绝输出见 run 的 PLAN、RESULT、recorded。普通数据 dry-run 通过，两个不完整／越过 dry-run 的调用在 processor 加载前拒绝。

本补丁实现并验证的是官方入口的数据选择与批次编码。它不启用决定数据训练，没有实例化 Trainer、加载模型或执行前向／优化。训练前仍需：来源及家族合格清单、正式资格门槛接入、loss 归一化与梯度累积核对、实际预算匹配、LoRA 配置审阅和单独训练授权。来源声明不能代替独立复核，83相等也不证明FLOP相等。

上游基线：[Scene-Designer a0565e5](https://github.com/ZJU-OmniAI/Scene-Designer/tree/a0565e50ed7d32c0e09132b9311c0b9b91165405)。四个上游文件在PLAN中锁定SHA；原文件与实际overlay留920，Git保存本地接入模块、补丁和恢复证据，不以仓库可读性推定额外许可。

## 新的有限机制试运行入口（尚未授权训练）

[PILOTPLAN](../../../runs/920/P920-PILOTPLAN-v1_20261105T094043Z/README.md)在新的独立上游overlay新增显式pilot dispatch，旧audit-only入口及专业资格清单保持原样。继承实际Transformers Trainer，通过真实官方加载器已生成的锁定张量，按来源等权计算选定目标位置的因果CE，微例累积20后一次更新。两组真实CLI CPU检查已验证Trainer.training_step的缩放，反向函数被拦截；未实例化底座、未执行优化。训练分支实现语言q/v LoRA、有限值停止、逐步保存与适配器同进程重载检查，但尚未执行该分支。

正式可审方案PLAN_V2每组2更新，r8/alpha16、3833856参数、总80微例前向/反向；必须收到用户对该方案的单独批准。批准仅允许此有标注条件重建的机制试运行，不能伪称专业偏好监督已取得、通用训练清单已合格或学习收益成立。缺少/false批准的实际CLI均在模型加载前拒绝。
