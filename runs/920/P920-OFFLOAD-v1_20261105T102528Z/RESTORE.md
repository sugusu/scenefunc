# 保存位置与恢复边界

底座使用PLAN.json内固定Scene-Builder revision `63e235a0c0d6e5bc2f8f3485214df4457fd2e672`，四权重文件来源与SHA均在计划内。LoRA文件留在服务器：

- `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-OFFLOAD-v1_20261105T102528Z/authorized_training/separate/adapter`
- `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-OFFLOAD-v1_20261105T102528Z/authorized_training/paired/adapter`

每组的`checkpoints/checkpoint-1`和`checkpoint-2`保留adapter、optimizer、scheduler、Trainer状态和RNG；逐文件字节与SHA在CHECKPOINT_MANIFEST.json。当前Git不含大权重、adapter权重或完整checkpoint。

读取前先核SHA；在同一固定软件环境中，用本地固定base加载对应PEFT adapter，设`is_trainable=False`并切换eval，才可进行另行冻结的推理评测。若迁移服务器，应连同原权重、哈希锁定的输入张量/五图和计划代码复制；不能拿不同底座接此适配器。

本轮执行并验证的是同进程适配器保存/清零/重载，以及CPU从磁盘读取的checkpoint一致性；**未执行完整模型的冷启动加载或优化器续训**。训练入口采用不可覆盖输出目录，不能把已消费的批准文件用于重跑。需要新的训练计划与明确批准才可继续优化。
