# 恢复与重现

仓库根目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc`。
运行根目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-MILAN-BASE-v1_20261105T225639Z`。

最后真实检查点为运行目录下`rollout/round_005/checkpoint`。不要重启已失败推理或覆盖旧输出。`TERMINAL_SELECTION.json`给出manifest哈希；`checkpoint_metadata/`为可分享元数据，不含完整资产。初态来自上一批`P920-MILAN-TASK-v1_20261105T224015Z/build/bundle`；资产来源和完整恢复路径沿用该批RECOVERY与manifest，权重固定来源见PLAN。

可直接查看运行目录下`export/room_geometry.glb`。对检查点做新的只读核验时，使用新输出目录，并沿用实际`execution/restore_stage/receipt.json`里的命令和`RESTORE_PLAN.json`；这不会重新生成场景。几何核验命令分别见`execution/final_audit_stage/receipt.json`与`execution/final_evaluation_stage/receipt.json`。

实际模型开始、退出、子调用与失败输出均在`execution/rollout_stage/`及`rollout/round_006/model/`。第7次输出未经执行。只有6条工具回执、4次真实场景改变。独立后处理成功不能把模型退出1改称成功。

公开归档不包含权重、完整checkpoint、网格、密钥或token。`ARTIFACTS.json`核对公开文件；`CODE_RECORD.json`记录实际用过的算法与归档代码。
