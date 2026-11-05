# 恢复

可编辑原生检查点 `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-KBASE-v1_20261105T151743Z/knowledge/rollout/final_checkpoint`；manifest SHA `1b842354d25f81a0ff5b279727d92fe80dc5d32d57c7003c71ce56a362be84e9`。完整资产留服务器，Git checkpoint_metadata 只是索引与小状态，恢复须复制manifest所有资源并校验。

GLB `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-KBASE-v1_20261105T151743Z/knowledge/export/room_geometry.glb`，4972308字节，SHA `08fd07b42f8e8cf5516c49a6f1a87f06ad7ceaaef1bee0dc1ec556cae0703c0c`。原生新进程冷加载已通过，对象/资产/24回执保留，恢复后新编辑未测。命令见 execution/restore/receipt.json 和 RESTORE_PLAN.json；不要重跑已完成推理。
