# 恢复

可编辑原生检查点 `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-KBASE-v1_20261105T151743Z/strong/rollout/final_checkpoint`；manifest SHA `6afc490206e34af021ae4f5a1a842b8893925c4c8441830a513e38c46762edcf`。完整资产留服务器，Git checkpoint_metadata 只是索引与小状态，恢复须复制manifest所有资源并校验。

GLB `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-KBASE-v1_20261105T151743Z/strong/export/room_geometry.glb`，20868字节，SHA `f5875c139f863a5b788b5a4f78db92441d67baaeff2e25aa2c4ba20268d8e678`。原生新进程冷加载已通过，对象/资产/24回执保留，恢复后新编辑未测。命令见 execution/restore/receipt.json 和 RESTORE_PLAN.json；不要重跑已完成推理。
