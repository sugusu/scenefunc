# 恢复

原生检查点：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-REF-v1_20261105T141516Z/native/checkpoint`，manifest SHA `ffa2393fd79be8aaec20da1be31fd9d67055b60876803cfe253f105f2cf46db3`。完整复制manifest列出资源并逐项核验，Git小状态不包含大资产。实际恢复命令见execution/restore_execution/receipt.json与RESTORE_PLAN.json，使用新输出目录。

GLB：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-REF-v1_20261105T141516Z/export/room_geometry.glb`，20332580字节，SHA256 `7197caff0b9ffc4f420b7e32283f6f17762bcf692ea1082996ca4571f5c4c931`。恢复9.146195秒/0，导出11.152782秒/0；保留对象ID节点。原生检查点仍是权威，不以GLB代替可编辑状态。
