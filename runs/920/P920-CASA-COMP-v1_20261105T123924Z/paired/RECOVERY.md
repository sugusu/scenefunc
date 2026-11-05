# 配对组终态恢复

原生目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-COMP-v1_20261105T123924Z/paired/rollout/final_checkpoint`。恢复完整manifest列出的文件并逐项核哈希；Git小状态不含全资产。实际命令见execution/cold_restore/receipt.json和RESTORE_PLAN.json，使用新输出目录避免覆盖。

新进程加载8.845002秒，退出0，所有对象ID、资产和24回执一致；四元数最大舍入2.1684e-19，其余字段一致。没有新增编辑、模型调用、渲染或冷优化器恢复。

GLB：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-COMP-v1_20261105T123924Z/paired/export/room_geometry.glb`，15309664字节，SHA256 `c53af514d30c4e40913a11370e469b1bdcc39cbbdef2b8c0581eaa14dcf15efa`，保留对象ID节点。权威状态仍为原生检查点；材质、关节、安装和真实用途未新增验收。原REPORT早于冷加载，补充证据见cold_restore/RESULT.json。
