# 分开训练组终态恢复

权威原生目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-COMP-v1_20261105T123924Z/separate/rollout/final_checkpoint`。完整复制manifest所列文件并逐一核字节/哈希，保留相对路径。模型、大网格和完整检查点留服务器；单独Git小状态不含全资产。

实际恢复命令见execution/cold_restore/receipt.json，绑定计划RESTORE_PLAN.json。采用固定原生Python/Scene-Designer/SceneSmith，用restore_terminal_catalog.py指向一个新输出目录，不能覆盖已存证据。本次新进程加载10.851678秒/0，资产引用与24步回执完全一致；四元数舍入最大2.1684e-19，其余字段一致。没有执行新编辑、渲染、训练或冷优化器恢复。

几何预览：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-COMP-v1_20261105T123924Z/separate/export/room_geometry.glb`，15,311,596字节，SHA256 `1652055a145475c323a0ab2e2630dcbc77dbe8ae14212fe6640247b762e07671`。GLB保留对象ID节点；材质/关节/安装未新增验收，原生场景仍是权威。原REPORT生成早于冷加载核验，补充证据见cold_restore/RESULT.json。
