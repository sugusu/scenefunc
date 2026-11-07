# 12 份可审阅终态与恢复索引

全部为研究代理几何。每格有五张原生观察、对象状态、最终审计、调用记录；GLB 是同一已验收导出的轻量分享副本。完整可编辑原生 checkpoint 留在服务器。

| 终态 | 俯视图 | 几何 GLB | 原始证据 |
|---|---|---|---|
| clear_floor_17_frozen_base_knowledge_feedback | [俯视](evaluation/clear_floor_17_frozen_base_knowledge_feedback/recorded/views/0_step_002.topdown.png) | [GLB](evaluation/clear_floor_17_frozen_base_knowledge_feedback/preview/room_geometry.glb) | [状态与回执](evaluation/clear_floor_17_frozen_base_knowledge_feedback/recorded/) |
| clear_floor_17_action_only | [俯视](evaluation/clear_floor_17_action_only/recorded/views/0_step_006.topdown.png) | [GLB](evaluation/clear_floor_17_action_only/preview/room_geometry.glb) | [状态与回执](evaluation/clear_floor_17_action_only/recorded/) |
| clear_floor_17_decision_action | [俯视](evaluation/clear_floor_17_decision_action/recorded/views/0_step_004.topdown.png) | [GLB](evaluation/clear_floor_17_decision_action/preview/room_geometry.glb) | [状态与回执](evaluation/clear_floor_17_decision_action/recorded/) |
| clear_floor_29_frozen_base_knowledge_feedback | [俯视](evaluation/clear_floor_29_frozen_base_knowledge_feedback/recorded/views/0_step_004.topdown.png) | [GLB](evaluation/clear_floor_29_frozen_base_knowledge_feedback/preview/room_geometry.glb) | [状态与回执](evaluation/clear_floor_29_frozen_base_knowledge_feedback/recorded/) |
| clear_floor_29_action_only | [俯视](evaluation/clear_floor_29_action_only/recorded/views/0_step_003.topdown.png) | [GLB](evaluation/clear_floor_29_action_only/preview/room_geometry.glb) | [状态与回执](evaluation/clear_floor_29_action_only/recorded/) |
| clear_floor_29_decision_action | [俯视](evaluation/clear_floor_29_decision_action/recorded/views/0_step_005.topdown.png) | [GLB](evaluation/clear_floor_29_decision_action/preview/room_geometry.glb) | [状态与回执](evaluation/clear_floor_29_decision_action/recorded/) |
| desk_work_17_frozen_base_knowledge_feedback | [俯视](evaluation/desk_work_17_frozen_base_knowledge_feedback/recorded/views/0_step_006.topdown.png) | [GLB](evaluation/desk_work_17_frozen_base_knowledge_feedback/preview/room_geometry.glb) | [状态与回执](evaluation/desk_work_17_frozen_base_knowledge_feedback/recorded/) |
| desk_work_17_action_only | [俯视](evaluation/desk_work_17_action_only/recorded/views/0_step_005.topdown.png) | [GLB](evaluation/desk_work_17_action_only/preview/room_geometry.glb) | [状态与回执](evaluation/desk_work_17_action_only/recorded/) |
| desk_work_17_decision_action | [俯视](evaluation/desk_work_17_decision_action/recorded/views/0_step_005.topdown.png) | [GLB](evaluation/desk_work_17_decision_action/preview/room_geometry.glb) | [状态与回执](evaluation/desk_work_17_decision_action/recorded/) |
| desk_work_29_frozen_base_knowledge_feedback | [俯视](evaluation/desk_work_29_frozen_base_knowledge_feedback/recorded/views/0_step_004.topdown.png) | [GLB](evaluation/desk_work_29_frozen_base_knowledge_feedback/preview/room_geometry.glb) | [状态与回执](evaluation/desk_work_29_frozen_base_knowledge_feedback/recorded/) |
| desk_work_29_action_only | [俯视](evaluation/desk_work_29_action_only/recorded/views/0_step_004.topdown.png) | [GLB](evaluation/desk_work_29_action_only/preview/room_geometry.glb) | [状态与回执](evaluation/desk_work_29_action_only/recorded/) |
| desk_work_29_decision_action | [俯视](evaluation/desk_work_29_decision_action/recorded/views/0_step_005.topdown.png) | [GLB](evaluation/desk_work_29_decision_action/preview/room_geometry.glb) | [状态与回执](evaluation/desk_work_29_decision_action/recorded/) |

各 checkpoint 路径、manifest 哈希和 GLB 哈希见 [RECOVERY.json](RECOVERY.json)。各格 `EXPORT_PLAN.json` 绑定了原生恢复输入和导出代码。已完成的导出与往返校验见 `EXPORT_RESULT.json`；这些是查看／恢复依据，不需要再次调用模型。

若需要重新导出查看，使用对应 EXPORT_PLAN 和新的输出目录：
```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python \
  scripts/920/export_catalog_geometry.py --plan <cell/EXPORT_PLAN.json> --out <new-output-directory>
```

模型版本、两个适配器的绝对路径及哈希见 [最终验收](review/FINAL_ACCEPTANCE.json)；原生执行恢复使用各格 `PLAN.json` 和保存的 checkpoint。本次作业已完成，不应再次启动一次性评测编排器。
