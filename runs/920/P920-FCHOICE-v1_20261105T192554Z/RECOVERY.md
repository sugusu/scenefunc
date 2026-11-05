# 恢复记录

完整资产、GLB和可编辑检查点留服务器；以下命令记录不重新生成结果。

- perimeter: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v1_20261105T192554Z/perimeter/native/checkpoint`，manifest SHA256 `4162e6f14916108fe460a12fd7ddb7a63e8a828d662baea1d730bf2c5dc13ec8`。
  GLB `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v1_20261105T192554Z/perimeter/export/room_geometry.glb`，SHA256 `b0dccd4a1d3115d7c0000773adb68673d539aa2f3bf365fd48e0bd88984bc75a`。
- divider: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v1_20261105T192554Z/divider/native/checkpoint`，manifest SHA256 `9bae515cc2b9e8db4c606c2d5520181ef9e96be1878bcdd496af61ff692e1156`。
  GLB `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v1_20261105T192554Z/divider/export/room_geometry.glb`，SHA256 `f99ac3a9e245ebc6a52e836cbc2adce5b20b9f59148b7f1a30112ab33e3a4303`。

各候选 RESTORE_PLAN.json 和 execution 下的实际命令给出固定原生环境及冷加载流程。复查选择新输出目录，不能覆盖既有receipt或结果。原项目版本、上游运行库与资产指纹在checkpoint元数据中；本机丢失大型资产时须按绑定来源重新取得对应授权资产，Git中的哈希不能重建资产本身。
