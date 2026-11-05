# 恢复记录

完整资产、GLB和可编辑检查点留服务器；以下命令记录不重新生成结果。

- perimeter: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v2_20261105T193428Z/perimeter/native/checkpoint`，manifest SHA256 `762f58ad83fd39f362da270c6d2cc12f70b3758575521e748984e51972dad641`。
  GLB `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v2_20261105T193428Z/perimeter/export/room_geometry.glb`，SHA256 `f8e5a109c3b5c1997202d68fd3d0ffbfbb5f400710f6bcff2420becef7a73e8a`。
- divider: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v2_20261105T193428Z/divider/native/checkpoint`，manifest SHA256 `a1ca1d082c5616fd02fed80ea5969de5aea09ab0fde2cadf5840070cc164b551`。
  GLB `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FCHOICE-v2_20261105T193428Z/divider/export/room_geometry.glb`，SHA256 `2ad531d0daf3f8810671f1e06cf6bb00eb1c945c4234ed07982647536e599a0f`。

各候选 RESTORE_PLAN.json 和 execution 下的实际命令给出固定原生环境及冷加载流程。复查选择新输出目录，不能覆盖既有receipt或结果。原项目版本、上游运行库与资产指纹在checkpoint元数据中；本机丢失大型资产时须按绑定来源重新取得对应授权资产，Git中的哈希不能重建资产本身。
