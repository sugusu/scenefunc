# 本次原生输入与结果恢复

运行目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-MILAN-TASK-v1_20261105T224015Z`。

- 输入：`build/bundle`，包含固定16资产目录、13初始家具、11墙体/6开口；input_metadata/manifest.json记录每文件SHA256。
- 实际终态：`east/native/checkpoint`，checkpoint_metadata/manifest.json及RESTORE_PLAN绑定版本。恢复用原生Python和restore_stage/receipt.json中的命令，`--out`指定全新审阅目录；不覆盖现有结果。
- `cold_restore/restored`是已验证冷恢复副本。几何GLB在`export/room_geometry.glb`；export/RESULT含源哈希、节点与往返检查。查看GLB不等于恢复编辑会话。
- 中央候选仅有`central/preflight/PROPOSED_STATE.json`和通路结果，没有原生checkpoint/渲染，不得当实际场景。

原生Python `/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python`，几何Python `/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python`。依赖代码哈希见各PLAN；源16目录、版本及资源链见BUILD_PLAN与bundle provenance。资源依HSSD许可保留服务器，公开小文件不包含完整资产。

若完整bundle仍在，先用native_project.validate_bundle核哈希，再按RESTORE_PLAN恢复；该程序不编辑或渲染。没有重新执行实验来修复Git发布的必要。源图文未复制；模型只允许读原始任务、当前状态和观察，参考结果始终是评价侧资料。
