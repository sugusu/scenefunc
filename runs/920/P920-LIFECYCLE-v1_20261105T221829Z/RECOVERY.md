# 恢复本次实际产物

服务器运行目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-LIFECYCLE-v1_20261105T221829Z`。原生输入在`build/bundle`，终态完整可编辑checkpoint在`native/checkpoint`，冷恢复副本在`cold_restore/restored`，可查看GLB在`export/room_geometry.glb`（23,974,216 bytes，SHA256 `e7d2615226df9bd44798336fee6b67b792ef27b2e549f070fded0c5b5079ba83`）。GLB是几何查看产物；继续编辑使用checkpoint。

先校验input_metadata/manifest.json或checkpoint_metadata/manifest.json中的每个相对文件哈希；它们含实际资产目录、绑定、状态和运行依赖。资产来自BUILD_PLAN记录的本机固定16目录，来源版本沿该目录provenance追溯，HSSD访问许可仍适用。完整网格留服务器，公开文件不能单独恢复缺失资产。

原生Python：`/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python`。恢复命令与依赖版本见RESTORE_PLAN.json和restore_stage/receipt.json；如需再次审查，将`--out`指向新的审阅目录，不覆盖本次结果。该工具只恢复，不编辑/渲染。导出命令见export_stage/receipt.json，用geometry311环境，新输出目录。52几何节点往返最大边界误差2.36e-7m。

运行中间的20张观察保留在native/house/room_michelet_lifecycle_transfer/rollout_furniture。ARCHIVE_CHECKS核验四状态、20图和三步合法参数。所有实际启动命令保存在各stage回执，勿重新运行实验以修复Git发布。
