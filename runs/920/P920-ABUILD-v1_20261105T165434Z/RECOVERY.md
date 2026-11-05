# 查看与恢复

服务器几何：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-ABUILD-v1_20261105T165434Z/export/room_geometry.glb`

SHA256：`edc97cf1bbfb80af3343a64daccf789afeef38996ec942d2ac439fdbf798aa2d`，5404280 字节。此 GLB 是静态展示；对象身份与编辑恢复使用完整原生检查点。

原生检查点：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-ABUILD-v1_20261105T165434Z/native/checkpoint`

manifest SHA256：`db458b5def740b571dd87a9ee4a3ca40a4d7c00eac5d257db2348a0f130e13c2`。源资产目录来自 BUILD_PLAN.json 的 base_manifest；版本、输入哈希与新建身份映射见 input_metadata 和 checkpoint_metadata。恢复与导出实际命令见 COMMANDS.md；这些命令已经运行，不应覆盖其输出目录。将完整检查点复制到新位置并按 manifest 逐文件验哈希，再在 RESTORE_PLAN 中绑定新路径/manifest 并使用新的输出目录。

本次新进程只读恢复已通过；未执行恢复后再次编辑测试，不宣称优化器恢复。原始渲染、来源重建、权重、完整资产与 checkpoint 不进入 Git。
