# 恢复与资源

权威可编辑检查点：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-SBUILD-v1_20261105T143623Z/native/checkpoint`，manifest SHA `35e8927fe865c7c88af475704df4354fb07feaf024ac0a3617507b5f1ab3c3d6`。初态：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-SBUILD-v1_20261105T143623Z/build/bundle`。Git 的 input_metadata/checkpoint_metadata 是小型索引与状态；按 manifest.files 获取服务器资源并逐项核验，不能只复制 JSON 假称已恢复完整资产。

GLB：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-SBUILD-v1_20261105T143623Z/export/room_geometry.glb`，5257036 字节，SHA `2059d7c63bf3d4bfd444fe8336a0e9b63db93567e4132655a56bbf240fac3a5e`。它保留对象节点，但原生检查点才是编辑恢复依据。新进程恢复命令与真实退出见 execution/restore_execution/receipt.json，后续检查应使用新输出目录。

候选数据原位置 `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-SBUILD-v1_20261105T143623Z/dataset_v2`；episode 路径相对 `/data/wyh/scene_cvpr_20261020/design_decisions_920`。完整复制 runtime 下此运行和相关 resources，或依据 ARCHIVE_CHECKS.json 的 source→destination 映射恢复 Git 中的小文件/图片到原运行目录。不得改写旧文件后继续沿用其哈希。构建输入原来源、版本与哈希在 BUILD_PLAN.json，登记时快照在 REGISTRY_AT_BUILD.json；实时 registry 若已更新需将快照放到单独恢复副本并声明路径修订，不能覆盖当前登记表。

官方加载器及 processor 路径/哈希在 ENCODING_PLAN_V2.json，数据包装使用 PACKAGE_PLAN_V2.json。权重未加载、无新适配器，不需要为此恢复训练。脚本和实际命令见 code/、COMMANDS.md；首次两次后处理失败保留，不使用失败版本生产数据。
