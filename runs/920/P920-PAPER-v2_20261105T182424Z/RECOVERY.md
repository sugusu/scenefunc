# 恢复与查看

原生checkpoint：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-PAPER-v2_20261105T182424Z/native/checkpoint`。manifest SHA256：`4769930bf3f8709f31d31d55aaead15fba56cbbf4530367897bf43ac504bd1be`。恢复计划RESTORE_PLAN.json，已执行命令及退出见COMMANDS.md，使用新输出目录可只读复核，不应覆盖原结果。

静态GLB：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-PAPER-v2_20261105T182424Z/export/room_geometry.glb`，3537436字节，SHA256 `65b0c3453fa99d80b0ec773b1ee6c40036bc1a8398fa15e9d27a81848bdde071`。完整资产、检查点和55张原始五视图留在服务器；Git保存可审阅状态、小日志、绑定/哈希和7张关键图。

输入资产/房壳来自v1参考构建，版本与哈希见BUILD_PLAN、REFERENCE_BUILD_PLAN、input_metadata/manifest；v2复用同一bundle，仅冻结不同的停椅动作。恢复检查已保留对象、资产与回执，未验证后续编辑或实际家具机械运动。
