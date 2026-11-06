# 恢复与核验

实际可编辑检查点：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FLINDERS-v2_20261106T174416Z/candidate/native/checkpoint`。
其完整文件列表、尺寸及 SHA256 见 [manifest](evidence/final/manifest.json)，入口状态与目录也已提交。完整网格保留服务器，不能仅凭 Git 中的 JSON 声称独立机器已经可恢复。资产来自固定 HSSD 目录；使用者须有对应访问权。本次没有下载新资产。

三维预览：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FLINDERS-v2_20261106T174416Z/export/room_geometry.glb`。
SHA256：`d528fb77ac5669420fc56afce7a94654530de7bbd19fe32ca5b00ce450fda433`。
这是几何预览，未新增贴图、结构或真实使用验收。

已完成恢复命令见 COMMANDS.json 中 restore_receipt 条目；RESTORE_PLAN 绑定检查点、代码和原运行环境。以后只读恢复须另选空输出目录，使用同一锁定环境执行 `scripts/920/restore_terminal_catalog.py --plan <RESTORE_PLAN绝对路径> --out <新空目录>`。该命令不调用模型或编辑。用 EXPORT_PLAN 可将相同终态重新导出到新目录。

查看结果不需要重做三次移动。不再次执行本轮 CANDIDATE_PLAN；CANDIDATE_DRAFT 含占位哈希、不是运行输入。两份 evaluation PLAN 未启动且已关闭，不能当续跑任务。

底座权重的固定 revision、四分片 SHA256 见 WEIGHT_VERIFICATION；本轮只核验权重，未调用。新 live 入口的任务评估选择已写入代码并冻结于计划，但真实模型循环仍未验收。前次 LIVEINPUT 的 CPU 检查不变。
