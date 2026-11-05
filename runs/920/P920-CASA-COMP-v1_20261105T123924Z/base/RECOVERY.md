# 底座终态恢复

权威可编辑源位于服务器：

`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-COMP-v1_20261105T123924Z/base/rollout/final_checkpoint`

将整个目录连同manifest列出的资源复制到目标机，保持目录内相对路径；先核manifest文件SHA和每个资源的字节/哈希。模型及大资源没有随Git发布，原始资产继续遵循已记录的HSSD访问范围。不能只下载house_state.json而声称资产齐全。

`RESTORE_PLAN.json`固定此检查点manifest与原生环境/脚本。`execution/cold_restore/receipt.json`保存实际命令。使用同一原生Python和固定Scene-Designer/SceneSmith代码，将新输出目录传给`restore_terminal_catalog.py`；不要覆盖已经保存的恢复证据。该入口加载原生catalog session，核对象ID、资产绑定、24步回执与退休ID记录，不启动模型/渲染/编辑。需要新的编辑时，应另建明确预算的续作，不能冒充本次结果。

本次新进程恢复11.152727秒/退出0；仅3个四元数分量出现最多1.11e-16的序列化舍入，其他原生字段一致。它验证房间能加载与身份记录保留，不是冷启动优化器恢复或可用住宅验收。原REPORT在此核验前生成，其cold_native_restore_tested=false保留；补充证据见cold_restore/RESULT.json。

可查看的几何文件：

`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CASA-COMP-v1_20261105T123924Z/base/export/room_geometry.glb`

大小13,444,980字节，SHA256 `64bb19c1b0c1a3a9ca0db4e4419d6c4b698519353738af263ff92af914de5936`。该GLB是带对象ID节点的几何预览，材质/关节/安装未新增验收，原生检查点仍是权威。
