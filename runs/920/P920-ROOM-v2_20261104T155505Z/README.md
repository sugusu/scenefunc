# P920-ROOM-v2：完整初态观察、冲突归因和恢复

实际原生会话载入 21 家具＋6 墙＋5 门窗，共 32 对象；五视图、一次真实状态查询和检查点完成。完整阶段耗时 217.31 秒，含两次检查点验证失败。没有设计编辑、模型调用或训练。

`RESULT.json` 是汇总；原始子阶段结果保持原样，后续人工看图结论单列 `VISUAL_REVIEW.json`。五张原图均已直接核查，紧凑俯视另存，未改场景或原图。原生观察通过 8 项程序检查，最终检查点恢复通过 10 项。`STATUS.json` 包含计划／完成数、配置和输入哈希、失败及下一步。

当前有 8 个边界提示和 8 对家具包围盒重叠：2 对原生成初稿已有、6 对替换后出现。`ENVELOPE_ATTRIBUTION.json` 保存统一口径的前后对照，原生侧数值与真实会话量测一致。包围盒提示不是实际碰撞或用途失败；墙、开门扫掠与活动净空尚未测。本初态未通过设计质量验收，不能用来证明模型损害用途或学习收益。

## 保留的失败

- 父运行 v1 不能解析 `package://scene`。`bundle_migration.json` 记录新副本中 17 个 SDF 网格引用改为相对路径，状态、几何及任务字节不变。
- `checkpoint_migration` 在沙箱中等待 TestClient 请求 120 秒后超时（退出 124），没有回执；根因未进一步隔离。
- `checkpoint_migration_host` 已导入相同状态和历史，但 `skip_render=True` 被现有会话合同拒绝，退出 1。
- `checkpoint_restore` 复用已迁移检查点，启用真实渲染接口而不请求观察，13.87 秒退出 0。完整状态、原生记忆、步数、对象／资产与输入保留；没有重复迁移、渲染或工具调用。

## 从现有结果继续

最新可恢复目录：

`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-ROOM-v2_20261104T155505Z/checkpoint_migration_host/checkpoint`

manifest SHA256：`ad55691c20d2a767f3f816437457e5b5b11a997fd51d535881e6dd005801786a`。原始 step 1 检查点也保留。紧凑标签修改仅影响一个 renderer 文件，通过明确迁移更新其代码指纹；旧包不覆写。

1. 核 `SERVER_ARTIFACTS.json` 中 manifest 及其所有文件，使用 ENV 已冻结的独立 CPython 3.11.7 环境。
2. 固定 Designer `a0565e50ed7d32c0e09132b9311c0b9b91165405`、Smith `e888297a8b0a2f3acbb8634edf9def59cceab836` 及官方集成。按 `research/920/project_input_v1/RUNTIME_PATCHES.json` 的顺序应用四份补丁并核最终文件哈希。
3. 使用 `scripts/920/existing_project_session.py` 的 `create_app`；`POST /session/existing` 传上述 `bundle_directory` 和不存在的新 `output_directory`，然后 `POST /session/stage` 传 `{"stage":"furniture","skip_render":false,"stub_assets":false}`。真实 HTTP 路由调用示例在 `code/restore_room_label_checkpoint.py`，其运行命令在 `execution/checkpoint_restore/receipt.json`。本次验证为 TestClient／本机后端，非对外 TCP 部署。
4. 恢复后停在 step 1；下一次动作应从该状态开始，不重发历史查询或编辑。要复核恢复脚本，只替换命令的输出目录，保留计划内源 manifest；不得覆盖已有产物。

大型 blend、完整 checkpoint 和网格留在服务器，`SERVER_ARTIFACTS.json` 给出路径与内容索引。可查看的 Blender 文件在其中 `step_000.blend` 记录。完整请求／上下文也留服务器；`HTTP_SUMMARY.json` 保留调用与返回哈希，`execution/` 保留小型实际日志。

Git 中 `views/` 为原俯视／南视，`compact_topdown/` 为同一几何的紧凑俯视及名称→原生 ID 映射。图像使用 HSSD 固定来源网格（CC BY-NC 4.0 及提供者条款），具体资产 ID、版本和哈希由父运行计划和 `paris/config/assets.lock.json` 标识；这是研究者的坐标／代理重建及渲染，未转载原报道图片。四个未匹配家具、实际重量、来源许可／专业复核等限制仍见整屋研究说明。
