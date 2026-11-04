# 跨房间接入：固定源码核验与实施边界

2026-11-04。目标是让同一个 Scene-Designer 建设／评审流程接受真实项目的初态，保持 SceneSmith 的房壳、家具、墙面和天花分工。本轮阅读了固定版本的完整会话、工具服务和安装入口，未启动服务、模型或场景编辑。[源码清单](OFFICIAL_EXECUTION_SOURCE_LOCK.json)记录新增 13 个文件的 Git blob、大小和 SHA256；代码保留服务器。

## 已核实的接入位置

| 接口 | 固定实现中的行为 | 对本项目的影响 |
| --- | --- | --- |
| `tool_server.StartReq`、`new_session` | 接受 prompt、config、results_root、stub_assets；新建 RolloutSession 并调用 start | 当前 HTTP 请求没有已有房间／初态导入参数。不能仅把 JSON 路径写进提示，就宣称引擎已装载房间。 |
| `RolloutSession.start` | 配置并启动后端服务，建立 floor-plan agent | 为检查参数而导入／启动此类可能触发服务；当前只读源码，没有运行。 |
| `handover_after_floor_plan` | 从已生成的 HouseLayout 选择房间、派生 RoomScene，保留 house_layout、room_geometry、room_summary | 导入已知房壳应补足这组会话状态，再进入原有阶段流程。 |
| `derive_room_scene` | 用 RoomGeometry 建立 RoomScene，将墙体对象加入场景 | 只有裸家具列表或可查看 GLB 不够。 |
| `make_roomscene_env` | 原有三类 agent 共享 scene；墙面还从 HouseLayout 提取墙面，天花读取房间边界 | 不能只替换家具 scene 而让墙面与天花继续引用旧房间。 |
| `ToolEnvironment` | 每阶段从 empty_memory 开始；观察渲染五视图；每批工具完成后更新 memory | 已有对象必须有显式初态输入和可核原生状态；不能用伪造的历史成功回执填充 memory。 |
| `room_summary.format_room_summary` | 为训练／推理一致性省略墙高与窗的精确位置 | 原格式保留；准确的现场信息另从同一项目输入生成状态／需求图，不依赖摘要恢复全部几何。 |
| SceneSmith `RoomScene.to_state_dict/restore_from_state_dict` | 保存／恢复 room_geometry、objects、text_description；相对路径以 scene_dir 为基准 | 可作为原生场景状态路径，但它不是完整会话快照；项目身份、HouseLayout、资产登记、计数器与真实历史须单独核实。 |
| SceneSmith `HouseScene.to_state_dict/from_state_dict` | 同时保存 layout 与按 room_id 索引的 rooms；恢复各 RoomScene 和对应房间目录 | 已有原生顶层恢复入口，应优先使用它；HTTP 会话仍需接入选中房间及其他运行状态。 |

主要证据：[会话及分阶段执行](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/agent_rollout_decoupled.py)、[HTTP 入口](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/tool_server.py)、[阶段交接](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/Designer/replay_chained.py)、[SceneSmith 状态序列化](https://github.com/nepfaff/scenesmith/blob/e888297a8b0a2f3acbb8634edf9def59cceab836/scenesmith/agent_utils/room.py)。后者与本仓库 `paris/config/upstream.lock.json` 的固定来源一致。

## 三种输入不能混为一谈

| 现有文件 | 已有内容 | 尚缺什么 |
| --- | --- | --- |
| Function2Scene 生成参考 | 房壳／家具参数、名称、适配文本；+Y 向上 | 原生对象／资产身份、真实任务书、工具回执、原生 HouseLayout/SDF、专业签认。 |
| 巴黎 floor-bridge checkpoint | 原生实例和模板序列化、网格哈希、真实历史、固定房壳引用 | 明确标为 physics_ready=false；没有完整原生房壳、支撑面或完整会话。其保存格式不能改名冒充完整状态。 |
| 完整上游房间／会话 | 原生 HouseLayout、RoomGeometry、RoomScene、服务与各阶段状态 | 本机尚无已验收的专业项目导入包。公开生成参考不能静默升级为此类输入。 |

历史 dispatcher 依赖巴黎常量 `P`，并硬编码 `frozen_initial.json`、room_id、文本、房高和资产目录；现有恢复函数另核 room_id 与房壳哈希。故给它换一个输入文件不足以接入新项目。历史实现与原结果保持不变。

## 待实现的项目输入包

一个项目输入包须可独立审阅，至少包含下列内容。下文列完整要求；末节记录已实际完成的原生状态导入子集及其限制：

1. **身份和来源。** project_id、family_id、用途／划分、源版本、许可记录；每项事实标为住户提供、文献转述、上游适配、研究者设定或未知。研究者创建的代理 ID 明确标记，不能称作真实旧物原 ID。
2. **设计开始时的信息。** 原始需求原文及哈希；若缺失保留缺失状态，另存适配文本。整理活动、同时／先后关系、已有对象及可改属性、用户指定区域和未知项。目标布局、事后设计理由、评分专用标签留在单独文件，运行输入只按显式清单取用。
3. **原生空间。** HouseLayout、指定 room_id、RoomGeometry、初始 RoomScene、网格／材质／SDF 依赖及哈希；说明单位、坐标轴、原点和朝向。代理几何和实测几何分别登记，屋顶斜面、门扇、内柜等未表达部分必须可见地保留未知。
4. **实例和资产。** 初始对象 ID、资产 ID、局部轴、变换、固定／可动属性、真实初始状态。定义不可更换实例集合；重建同形物不能满足实例保留。原生资产登记及后续 ID 生成不得与已有对象冲突。
5. **执行与观察。** 固定源码、配置、可用工具和资产范围、全部预算；加载后实际导出状态和五视图，绑定同一状态哈希。各阶段交接同时更新 scene、layout、geometry 和摘要。原始任务每步可见，最近真实回执可见。

实现顺序是：先完成全依赖和房壳入口预检；再在独立副本中接入原生场景及会话状态；随后核对加载前后对象身份、位姿、门窗／墙面、资产路径与五视图；最后验证一次合法编辑、拒绝动作和恢复继续。每步都保留真实执行回执。涉及新模型实验时另立有限协议，训练继续单独审定。

育儿房和舞者候选仍是来源审阅材料。后者可用于检查活动切换的可表达性，但暂定目录关联、代理尺寸和缺失身份必须保留；尚未建立专业可回放决定。下一个具体动作是核全上游依赖与配置，并明确原生导入所需字段及恢复行为，而不是对新项目继续调用写死巴黎的执行器。

## 实际环境检查

`inspect_runtime_environment.py` 在既有两个解释器中实际执行，只读取发行包元数据与顶层 module spec，不导入模型或引擎。详细结果见 [ENVIRONMENT_READINESS.json](ENVIRONMENT_READINESS.json)及其中的两个带哈希报告。

| 既有环境 | 实际情况 | 限制 |
| --- | --- | --- |
| c15_genrec，Python 3.10.20 | 15 条官方包要求中 8 条满足；有模型运行栈 | Transformers、PEFT、NumPy 版本不符，部分工具服务依赖缺失；bpy、Drake、SceneSmith 未出现在该解释器搜索路径。 |
| geometry311，Python 3.11.7 | 15 条中 3 条满足，Drake 可定位 | 无模型栈及多项服务依赖；bpy、SceneSmith 未出现在该解释器搜索路径。 |

原实验另用独立 Blender 可执行文件渲染；上述结果不意味着服务器没有 Blender，也不否定原局部管线的成功。它说明不能把模型预检、原生几何调用和外部渲染的组合，当作完整官方服务已安装。现有环境未修改。

[官方安装说明](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/USAGE.md)要求独立 SceneSmith 固定 checkout 与集成补丁。默认配置还依赖几何权重、材料检索和关节资产，并配置辅助 API 模型；这些与本地 Scene-Builder 不同。初次检查时这些尚未准备；随后实际完成的源码、包安装和原生状态导入见下文。资源及完整系统仍未验收；随后有限既有物体会话接入见文末，没有新外部模型调用。

仅恢复本次锁定的小型文件，可用已有下载器；这不会安装整个运行系统：

```bash
python3 scripts/920/fetch_upstream.py \
  --lock research/920/project_input_v1/OFFICIAL_EXECUTION_SOURCE_LOCK.json \
  --data /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/recovered_execution \
  --receipt /tmp/p920_execution_source_recovery_new.json
```

环境复核使用 `scripts/920/inspect_runtime_environment.py --requirements <固定 requirements.txt> --out <新报告路径>`；命令仅检查，不安装包、不启动服务。

## 后续实际准备：P920-ENV-v1

`P920-ENV-v1_20261104T131606Z` 已取得两个固定版本的运行源码目录，放在服务器 `resources/full_runtime_20261104/`。采用 Git 部分克隆和稀疏检出，保留完整运行代码／配置目录，排除网站、演示媒体和测试数据；这不是模型／资产资源已齐备的声明。8.40 秒退出 0，两个 HEAD 正确；官方 setup 的应用前检查、应用、应用后检查均成功。原实验代码和既有环境未改。

已在 `envs/scene_designer311` 新建 Python 3.11.7 环境，并从官方 PyPI 下载／核哈希安装 uv 0.9.13。依赖解析的首次通用平台参数限制为 manylinux 2.28，与 PyMeshLab 的 2.31 安装包不匹配，退出 1。核实本机 glibc 2.35 后按实际解释器解析，39.23 秒退出 0，得到 172 包的初始哈希锁。初始解析选择了较新的主版本，故继续按 SceneSmith 自带锁约束直接依赖，并保留 Scene-Designer 明确改变的模型栈和已记录的 PEFT 版本；所有尝试分别留档。

完整源码还揭示了两个原生导入要求：`HouseScene.from_state_dict` 需要实际存在的房壳 SDF 文件；`RoomScene` 的状态字典没有保存 `_surface_id_counter` 和 room_type。首次源码审阅时只有静态风险；随后 IMPORT 运行已实际复现编号重复，并验证显式元数据恢复，见下文。

依赖进一步约束为 165 包。首次安装因 Blender 附带 Python 缺少 `Python.h` 而失败，374.50 秒退出 1；保留日志与缓存。改用 uv 管理、带头文件的 CPython 3.11.7，在另一个新环境 `envs/scene_designer311_native` 中按同一哈希锁安装，38.16 秒退出 0。未修改已有模型、几何或 Blender 环境。

官方 `preflight.py --runtime --model ...` 退出 0；Torch、Transformers、bpy、Drake 和 HouseScene 五个真实模块导入共 7.39 秒退出 0，CUDA 未初始化。**`uv pip check` 仍退出 1**：锁定的 bpy 4.5.4 安装包内部 WHEEL 标为 `cp39-cp39`，与实际 Python 3.11 不符；其实际导入通过，但未修改元数据来消除检查。完整工具服务、资源与渲染／物理未验收。详见 [ENV 结果](../../../runs/920/P920-ENV-v1_20261104T131606Z/RESULT.json)、[依赖锁](../../../runs/920/P920-ENV-v1_20261104T131606Z/runtime-requirements-pinned.lock)及每阶段真实回执。

## 已实现的原生状态导入子集

[`native_project.py`](../../../scripts/920/native_project.py) 采用固定 SceneSmith `HouseScene.from_state_dict`，不另建几何引擎，也不启动服务。输入目录含 `manifest.json`、原生 `house_state.json`、`runtime.json`、`task.json` 和显式列出的资源；可运行例子见 [合成输入包](../../../runs/920/P920-IMPORT-v1_20261104T134317Z/audit/source/manifest.json)。

- 先核文件清单、哈希、相对路径、SDF 引用、房间／对象 ID、房壳一致性、资产绑定与支撑面引用，再复制到新目录恢复。原始输入不写入；上游自动产生的 package.xml 也只在副本中处理。
- 显式补回每房间的 `room_type` 和下一支撑面编号，保留已删除对象曾占用编号后的计数；房壳和房间共用恢复后的几何实例。初始对象表和当前资产身份可单独提供给模型，没有编造旧工具回执。
- `visible_initial_context()` 只选择原始／适配需求、未知项、当前对象和资产绑定；测试中的评分专用标记未进入此输出。它尚未接入真实模型回合，不能据此声称线上信息隔离已验收。

`P920-IMPORT-v1_20261104T134317Z` 于 13:43:26—13:43:27 UTC 实际执行，1.07 秒退出 0，27 项检查通过。真实上游直接恢复后返回已有 `S_0`；适配器按输入恢复为 `S_7`，继续保存／恢复后为 `S_8`。两个不同房间的原生对象、位姿和资产绑定一致，选定房间的状态移动不改变另一个房间；缺 SDF、变更资产、对象身份错配、房壳不一致和陈旧计数等输入被拒绝。[完整结果](../../../runs/920/P920-IMPORT-v1_20261104T134317Z/RESULT.json)与[实际命令／退出](../../../runs/920/P920-IMPORT-v1_20261104T134317Z/execution/audit/receipt.json)可复核。

这些是研究者构造的两个合成房间，只有状态接口验收；原生 `move_object` 的成功不等于物理有效、工具成功或用途改善。尚无真实专业项目、HTTP 会话接入、资产检索、五视图、物理或真人评审；网格内部贴图依赖闭包尚未核验，材质重定位分支尚无动态覆盖。0 模型调用、0 渲染、0 训练。下一步是检查实际服务资源并接入完整会话，保持原任务、显式初态和真实执行回执可见；模型及训练实验仍须遵守各自授权／冻结预算。


## 默认后端资源盘点

[BACKEND_RESOURCE_INVENTORY.json](../../../runs/920/P920-ENV-v1_20261104T131606Z/BACKEND_RESOURCE_INVENTORY.json) 来自实际 Hydra 配置合成和文件检查，0.87 秒退出 0。读取的是固定默认配置，未构造 agent、启动服务或读取凭据。

默认四类资产流程选择 SAM3D 生成；家具／墙面／天花启用关节对象策略，并有辅助 VLM 与图像生成调用。配置指向的 SAM3 checkpoint、SAM3D pipeline、ArtVIP SDF／嵌入、材料库／嵌入均不在指定位置。现有另一个缓存中只有 23 件已选 HSSD 资产，逐个哈希均匹配原锁；原锁记录的 436 候选池和完整 HSSD 检索索引并未因此恢复。

源码还确认，构造家具 agent 就可能启动 Blender 和碰撞服务，不能将构造器当作纯参数检查。下一会话接入应先明确真实服务、工具和资产范围及有限预算；仅编辑既有对象或使用固定本地资产可作一个明确受限的阶段，不能称为默认完整系统或跨项目设计学习成果。


## 实际会话接入：P920-SESSION-v1／v2／v3

`existing_project_session.py` 给固定官方 tool_server 增加 `/session/existing`，使用真实 RolloutSession、StatefulFurnitureAgent 和本地 Blender／碰撞服务。当前范围是已导入物体的家具修订；显式公开真实工具 schema，保留原始需求、当前对象／资产和真实回执，观察文件记录同一状态哈希。不会默认启动资产生成／检索。前端路径通过进程内 FastAPI TestClient 验证，尚非 TCP 部署验收。

| 尝试 | 实际时间／退出 | 结果 |
| --- | --- | --- |
| v1 | 22.90 秒／1 | 7 检查通过；静态房壳与上游 world weld 冲突，0 图／工具。 |
| v2 | 69.77 秒／1 | 18 检查通过，2 次工具和 10 图；整房壳误作墙体 SDF 重复导出，断言误把请求位姿当实际位姿。 |
| v3 | 64.19 秒／0 | 新输入纠正原点和结构墙表达，20 程序检查通过；合法移动、固定墙拒绝、状态／回执／十图绑定成立。 |

[第三次结果](../../../runs/920/P920-SESSION-v3_20261104T141050Z/RESULT.json)须与[视觉复核](../../../runs/920/P920-SESSION-v3_20261104T141050Z/VISUAL_REVIEW.json)一起阅读：**北侧视图仍被墙遮挡，完整视觉验收未通过**。实际 Blender 世界包围盒确认，父节点轴旋转使上游基于局部 dimensions.z 的墙体判别失效。部分合成房只有一面墙，摘要／边界标签却列四墙，尚需修正。20 项检查及十个 PNG 不代表十张有用观察。

原生 floor placement 固定 z=0；即便 perfect 配置也有小幅位置／角度噪声，当前上下文必须读执行后的真实状态。合成输入、两次失败、精确运行代码、首日志／退出、真实工具回执和选定俯视图全部保留；大型 blend 和全部原图由 SERVER_ARTIFACTS.json 记录服务器位置、哈希及恢复命令。

这些只证明受限原生会话能编辑既有对象，没有新模型、训练、专业项目、碰撞评分或整屋用途结果。完整检查点／历史恢复尚未实现；另存的 final_house_state 是终态证据，运行目录原 house_state 仍为初态。下一步修复观察几何／摘要并验证完整恢复，随后结合可信真实项目构建可回放决定。

## 后续可视修复：P920-VIEW-v1

[VIEW-v1](../../../runs/920/P920-VIEW-v1_20261104T142847Z/VISUAL_REVIEW.json) 在同一保存的原生 blend 上实际重绘四个斜视角，7.99 秒退出 0。北侧现在能观察家具，另外三个方向保留北墙，原 blend SHA 不变。旧 SESSION-v3 的失败图片和状态未改写；本次没有会话重跑或新模型调用。

[运行补丁清单](RUNTIME_PATCHES.json)记录固定上游 HEAD、原文件／改后文件及 [patch](../../../patches/920/cardinal_world_panels.patch) 哈希。修改以世界包围盒判断薄面板，并排除侧面／后方；各图保存实际隐藏的对象。它仍是几何启发式，高薄家具和斜／曲墙未验证。当前部分房壳的四墙摘要／俯视标签仍需修正，完整五视图和检查点恢复不能据此验收。


## 已实现的协作式检查点与冷恢复

`/session/checkpoint` 接收 `{"output_directory": "<新的绝对目录>"}`。输出必须位于活跃项目目录之外，且尚不存在。它保存当前 HouseScene、房间类型／表面计数／资产绑定、原始输入副本、任务、原生记忆、全部真实工具批次、步数、噪声配置、实际代码指纹和有效配置哈希。原始输入和已有检查点不覆盖。

关闭当前会话后，另一个进程使用 `/session/existing` 的 `bundle_directory` 指向该检查点、`output_directory` 指向另一个新目录，再启动家具阶段；恢复同一实例／资产与历史，无需重发已执行工具。提示保留当前需求、实际对象和最近四批真实回执，完整历史仍在检查点。原始加载和恢复均沿用真实 SceneSmith，未另造几何引擎。

实际验收见 [CHECKPOINT_EVIDENCE.json](CHECKPOINT_EVIDENCE.json)、[最终恢复结果](../../../runs/920/P920-RESUME-v3_20261104T145552Z/RESULT.json)及各 run/execution 的完整命令。v1 首次恢复因自动运行名包含时间失败；实际配置比较只有该字段不同。v2 明确迁移原包后恢复和第三次编辑成功，随后精确状态 hash 因原生四元数往返的 2.78e-17 差异失败。两次失败、已执行动作、原包和精确代码均保留。

v3 从已保存的第三步继续，真实查询一次后保存、重新读取，14 项检查通过，12.96 秒退出 0。原始工具 hash 不改；仅 `transform.rotation_wxyz` 中 <=1e-12 的重建差异以独立非工具事件记账，位置、身份或较大的旋转变化被拒绝。所有有效算法配置仍核验，只有上游自动时间标签排除。后续源码变更须显式迁移。

五图已在 v2 的 step 2 直接复核，摘要和俯视标签只列实际墙对象；它们不是 step 3／4 的终态图片。此实现验收的是**同一固定环境中、已有对象家具阶段、已完成工具批次之间**的检查点。不包含动作执行中崩溃、模型／生成后端、其他建设阶段或任意资源依赖闭包。没有模型训练、专业样本或整屋质量结论。

两个来源候选已关联此原生入口，仍需确认初态／尺寸／资产、研究权限和独立专业复核。研究者重建 ID 必须明确标注，不能当作历史实物 ID；生成参考的终态坐标不能写入所谓原始任务。

## 完整生成参考的原生重建

随后 ROOM-v1／v2 已把舞者初稿的全部 21 件家具、6 墙及 5 门窗载入原生会话，保存五视图、一次真实查询和检查点。原房壳、家具、替代资产及碰撞提示的证据边界见 [整屋重建说明](WHOLE_ROOM_RECONSTRUCTION.md)。当前布局尚未通过用途／几何验收，生成参考及重建均不能充当专业历史动作。运行时新增两份结构观察／紧凑标签补丁，精确顺序及版本见 [补丁清单](RUNTIME_PATCHES.json)。
