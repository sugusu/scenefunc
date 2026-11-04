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

主要证据：[会话及分阶段执行](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/agent_rollout_decoupled.py)、[HTTP 入口](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/tool_server.py)、[阶段交接](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/Designer/replay_chained.py)、[SceneSmith 状态序列化](https://github.com/nepfaff/scenesmith/blob/e888297a8b0a2f3acbb8634edf9def59cceab836/scenesmith/agent_utils/room.py)。后者与本仓库 `paris/config/upstream.lock.json` 的固定来源一致。

## 三种输入不能混为一谈

| 现有文件 | 已有内容 | 尚缺什么 |
| --- | --- | --- |
| Function2Scene 生成参考 | 房壳／家具参数、名称、适配文本；+Y 向上 | 原生对象／资产身份、真实任务书、工具回执、原生 HouseLayout/SDF、专业签认。 |
| 巴黎 floor-bridge checkpoint | 原生实例和模板序列化、网格哈希、真实历史、固定房壳引用 | 明确标为 physics_ready=false；没有完整原生房壳、支撑面或完整会话。其保存格式不能改名冒充完整状态。 |
| 完整上游房间／会话 | 原生 HouseLayout、RoomGeometry、RoomScene、服务与各阶段状态 | 本机尚无已验收的专业项目导入包。公开生成参考不能静默升级为此类输入。 |

历史 dispatcher 依赖巴黎常量 `P`，并硬编码 `frozen_initial.json`、room_id、文本、房高和资产目录；现有恢复函数另核 room_id 与房壳哈希。故给它换一个输入文件不足以接入新项目。历史实现与原结果保持不变。

## 待实现的项目输入包

一个项目输入包须可独立审阅，至少包含下列内容；当前尚未声称此接口已实现：

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

[官方安装说明](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/USAGE.md)要求独立 SceneSmith 固定 checkout 与集成补丁。默认配置还依赖几何权重、材料检索和关节资产，并配置辅助 API 模型；这些与本地 Scene-Builder 不同。需要先准备隔离环境、完整依赖及有限资源方案，再启动服务。完整 checkout、补丁、资源预检和原生导入均尚未完成，没有新外部模型调用。

仅恢复本次锁定的小型文件，可用已有下载器；这不会安装整个运行系统：

```bash
python3 scripts/920/fetch_upstream.py \
  --lock research/920/project_input_v1/OFFICIAL_EXECUTION_SOURCE_LOCK.json \
  --data /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/recovered_execution \
  --receipt /tmp/p920_execution_source_recovery_new.json
```

环境复核使用 `scripts/920/inspect_runtime_environment.py --requirements <固定 requirements.txt> --out <新报告路径>`；命令仅检查，不安装包、不启动服务。
