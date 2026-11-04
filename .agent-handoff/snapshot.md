# 当前研究状态

- 更新：2026-11-04 14:20 UTC 左右。整体研究 Goal active，有实质进展；训练须单独审定。主线是原生系统、可信同项目决定、公平配对学习及独立整屋评价。
- 最后已核验远端 runs/920 为 d796cac4c5b084a85f06a6875d5a70bbcfa92fdd。ENV／IMPORT／SESSION 正准备小时合并发布；14:11:15 UTC 限制已过。精确发布状态见 ../GIT_PUBLICATION_STATUS.json，提交后正常 push 并 ls-remote 核验，不重跑实验。
- ENV：固定 Designer a0565e5、Smith e888297 及官方集成，独立 CPython 3.11.7 环境 ../envs/scene_designer311_native，165 包哈希锁。保留 Python.h 缺失等全部安装失败；官方 preflight 和真实模块导入通过。bpy 内部 cp39 标签令 uv pip check 退出 1，但实际 CPU 渲染成功；未修改元数据掩盖问题。
- IMPORT：P920-IMPORT-v1_20261104T134317Z，1.07 秒退出 0，27 项合成两房间状态检查。真实 HouseScene.from_state_dict，恢复 room_type／支撑面计数／共享几何，源输入独立复制。历史 fixture 只验状态，随后 SESSION 发现其 SDF 和对象原点不适合原生导出；旧输入不改。
- SESSION-v1 22.90 秒退出 1：房壳 static 和上游 world weld 冲突，7 项通过、0 图／工具。v2 69.77 秒退出 1：18 项通过、2 工具／10 图；北墙引用整房壳造成重复几何，位姿断言误把请求值当实际结果。原始代码、输入和失败全部存证。
- SESSION-v3 P920-SESSION-v3_20261104T141050Z：14:11:32—14:12:36 UTC、64.19 秒退出 0，20 项程序检查通过。真实 RolloutSession／StatefulFurnitureAgent／本地 Blender 和碰撞服务；FastAPI 路由通过进程内 TestClient 调用，前端 TCP 未验收。真实移动 table_0，拒绝移动固定墙，状态哈希与回执一致，前后各五图。没有模型／训练／付费调用，服务已关闭。
- 视觉验收仍失败：v3 重复几何已消除，但北侧图被北墙挡住；真实 blend 检查证明父节点轴旋转使 local z≈0.05，而 world height≈2.5，原渲染器漏判。部分合成房只有一墙，摘要及图上却标四墙，必须修正。20/20 和十个 PNG 不代表完整五视图或房间可用。
- 下一步：修正世界坐标墙体／相机侧判断并以保存的 blend 作有限验证；修正摘要和边界标签与真实几何的对应；实现保存完整会话／运行元数据／真实历史后恢复继续。目前 final_house_state 已保存，但运行目录原 house_state 仍初态，禁止当恢复检查点。冻结脚本与失败版本保留，后续独立 run，不覆盖旧证据。
- 缺资源：默认 SAM3／SAM3D、ArtVIP SDF／嵌入、材料库／嵌入；只有 23 件已选 HSSD 资产哈希复核通过，非完整检索池。现阶段仅既有物体家具编辑，没有资产获取、墙／天花阶段和完整系统验收。
- 数据：官方 viewer 33 目录／336 快照，30 终态／572 家具，其中带适配需求 29／545；计数对应不证明 Windows 原包字节一致。7 卡、165 图、原柜 result.json 仍缺。5 已看家族归开发，2 来源支持候选，0 合格专业训练决定；dancer 别名未确认。补许可、真实初态／ID、可回放决定和专业复核。
- DATA／STATE／REVIEW 工程检查已通过，84 对 92 token 不公平问题未解决，专业正向准入未验收；独立整屋评价仍草案。巴黎 DD 六场景整屋成功 unknown；TC 四生成一次截断，动作未执行／评分。不得扩充微实验替代科研主线。
- 无活跃作业。最新状态 runs/920/latest.json；各 run/execution 保存真实首日志、起止和退出。全研究未完成。
