# 当前研究状态

更新 2026-11-05T17:15:13.983203+00:00。总体 Goal active，论文/专业学习/独立泛化未完成；此前获批 OFFLOAD 四更新已全部消耗，本轮0新训练。上一轮 KBASE 完整发布属于 progress；本轮新增实际建设与来源理由候选，继续 progress。

## 最新完成

- P920-ABUILD-v1_20261105T165434Z：从6件固定厨卫设施开始，原生添加7件家具，8组五图/40图，347.457527秒/0；真实起止2026-11-05T16:55:35.905648→17:01:23.363154 UTC。native session56445已退出0，无活跃作业。
- 终态13件、0三角接触/越界、固定记录不变；.4/.5/.6宽下7固定活动点全可达；挂杆-Y每侧5点可达、+Y0。每宽9个通用目标仍失败，含床左/冰箱/挂杆正面等。静态代理、容量、人体使用、安装和隐私均未知，不称整屋专业通过。
- 终态5视图+第6步俯视实际审阅；40图hash/decode和状态绑定核验。冷恢复8.494589秒/0、7回执与资产身份通过，未做恢复后编辑。GLB 5,404,280B，SHA edc97cf1bbfb80af3343a64daccf789afeef38996ec942d2ac439fdbf798aa2d，留runtime。
- archive_execution因c15缺jsonschema退出1；保留代码和日志，改用既有native环境与显式失败记账后archive_v2通过。原生构建未重跑。summary已通过。
- 新两案来源理由候选：本次7动作+旧SBUILD9动作重新包装，每组16动作/2来源理由，14普通动作。只把来源原则放入理由目标，坐标/顺序另存。旧SBUILD/四更新材料不改。新混合token/Trainer预算未核、不训练，两案几何同模板相关，不是独立泛化数据。
- 入口 research/920/data_readiness_v1/CONSTRUCTION_CANDIDATES.json；运行SUMMARY、SOURCE_ONLY_DATASET_INDEX、DATASET_SCOPE、RECOVERY为权威。知识仍21条/25家族，没有新增独立知识或人评。

## 前序证据与发布

- KBASE fac920c63a28327def249953ca75e8d115809cc1 已推送并ls-remote核对；回执 ../preflight/publication_kbase_complete_20261105/PUBLICATION.json。strong24次查询0家具；knowledge仅床/淋浴2件，不完整，不能证知识质量或训练必要。
- 原OFFLOAD两组各2更新通过保存/重载，loss两组略升；Casa三组7/6/6家具各2接触，必要用途缺失。SBUILD九步已发布3a85da9。论文仍工作稿、主假设未证明。
- 本轮发布回执 ../preflight/publication_abuild_20261105/PUBLICATION.json；文件存在且remote_verified真才算已推送。本地代码和小日志、视图已准备，完整checkpoint/GLB留服务器。

## 下一步

1. 完成本轮关键节点正常commit/push及ls-remote；保留失败日志，不重跑七步构建。
2. 专业决定对应已有进展，下一优先增加真实条件/空间和动作覆盖，避免相同矩形模板和来源的重复包装冒充数据规模。Paper Paper或F-House的可表达子决定需落实场景条件，知识层不因完整动作欠缺归零。
3. 模型整屋任务覆盖和工具行为仍有缺口；同任务几何/用途反馈对照尚未补齐。新训练须具体数据/预算方案另审；当前可继续数据和独立任务准备，不重复tiny/加载器检查。

## 工作区

runs/920；专用SSH repo-local，保护main/AutoDL、不force。保留未跟踪 scripts/920/extract_swiss_development_input.py。新增scoped construction脚本冻结到本轮code；旧SBUILD脚本/计划未改。
