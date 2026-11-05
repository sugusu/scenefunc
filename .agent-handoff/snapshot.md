# 当前研究状态

更新 2026-11-05T23:34:32.223062+00:00。Goal active。本轮progress：已有取衣来源的静态封闭资产接入、使用面预检拒绝及显式检查点目录迁移完成。**全部作业已终态，无模型或原生进程待续。** 旧OFFLOAD四更新已消费，0新训练授权。

## 当前 CLOSETALT（2026-11-05T23:34:32.223062+00:00）

- P920-CLOSETALT-v1_20261105T232013Z。已有AIR短住取衣来源，不增知识/家族/专业决定/39动作数。原文支持快速取用挂杆+床储物，不支持长期必用衣柜或原设计师否决。本轮0模型/训练/原生家具动作。
- 缓存HSSD语义筛9项按ID取首个，唯一asset0546a524f6584d5beeaf8a72b3df494f9ceeec90；网格3464B，实际.9002×.5847×2.112m。两预览实看门板在元数据背侧，原生框架+180度修正，无缩放。静态3盒/36面，内部容量/关节/扫掠/取衣未知。
- 目录7→8，原13家具不变。唯一拟remove旧rail/add衣柜同锚点yaw0，预检无包络冲突/越界，原7用途点三宽全通；按原生前门侧采5点，衣柜三宽0/5，原挂杆+Y0/-Y5。gate拒绝，0新编辑/房间观察；拟状态绝非执行终态，不旋转/换资产重试。
- 同锚点yaw不是公平使用机会，失败不能当衣柜专业负例。SOURCE_BOUNDARY/DECISION_EVIDENCE/README保留条件/动作/未知，source ledger与NEXT_STUDY_REVIEW已补缺口但training_eligible仍false。
- catalog直接冷恢复422（8.194607秒/1）因旧canonical catalog绑定；migrate_catalog_extension.py显式派生，检查原场景/task/runtime/全部旧目录/资源/历史/保留ID/指纹不变。v1误比较文件SHA与canonical hash，.215689秒/1且无输出；代码快照code/migrate_catalog_extension_v1.py和旧计划留存。v2用真实canonical_hash，.416563秒/0。新coldrestore8.545887秒/0，8条目/7旧回执/ID保留、4个约1e-19四元数差。原守卫不改、原CP不改。
- 唯一可恢复新目录../runtime/本run/migrated_v2/bundle，禁止把未迁移catalog/bundle当续跑，也不执行被拒EXECUTION_PLAN。新资产add动作尚未实测。全部进程终态，0新训练权。
- 本批含实际检查点恢复阻断及修复，准备按错误节点合并一次commit/push；若未完成，以实际publication回执/远端为准。原最新9b46fc1依旧有效。排除Swiss未跟踪文件。

## 最新 MILAN-BASE

- run P920-MILAN-BASE-v1_20261105T225639Z。输入来自MILAN-TASK同初态，13家具/16目录/五图；不见来源成品理由、参考动作/坐标/末态或评价反馈。BF16 frozen base，无adapter，仅GPU0。12模型/工具预算、输出1024、输入16384、30分钟总上限，0重试/训练。
- prepare57.113084秒/0，初始14111tokens；全部七输入14111–15431未截断。参考评价校准v1错误checkpoint路径.717730秒/1保留；v2正确checkpoint5.535114秒/0，未改代码/阈值，未重跑场景。
- rollout真实23:01:01.903080→23:08:00.618269UTC，418.715213秒/**退出1**，session86216终态。7模型/6工具/4实际改变；第7次1024token未闭合查询，原文保留、未执行。禁止以剩余5次预算重启。
- 顺序：schema外参数查询被原生接受；删除retained_shelf_0；新增同资产bookshelf_0；移动living_sofa_0；加coffee_table_0；删除不存在retained_desk_0失败；最后截断。家具13→14，原实例12/13，固定记录保留。
- 终态1对coffee_table_0/living_sofa_0网格相交、0越界。动态14用途点初14/14三宽全通，末13/14(.4)、12/14(.5/.6)，缺原书架点始终在分母，较宽另失南餐椅侧方。未找到预定义独立办公桌椅，不等于所有共享桌/沙发方案不可能。
- FINAL_AUDIT混合旧固定探针和通用点；其旧sofa_front失败不等于当前沙发用途点失败。主解释EVALUATION对象相对点。舒适/采光/容量/设施/完整整屋使用未知，无专业/学习或泛化结论。
- 七观察35图实例hash/decode/state绑定，五种不同native状态；最后五图已目视。最后真实checkpoint为rollout/round_005/checkpoint，没有final_checkpoint。独立评价5.285028秒/0，网格5.435548/0，coldrestore9.047005/0，export13.012491/0，archive.517364/0。
- 冷恢复资产、身份、六回执保留，两四元数约1e-19舍入。GLB23,869,956B/60nodes/SHA0bb193735c70fe07fdda275ea20d646797b32e2b0496e7a7e06c3af106d344f7，serveronly。输出1378tokens/输入103487tokens，模型加载33.068秒/生成93.265秒。README/RECOVERY/INTERPRETATION有边界。

## 同批已发布的已完成工作

- MILAN-TASK P920-MILAN-TASK-v1_20261105T224015Z：SET architects居住兼办公，原文2023/70m²。test分配先于全文但已见搜索成品，sealed=false；家族禁训练/检索。10×7gross/shaft/卧室卫浴/13家具均研究迁移，非原CAD。固定两候选：east两add实际完成，central .6四点不通被拒0执行，不调位。east15家具、0mesh/outside、14原角色+办公侧三宽通、材料块覆盖与15无挡线段；椅朝向离开沙发、非真实视野/舒适。131.542830秒/0、15图/冷恢复/GLB齐，103公开文件。
- LIFECYCLE P920-LIFECYCLE-v1_20261105T221829Z：Michelet已有train，第8来源决定。研究3→2人/家庭变化，remove旧床/add桌/add椅三步157.042025秒/0。原15其他家具保留，16→17，0mesh/outside，15声明点三宽通，20图/恢复/GLB齐。原冻结former_bed_side其实西床端，床两侧.323m、初三人睡眠未知；不改旧指标，只留语义纠正。117公开文件，3动作/1理由候选未官方编码，0训练准入。
- STUDYMAP P920-STUDYMAP-v1_20261105T223531Z：两候选清单39既有动作24add/14move/1remove、5家族/6理由位置/5独特理由/33普通动作，未合并/未重新编码。六理由锚点两组均已有完整同场景I→R，主干预是动作的理由前缀，不是有无所有项目对应。Registry四家族回放漏计已按原文件纠正，不增加执行，10公开文件。
- 当前知识21、部分工具候选9、源回放14/总16含重复版本与2研究对照，8唯一来源决定、0独立专业评审、0训练准入。米兰参考/模型不进入训练候选计数。26家族=13dev+11train来源+2exposedtest。

## 发布与工作树

- 最新已核远端9b46fc10220a632241154f62aa7f1bf4bd161b0b；四批已合并提交推送。push23:15:40.426383→23:15:57.808077UTC/17.381701秒/0；ls-remote23:16:26.243842→23:16:29.321214UTC/3.077393秒/0，与本地完全相同。380个Git blob证据核验。回执../preflight/publication_milan_20261105/PUBLICATION.json。下一普通小时窗口>=2026-11-06T00:15:57UTC，不为空等窗口。
- 新代码/论文方法/数据和评价索引/latest/六交接均已发布；无关未跟踪scripts/920/extract_swiss_development_input.py已排除。本次发布后的核验交接补记留本地下一实质批，不单独提交。权重、大型mesh、完整CP留服务器。不要force/main/AutoDL。Git断网保留同一commit，不重跑实验。
- Repo /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc；runtime在父目录/runtime。通用Python /data/wyh/conda/envs/c15_genrec/bin/python无jsonschema；native父目录/envs/scene_designer311_native/bin/python；geometry父目录/envs/geometry311/bin/python。禁止裸python。
- GPU0 UUID GPU-6df2660c-5a77-4e41-0b1a-c6208368fcd1；两锁父目录/gpu0.lock及/data/wyh/scene_cvpr_20261020/work/gpu0.lock；已用模型进程退出。新训练必须具体方案另审。

## 立即下一步

1. 四批发布已完成并核远端（见上），无需再推同一提交。保留本次未闭合输出、要求违例和首错停止；不要重跑米兰、已有编码/材料盘点/FINALFIT或同壳摆位。
2. 主线回到条件化专业决定与公平学习研究。39动作主要仍为执行示范，尚不能隔离专业策略适用性；NEXT_STUDY_REVIEW中的具体缺口（如开放收纳案例没有可执行封闭备选）是有用下一工作单位。围绕已登记来源核适用条件/实际可行备选，避免不断增加相似房壳与轨迹数。
3. 独立任务现在有一次实测失败，不等于完整评价ready；动作接口、身份、几何错误与专业理由缺失不可混因。保留强分开/配对同知识同动作对照，错配仅辅助诊断。正式训练、专业盲评/跨项目收益和完整论文证据尚缺，不标Goal完成。

## 已发布重要负结果

- OFFLOAD原四更新已完成，无再用授权。FINALFIT-v2原三家族10动作/536目标宏NLL base.375905431、separate.370153225、paired.377761131；30冻结前向，配对未改善此直接动作拟合。v1第二adapter精度失败0前向，v2精确恢复FP32LoRA144tensors；旧Casa精度协议/结果不回写。
- Casa三组均未完成，KBASE强提示0新增家具/同知识2新增家具均未完成；不同输入上限不可冒充严格跨批公平。Moabit唯一14角色参考被连通门槛拒，0native，不证明任务无解。FHouse两条件仅一来源开发构造，不是泛化。停止这些重跑。
