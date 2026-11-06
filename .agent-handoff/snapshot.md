# 当前研究状态

更新 2026-11-06T00:45:48.074345+00:00。Goal active。用户要求自主迭代到CCF A级研究，但不能把工程、更多目录或负例堆积当主线。**暂停新增来源房间，下一步是已有合理备选的条件化策略选择对照，区分知识/策略/执行瓶颈，预设研究继续/修改/放弃条件。** 原OFFLOAD4更新消费完，0新增训练授权。

## 本批准备发布

- P920-PINE-v1_20261106T002523Z完整记录已完成。新家族Pine/55sqm/Sydney/Gurney2022；窗桌/书架设计师，旧IKEA桌住户复用。研究4.8×4m迁移，不是原尺寸/历史初态；native ID是替代实例；保留要求研究施加。
- 4动作/5观察25图/4→6家具；两桌实际top面积各0.697337m²、40×30cm块通过且不重叠；两座0.4/.5/.6方形代理无冲突、两个侧向接近点均通。共有点3/3/1→3/3/2，0.6m书架仍失败。0相交/越界；客睡/膝部/舒适/容量未知，非整屋通过。
- native00:32:48.351131→00:36:34.496947UTC，226.145840s/0；restore8.846806s/0（3微小四元数往返差），export5.584923s/0，GLB9757076B/23nodes/SHAcd9dd408dfaa770630b01c0045636e94c705e556a98fe956dd4344e0a86fe018。archive.366413s/0；全部13stage终态0。无活跃模型/native进程，不重启。
- 来源账本22知识/10partial/9unique replay decisions、15来源episodes+2研究对照=17；registry27家族。规范39动作仍不变；Pine4为可审阅执行记录、未编码/未纳入训练。
- 同批ACTIONCONTRACT与STRUCTBASE已完成：22+15接口CPUchecks；12实际模型/native调用全格式通过，5add但床沙发相交/窗口损失，房间不合格；不是训练收益。rawJSON接口与官方SFT格式差异须统一。
- 本批当前待commit/push，上一核远端9f75c1891fd795a4190e4174c590cfd59518dc64。当前已过00:35:47UTC合并窗口，完成公开文件哈希核验后推送并ls-remote。排除无关extract_swiss_development_input.py。

## 下一实质研究

先读FCHOICE/FPOLICY实际证据和公平协议，不再新增房壳。使用既有双方几何合理的布局，候选顺序对换、需求反转、相同知识/预算的策略选择测量；它是开发诊断，不是封存泛化或训练收益。若底座已有正确选择，改查执行/长期控制；若同知识改善且真实执行可保持用途，再提出任务匹配训练方案。不得以测试项目事后理由作模型自己生成理由，也不得自动继续已批准四更新。

## 前批详情（历史，最新状态以上为准）

## 当前 STRUCTBASE（待普通小时批提交）

- P920-STRUCTBASE-v1_20261105T235442Z。六固定设施+7目录（3检索mesh/4程序代理），已暴露AIR来源开发；原需求/五图/对象表/回执，无示范坐标/理由/终态或评价反馈。首次该构建输入的模型尝试，不续旧米兰/Casa。
- prepare55.757938s/0、初始6641tokens。rollout真实23:56:13.151694→次日00:08:44.303420UTC，751.151750s/exit0；12/12预算用尽：6query、5add、1facing。床/沙发/桌/冷藏代理/衣架实际新增，12host+child contract及native success通过。输入6641–8889无截断，0训练，6不同状态/13观察/65图实例。
- 末11家具，原固定记录全留；床/沙发1网格相交，0越界。三宽声明window_front从通到不通，厨卫中心保持；generic末7/15、4/15、4/15，六个设施正面初态就失败不可归模型。餐椅0/2只是目录代理指标，沙发配桌进餐未验，不断言所有替代方案不可能。整屋not_accepted，无专业/学习/泛化结论。
- 终态rollout/final_checkpoint，12回执/7目录冷恢复8.595921s/0，3四元数约1e-19；export2.825142s/0，GLB5402960B/36nodes/SHAb49792ed58adbea6d98f565ddf280589dbf0c71245293bf7a6294e9cc2ce6d49；archive1.119015s/0。末5图实看、旧round003 topdown/south实看，未宣称全中间目视。全部20stage receipts exit0。
- 263公开产物/20681320B；README/STATUS/SUMMARY/INTERPRETATION/RECOVERY plans/validation/ARTIFACTS齐。3新脚本（structured_catalog_step、run_structured_catalog_policy、archive_structured_policy），旧代码不改。归档实现首query后写、评价定义输出前冻结，时序POSTPROCESS_CODE明确。
- 官方SFT think/tool_call与此rawJSON有差异：未来各组必须统一训练/推理序列化、token和mask，不能静默用旧缓存或归因为配对算法。算法/下一研究设计/论文结果已补。本批只证明可执行接口，普通工程不算创新，不再扩大格式微批作为数据前提。
- 本批与前ACTIONCONTRACT仍本地待小时合并，尚未commit/push；上次核远端9f75c1891fd795a4190e4174c590cfd59518dc64。普通窗口>=00:35:47UTC，不空等。无关Swiss未改/未stage。

## 前一 ACTIONCONTRACT（待同批提交）

- P920-ACTIONCONTRACT-v1_20261105T234424Z；独立shared_action_contract.py/check_shared_action_contract.py。新单JSON协议，exact5native schemas+current furniture/catalog引用，重复键/非有限/额外字段/不完整/多动作严格拒绝，stale context拒绝，空状态仍可query/done。旧runner/所有旧冻结代码未改。
- 22/22参数正反例；15/15真实tokenizer前缀样例（7正8负）；7旧米兰输出只读分类0/5/6拒绝、1–4通过。违规删书架/坏几何仍通过，清楚区分接口与任务/设计，不推断反事实轨迹。
- isolated dependencies14wheels/381files+hash lock，LMFE0.11.3/jsonschema4.23.0；原环境未升级。依赖13.517160s/0、guard.417003s/0、tokenizer12.861326s/0。tokenizer151669/EOS151645；原model另有EOS151643未被此默认adapter允许。原生direction语义/immutability、GPU兼容、有限token可能截断仍未由本检查解决。
- 22公开产物/74398B，README/STATUS/SUMMARY/VALIDATION/ARTIFACTS/原始回执齐。0模型/场景编辑/优化，0新专业知识/决定/训练样本。NEXT_STUDY_REVIEW加入各组共享接口约定；不是学习贡献，也不阻止专业决定数据主线。
- 尚未commit/push，上一核远端仍9f75c1891fd795a4190e4174c590cfd59518dc64。普通窗口>=2026-11-06T00:35:47UTC；下一实质批合并，不空等、不单独补发布。无关extract_swiss_development_input.py排除。

## 前一 CLOSETALT（已发布）（2026-11-05T23:34:32.223062+00:00）

- P920-CLOSETALT-v1_20261105T232013Z。已有AIR短住取衣来源，不增知识/家族/专业决定/39动作数。原文支持快速取用挂杆+床储物，不支持长期必用衣柜或原设计师否决。本轮0模型/训练/原生家具动作。
- 缓存HSSD语义筛9项按ID取首个，唯一asset0546a524f6584d5beeaf8a72b3df494f9ceeec90；网格3464B，实际.9002×.5847×2.112m。两预览实看门板在元数据背侧，原生框架+180度修正，无缩放。静态3盒/36面，内部容量/关节/扫掠/取衣未知。
- 目录7→8，原13家具不变。唯一拟remove旧rail/add衣柜同锚点yaw0，预检无包络冲突/越界，原7用途点三宽全通；按原生前门侧采5点，衣柜三宽0/5，原挂杆+Y0/-Y5。gate拒绝，0新编辑/房间观察；拟状态绝非执行终态，不旋转/换资产重试。
- 同锚点yaw不是公平使用机会，失败不能当衣柜专业负例。SOURCE_BOUNDARY/DECISION_EVIDENCE/README保留条件/动作/未知，source ledger与NEXT_STUDY_REVIEW已补缺口但training_eligible仍false。
- catalog直接冷恢复422（8.194607秒/1）因旧canonical catalog绑定；migrate_catalog_extension.py显式派生，检查原场景/task/runtime/全部旧目录/资源/历史/保留ID/指纹不变。v1误比较文件SHA与canonical hash，.215689秒/1且无输出；代码快照code/migrate_catalog_extension_v1.py和旧计划留存。v2用真实canonical_hash，.416563秒/0。新coldrestore8.545887秒/0，8条目/7旧回执/ID保留、4个约1e-19四元数差。原守卫不改、原CP不改。
- 唯一可恢复新目录../runtime/本run/migrated_v2/bundle，禁止把未迁移catalog/bundle当续跑，也不执行被拒EXECUTION_PLAN。新资产add动作尚未实测。全部进程终态，0新训练权。
- 本批已按错误/修复节点发布9f75c1891fd795a4190e4174c590cfd59518dc64，68个Git blob核验；push23:35:43.107041→23:35:47.539541UTC/4.432516秒/0，ls-remote23:36:24.342011→23:36:27.720228UTC/3.378240秒/0完全一致。../preflight/publication_closetalt_20261105/PUBLICATION.json。下一普通小时窗口>=00:35:47UTC，不空等。Swiss排除，发布后交接补记待下实质批。

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

- 前一批已核远端9b46fc10220a632241154f62aa7f1bf4bd161b0b；四批当时已合并提交推送（当前最新为上方CLOSETALT提交）。push23:15:40.426383→23:15:57.808077UTC/17.381701秒/0；ls-remote23:16:26.243842→23:16:29.321214UTC/3.077393秒/0，与本地完全相同。380个Git blob证据核验。回执../preflight/publication_milan_20261105/PUBLICATION.json。该旧窗口已被后续错误节点推送更新，当前见上方CLOSETALT。
- 新代码/论文方法/数据和评价索引/latest/六交接均已发布；无关未跟踪scripts/920/extract_swiss_development_input.py已排除。本次发布后的核验交接补记留本地下一实质批，不单独提交。权重、大型mesh、完整CP留服务器。不要force/main/AutoDL。Git断网保留同一commit，不重跑实验。
- Repo /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc；runtime在父目录/runtime。通用Python /data/wyh/conda/envs/c15_genrec/bin/python无jsonschema；native父目录/envs/scene_designer311_native/bin/python；geometry父目录/envs/geometry311/bin/python。禁止裸python。
- GPU0 UUID GPU-6df2660c-5a77-4e41-0b1a-c6208368fcd1；两锁父目录/gpu0.lock及/data/wyh/scene_cvpr_20261020/work/gpu0.lock；已用模型进程退出。新训练必须具体方案另审。

## 立即下一步

1. 共享接口CPU及实际模型/原生闭环已经有证据，停止本源房间的格式/提示重跑，不追加预算/优化。主线回到条件化专业决定数据：专业来源条件、可行选择及完整房间后果；不用继续接口微测试替代数据或学习对照。
2. 39既有候选动作/5家族/6理由位置口径不变。本批5个模型add是诊断轨迹，不自动加入专业示范/训练。AIR封闭静态目录已补但同锚点坏姿态不是公平专业负例，不重复下载/迁移。优先能被来源支持、现工具可表达的条件差异，保持失败和未知。
3. 下一正式训练须具体另审；未来各组训练/推理动作序列化一致。保留强I→R共同样本，干预仍为动作理由前缀；动作only/同知识提示/检索/几何用途检查照原设计。当前没有独立整屋学习收益/专业盲评，不标Goal完成。
4. >=00:35:47UTC后在下一个实质批合并ACTIONCONTRACT/STRUCTBASE和方法/结果/交接；核ARTIFACTS对应Git blobs后push runs/920并ls-remote。当前均本地，不误称已发布，不stage无关extract_swiss_development_input.py。

## 已发布重要负结果

- OFFLOAD原四更新已完成，无再用授权。FINALFIT-v2原三家族10动作/536目标宏NLL base.375905431、separate.370153225、paired.377761131；30冻结前向，配对未改善此直接动作拟合。v1第二adapter精度失败0前向，v2精确恢复FP32LoRA144tensors；旧Casa精度协议/结果不回写。
- Casa三组均未完成，KBASE强提示0新增家具/同知识2新增家具均未完成；不同输入上限不可冒充严格跨批公平。Moabit唯一14角色参考被连通门槛拒，0native，不证明任务无解。FHouse两条件仅一来源开发构造，不是泛化。停止这些重跑。
