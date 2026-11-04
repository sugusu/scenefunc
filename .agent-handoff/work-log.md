# 最近工作

## 2026-11-04 长期研究启动

- 已创建线程 active Goal，核对官方 Goals 文档和可用工具；独立 automation_update 不可用，未创建其他调度。
- 建立 AGENT_HANDOFF.md、AGENTS.md、AGENT_SESSION_PROMPTS.md 与 .agent-handoff/，填入真实研究状态和后续队列。
- 新增 research/920/RESEARCH_GOAL.md，定义论文目标及七项验收条件。
- 新增并运行 scripts/920/audit_tool_contract.py，只读取 v1 请求、调度器和真实回执，输出 research/920/tool_contract_audit_v1.json。确认 3/9 次调用、3/4 个回合受到未接入 check_physics 影响；完整提示有前后冲突。
- 未运行新的模型、渲染或训练，未修改历史实验产物。
- 正常重试 git push origin runs/920，仍返回 publickey 失败；保持本地历史并继续 CPU 工作。

## 2026-11-04 协议诊断准备

新增 prepare_contract_diagnostic.py；为现有 guarded model、执行、评分脚本增加独立 --run-id（默认仍为历史 v1）。状态路径隔离的模拟检查通过，原始历史结果树未修改。新 run P920-TC-v1_20261104T104227Z 的 CPU 准备退出 0，耗时 5.79 秒，未初始化 CUDA。四输入相对 v1 均减少 253 tokens，用户消息和图片哈希不变。新协议在任何模型调用前冻结，四正式回合加一次 32-token 预检，无重试。

## 2026-11-04 总体架构与算法目标修正

- 按用户给出的系统／数据／训练算法重写 RESEARCH_GOAL.md，新增 GOAL_TEXT.txt 与 GOAL_SYNC.json；主线改为完整系统、可信项目决定、公平配对学习、独立整屋评价及论文验收。训练仍须单独审定。
- 实际 Goal paused，旧 objective 未变，API 无编辑能力；明确区分仓库修改和界面同步，未标记研究完成或另建 Goal。
- 更新 AGENTS、会话提示及交接队列，避免下一会话继续扩大巴黎诊断而遗漏主线。
- 归档此前已生成的诊断记录：1 次预检和 4 次正式调用已结束；正式批次 10:47:17—10:49:39 UTC，退出 1（B_G 达 token 上限）。保存启动、首日志、退出、原始输出、提案及哈希；本轮没有新增模型调用、场景编辑或训练。

## 2026-11-04 整体研究 Goal 创建成功

用户已删除旧 Goal；先核验 get_goal 为 null，再用 GOAL_TEXT.txt 创建新目标，返回 active 并再次读取确认。正文与文件一致（接口去除末尾换行），未另设 token 预算。同步目标说明、JSON 状态、风险、队列及交接，解除手动编辑／恢复待办；本轮未启动模型或训练。后续优先核验项目来源／权限、三维可表达性及同项目决定证据。

## 2026-11-04 来源与实际加载器审计

读取本机输入、远端三个分支及 main 树、原育儿房访谈；新增来源清单、两项目家族登记、一个不合格候选和七类覆盖账。发现原柜目录的空壳与巴黎同哈希，旧专业包未定位，已询问可读路径。固定下载 7 个官方 SFT 文件并核 Git blob／SHA256；源码只留服务器。新增 audit_sft_loader.py 与 audit_project_sources.py 并实际运行：CPU 12.81 秒及 0.27 秒，均退出 0，未加载模型或训练。实测元数据配对／initial_state 不影响张量，目标内理由生效；五视图、mask、padding 和官方 dry-run 通过。保存 P920-DATA-v1_20261104T114417Z 证据并更新根 README 主线入口。

发布恢复：2026-11-04 11:54 UTC 同一专用密钥正常 push 成功，ca49483→c008080 共五个保留提交已推送；HTTPS git ls-remote 核验完整 SHA 一致。此前最终实验两提交已发布。新数据审计尚未提交，依用户小时合并频率待 12:06:47 UTC 后处理。

## 2026-11-04 扩查交接与状态接口

回应用户“没有吗？”：扩查 920 和附件 384452 路径、相关文本与交接 manifest，确认原 payload 只有四个巴黎文件，旧专业包仍无可读副本，未宣称全局不存在。保存 SOURCE_TRANSFER_CHECK.json。新增 decision_state.py 与 audit_state_interface.py，冻结 P920-STATE-v1_20261104T120532Z 后实际运行 7.39 秒、退出 0。四回合真实 memory/实例/位姿匹配，保留五次删除四次失败，单批次导出与 mask／未来元数据隔离通过；三类篡改被拒绝。新增状态接口说明及运行记录，无新模型、引擎执行或训练。Goal／DATA／STATE 成果合并按小时发布，精确 SHA 和网络结果保存 ../GIT_PUBLICATION_STATUS.json。

## 2026-11-04 官方三维参考来源恢复

上一 goal 轮为实质进展：1bfab22 推送成功且远端核验一致。本轮锁定 4 个官方 critic 源文件并核 Git blob；发现严格排序的并列限制与按目录划分机制。读公开数据卡确认 OTC Bench 为评测集，没有下载答案／图片。追查 Function2Scene 官方 viewer，发现内嵌 JSON；在 4 MB 上限内保存 2.8 MB HTML（SHA 4108acb3…），未执行网页脚本。P920-SOURCE-v1_20261104T122131Z 正式恢复 0.67 秒、退出 0：33 目录、336 快照、30 个终态／572 家具，其中带需求者 29／545 与历史计数吻合。记录共享原文、dancer 命名别名及巴黎／育儿房来源对应；柜子仅有生成名称、无真实实例 ID，仍不具备专业训练资格。恢复全量只留服务器，小型来源清单／回执和说明进入工作树，按小时合并尚未提交。未调用模型或训练。

## 2026-11-04 评审入口验收与项目身份核实

新增 review_samples.py、training_manifest.py 和 audit_review_inputs.py；冻结 P920-REVIEW-v1_20261104T123915Z 后实际 CPU 12:39:29—12:39:38 UTC、9.30 秒、退出 0。18 项工程检查通过；当前工程样本被拒绝，没有训练目录。保留候选换序、并列／未知、缺图行为、资格拒绝及首日志，归档小型结果与精确输入登记副本。原始 SOURCE／DD／TC／DATA／STATE 结果不改。

继续核对三篇原始设计报道，SOURCE_RELATIONS 区分项目身份、文章关联及暂定文件别名；记录适配需求中尚无原文支持的细节。登记扩展至 5 个开发家族、2 个有来源候选；合格专业决定仍 0。检查实际 dispatcher 发现新房间导入被硬编码巴黎路径限制，记录接口缺口，未启动新的模型或回放。以上为实质进展，本轮仍受最早 13:10:56 UTC 小时提交限制。

## 2026-11-04 原生房间接入与整屋评价准备

通过 GitHub 连接读取并保存 13 个固定官方执行／安装小文件，逐个核 Git blob／大小／SHA256，未导入执行或启动服务。核实 HTTP 新会话仅接文字，阶段交接依赖 HouseLayout、RoomGeometry 和 RoomScene；原生场景序列化不含完整会话；摘要有意省略墙高与窗精确位置。完成 project_input_v1 接入方案和源锁。

新增 inspect_runtime_environment.py，分别在 c15_genrec 和 geometry311 实际读取依赖元数据，满足 8/15、3/15 项，后者有 Drake；独立 Blender 仍可用于旧管线，不能把 module spec 检查解释为整机没有该软件。既有环境未改。新增 evaluation_v1 整屋评价草案与冻结清单，明确活动切换、实例保留、unknown、项目聚类、全部失败、盲评和公平训练预算；正式评测未开始。下一步准备完整隔离源码及资源方案，再实施原生导入。
