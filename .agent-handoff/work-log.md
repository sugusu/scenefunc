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
