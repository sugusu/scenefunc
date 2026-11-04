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
