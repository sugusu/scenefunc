# 最近工作

## 2026-11-04 长期研究启动

- 已创建线程 active Goal，核对官方 Goals 文档和可用工具；独立 automation_update 不可用，未创建其他调度。
- 建立 AGENT_HANDOFF.md、AGENTS.md、AGENT_SESSION_PROMPTS.md 与 .agent-handoff/，填入真实研究状态和后续队列。
- 新增 research/920/RESEARCH_GOAL.md，定义论文目标及七项验收条件。
- 新增并运行 scripts/920/audit_tool_contract.py，只读取 v1 请求、调度器和真实回执，输出 research/920/tool_contract_audit_v1.json。确认 3/9 次调用、3/4 个回合受到未接入 check_physics 影响；完整提示有前后冲突。
- 未运行新的模型、渲染或训练，未修改历史实验产物。
- 正常重试 git push origin runs/920，仍返回 publickey 失败；保持本地历史并继续 CPU 工作。
