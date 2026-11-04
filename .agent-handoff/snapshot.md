# 当前研究状态

- 更新：2026-11-04；执行者：Codex。
- 目标：按 research/920/RESEARCH_GOAL.md 推进至可复现、主张有证据支撑的论文成果包。线程 Goal 已 active；设置完成不等于研究完成。
- P920-DD-v1 已验收，原始结果和两个最终提交保留。原始最终提交 6bed83aa33369e2b8fd190d233639eb344065bf4。
- 当前阶段 R1：新 run_id P920-TC-v1_20261104T104227Z 已完成 CPU 编码与协议冻结，四输入 6604/7187/7190/7685 tokens，尚未运行新模型。协议在 runs/920/P920-TC-v1_20261104T104227Z/。
- 下一步：comparison_model.py --run-id P920-TC-v1_20261104T104227Z --kind probe --label B_U 做守门预检；通过后 run_formal_models.py 用同 run-id 恰好执行四臂，再 process_revisions.py 与 score_comparison.py；不得覆盖或重新标记 v1。
- 并行待办：重试 Git 发布原提交；最新已核验远端 ca494838a38fb4542937a784cc42efc7ba32c702，最新 SSH push 仍 publickey 失败。该问题不阻断本机 R1 工作。
- 关键文件：research/920/RESEARCH_GOAL.md、research/920/tool_contract_audit_v1.json、scripts/920/audit_tool_contract.py、runs/920/P920-DD-v1_20261104T083214Z/README.md。
- 未解：相关工作创新边界、可信同项目专业配对数据、独立项目测试集、整屋用途验收、人类盲评。均未达到论文验收。
