# 保存状态与真实回执核验

CPU 审计已完成，实际退出 0，耗时 7.39 秒。四个原始动作批次的身份、位姿、状态差异与真实回执一致；实际 SFT 加载器的输入／目标边界通过。无模型调用、参数更新、渲染或新场景执行。方法及限制见 [状态接口说明](../../../research/920/STATE_INTERFACE.md)。

- `PROTOCOL.json`：执行前冻结的问题、预算、代码和输入哈希。
- `STATUS.json`、`execution/state/receipt.json`：真实开始、首日志、退出及当前下一步。
- `audit/RESULT.json`：四回合结果、源文件哈希和负向拒绝检查。
- `audit/episodes/`：当前状态和真实动作的工程导出，禁止作为专业训练集。
- `audit/consequences/`：实际后果与回执，和当前决定的输入隔离。
- `audit/evaluation_metadata_variants/`：评分元数据不影响张量的检查输入。

在新的目录重算，不能覆盖本次证据：

```bash
python3 scripts/920/run_stage.py --receipt-dir /tmp/scenefunc_state_recheck_receipt_new \
  --timeout 300 -- /data/wyh/conda/envs/c15_genrec/bin/python \
  scripts/920/audit_state_interface.py \
  --root /data/wyh/scene_cvpr_20261020/design_decisions_920 \
  --source /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/research_support/Scene-Designer/a0565e50ed7d32c0e09132b9311c0b9b91165405 \
  --lock research/920/data_readiness_v1/OFFICIAL_SOURCE_LOCK.json \
  --out /tmp/scenefunc_state_recheck_new
```

依赖原始 P920-DD-v1 服务器 checkpoint、对应几何、五视图、请求与回执，以及固定 processor 小文件。第三方源码按来源锁恢复，完整 checkpoint／网格／权重继续留在服务器；Git 仅保存小型记录。这里的 memory 重建不等于引擎重放，工程通过不等于专业数据资格或设计质量通过。
