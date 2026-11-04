# 评审输入与资格清单 CPU 核验

实际运行时间 2026-11-04 12:39:29—12:39:38 UTC，9.30 秒、退出 0，18 项检查通过。`PROTOCOL.json` 在运行前冻结代码及输入哈希；`execution/review/receipt.json` 保存真实命令、首日志和退出状态。完整原始过程保存在服务器同名 runtime 目录。

明确偏好及换序使用官方 critic；并列、证据不足使用官方 QA。四种输入均经实际 `StepwiseSFTDataset` 处理并核对输入／目标 mask。工程标签只对应指定实例的保留数；没有设计师评审、模型输出或整屋质量结论。QA 缺图静默跳过的问题在固定官方实现中被复现；新增导出层会拒绝缺图和候选错图。详见 [接口与限制](../../../research/920/critic_contract_v1/README.md)。

`audit/audit_selected_loader_inputs/` 仅用于本次工程审计，不是训练目录。`audit/qualification/registry_with_deliberate_leak.json` 是故意构造跨划分的拒绝用例，不能当成真实项目登记。`audit/fixtures/strict/unlisted_neighbor.json` 是应被名单排除的相邻文件。所有工程样本均不合格，没有生成专业训练输入目录。

输入登记的精确副本已保存为 `inputs/PROJECT_REGISTRY.json`，哈希与原执行命令读取的登记一致。后续登记会更新，复核应使用此副本。选择新的输出和回执目录，不能覆盖本次结果；在仓库根目录执行：

```bash
python3 scripts/920/run_stage.py --receipt-dir /tmp/p920_review_receipt_new \
  --timeout 300 -- /data/wyh/conda/envs/c15_genrec/bin/python \
  scripts/920/audit_review_inputs.py \
  --root /data/wyh/scene_cvpr_20261020/design_decisions_920 \
  --source /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/research_support/Scene-Designer/a0565e50ed7d32c0e09132b9311c0b9b91165405 \
  --lock research/920/data_readiness_v1/OFFICIAL_SOURCE_LOCK.json \
  --critic-lock research/920/critic_contract_v1/OFFICIAL_CRITIC_SOURCE_LOCK.json \
  --registry runs/920/P920-REVIEW-v1_20261104T123915Z/inputs/PROJECT_REGISTRY.json \
  --out /data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/p920_review_reproduction_new
```

原场景、网格、渲染和处理器文件沿用服务器缓存，来源及恢复方式见原 P920-DD-v1 与 DATA 记录。Git 只保存小型 JSON、日志、脚本和哈希。脚本不加载模型权重，不进行参数更新；此命令不构成训练授权。
