# 公开三维参考恢复记录

从固定 SHA256 的官方 viewer HTML 解析内嵌 JSON，33 个目录、336 个快照；29 条同时包含需求与最终布局，合计 545 件家具。CPU 0.67 秒、退出 0，无模型、训练、渲染或场景执行。完整来源与限制见 [研究说明](../../../research/920/source_recovery_v1/README.md)。

`SOURCE_FETCH_RECEIPT.json` 记录实际下载时间、URL、大小、服务器标头与 SHA256。`PROTOCOL.json` 在正式恢复前冻结；计数已在来源发现阶段观察到，故这不是事前未知结果的统计实验。`recovery/REFERENCE_INVENTORY.json` 列每个源快照的规范化 JSON 哈希，`recovery/RESULT.json` 与 `execution/recovery/` 保存真实进程记录。`PARSER_CHECKS.json` 记录截断、重复赋值、非字面量表达式和错误数据形状四项拒绝检查。

在新的输出目录恢复；网页若变化须另记新来源版本，不能覆盖已有证据：

```bash
python3 scripts/920/run_stage.py --receipt-dir /tmp/f2s_recovery_receipt_new \
  --timeout 120 -- python3 scripts/920/recover_reference_inventory.py \
  --html /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/research_support/public_source_inventory_20261104/function2scene_viewer.html \
  --receipt /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/research_support/public_source_inventory_20261104/FUNCTION2SCENE_HTML_RECEIPT.json \
  --out /tmp/f2s_reference_recovery_new
```

原始 HTML 来源为 `https://function2scene.github.io/viewer.html`，预期 SHA256 `4108acb36cef2ecbaaa39c385ef8da6a6fd64988b852c2a3b96c58a1523f23d0`。完整解析数据只在服务器 `runtime/P920-SOURCE-v1_20261104T122131Z/recovery/embedded_data.json`，未提交 Git。公开访问不等于专业示范、原始实测空间或已获训练许可。
