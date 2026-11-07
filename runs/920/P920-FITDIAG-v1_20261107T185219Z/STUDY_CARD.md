# 最终适配器拟合与自由输出诊断（待批准，仅评测）

目的：检查已经完成的 STOPDATA 最终权重，在不提供标准理由时是否学会训练中的条件化编辑与停止。已有 12 份开发终态保持原样；本计划不产生新的整屋比较，也不新增专业示范。

- 固定底座 revision `63e235a0c0d6e5bc2f8f3485214df4457fd2e672`，复用已验收的两个 checkpoint-8 adapter；0 训练更新，0 场景编辑。
- GPU0：最多 96 次教师强制前向（72 次共同动作目标＋24 次显式标准理由条件），18 次贪心自由生成（6 个已训练状态×3 组），每次最多 768 新 token。
- 六状态：Paper 的两种桌面组织、Waterloo 的两种活动编辑、Waterloo 的两个保持现状/停止目标。原训练输入和五视图不改；自由生成只取助手输出之前的前缀。三组可自行输出理由，不给标准理由。
- 总 GPU 子进程墙钟上限 1800 秒：外部 supervisor 在 1795 秒发终止信号，最多 5 秒清理；模型加载、哈希校验及所有前向包含在该子进程限额内。首基础设施错误即停，保留部分结果，不自动重试或补预算。
- 固定 FP32 LoRA、BF16 底座；模型全参数冻结且无反向。加载后恢复并核对磁盘适配器精确值，结束再核不变。

## 判断和边界

共同输入：在每个项目族内按 token 汇总动作 NLL，再对六族等权；两项停止目标单列。标准理由条件：按相同输入比较三组，明确这是知道答案理由的教师强制诊断，不能用它替代部署表现。自由输出保存全部原文、token、目标结构及一致性；格式与目标动作匹配只证明复现，不能证明执行可行或整屋功能。

目标不匹配也可能是其他可行动作，不自动标成设计失败。本计划不执行新原生场景；若需判断替代动作的后果，先审读结果再提出有限范围。只做训练内诊断，不声称独立项目泛化、专业理由必要性或配对机制成立。分量损失下降少也不能单独证明优化失败。

如果只有标准理由条件有利：优先检查自行形成决定的能力及部署一致性。若训练内共同目标和自由输出也无改善：不扩大同配方，先定位监督与优化。若训练内有改善但开发整屋无收益：下一问题才是覆盖/迁移。以上均不是自动训练授权。

## 已完成验收与启动

[PLAN.json](PLAN.json) SHA256：`b06d168cfba27d68b0163cd7264a655782d215760dd597fd05877546f58b982d`。模型/适配器/编码/运行代码哈希已冻结。48 个真实张量与 24 个原目标解码通过；自由生成前缀无标准理由/目标泄漏。损坏标签、提示、图像和无批准入口均拒绝；实际 CPU 超时进程返回 124。12 组 CPU 检查通过，GPU 尚未运行。

在收到对此计划的明确批准后，才创建 `AUTHORIZATION.json`，内容为 `approved: true`、上述 `plan_sha256` 和真实 `user_message`；当前不存在授权文件。唯一正式启动命令：

```bash
cd /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc
python3 scripts/920/launch_final_edit_fit.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-FITDIAG-v1_20261107T185219Z/PLAN.json --authorization /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-FITDIAG-v1_20261107T185219Z/AUTHORIZATION.json
```

运行目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-FITDIAG-v1_20261107T185219Z`。启动时记录真实时间和首日志，结束/超时写入实际进程退出回执。`EXECUTION.json` 与 `STATUS.json` 以外部进程回执为准；若超时杀死 worker，部分 `RESULT.json` 的旧 running 字段不代表仍在运行。

需要新批准的原因：上次获批的是固定的 71 次新增调用续行，已完整结束；这是新增的有限 GPU 诊断范围。[AGENTS.md](../../../AGENTS.md) 要求 “New experiments require an authorized scope, distinct run IDs and prospectively frozen, finite budgets.” 此处不是新训练许可申请。
