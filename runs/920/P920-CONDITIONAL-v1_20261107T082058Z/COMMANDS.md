# 实际执行命令

工作目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc`。以下CPU命令已经执行；输出目录采用只写新目录策略。再次复现须选择新的输出和回执目录，不能覆盖原证据。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/check_executed_live_loop.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-CONDITIONAL-v1_20261107T082058Z/PLAN.json --out /data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CONDITIONAL-v1_20261107T082058Z/loop
```
实际开始：2026-11-07T08:21:11.524289+00:00；退出：0；耗时：0.7683775459881872 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/build_conditional_calibration.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-CONDITIONAL-v1_20261107T082058Z/PLAN.json --out /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-CONDITIONAL-v1_20261107T082058Z/calibration
```
实际开始：2026-11-07T08:21:11.496220+00:00；退出：0；耗时：0.2164907429832965 秒。

未执行任何模型或训练命令。夹具内部的模拟子进程回执不是实际运行回执；实际证据在 execution/。

新原生入口仅供将来冻结方案使用：`run_edit_policy_executed.py --plan APPROVED_NEW_PLAN.json --phase prepare|rollout --out NEW_DIRECTORY`。本批不提供可直接启动的原生计划。
