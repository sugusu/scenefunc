保留现有分支与已完成实验；本批不重跑模型、原生动作或训练。复核从 SUMMARY.json → calibration/TASKS.json → evaluation_only/REFERENCE_SCORES.json → LOOP_CHECK.json 读取。

源码快照在code，实际两阶段回执在execution；CPU测试产生的服务替身状态仅在服务器 /data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-CONDITIONAL-v1_20261107T082058Z/loop，不是真实原生检查点。模型、网格与原五图保持原位置，input_hashes给出路径与SHA。四份任务引用原图不额外发布图片。

新原生任务必须另建冻结计划，指定真实资产checkpoint、评价脚本、共同知识和有限预算；不得拿 FIXTURE_PLAN_NOT_FOR_NATIVE_LAUNCH.json 启动GPU。当前不存在获批新训练。
