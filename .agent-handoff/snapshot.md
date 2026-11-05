# 当前研究状态

- 更新2026-11-05T10:20:04.637951+00:00；总体Goal active，论文目标未完成。无活跃作业；session14349已exit1。
- 用户明确批准PILOTPLAN精确PLAN_V2四更新试运行，记录USER_APPROVAL.json；已实际执行并按首错停止，不再待本次答复。批准不覆盖换配置、自动重试或后续训练。
- 当前run P920-PILOTPLAN-v1_20261105T094043Z；planSHA6443a3bf19b2a7ec07df0ee4b76f48fb8e30f262da8fa459bd19869ef9d1f2c2不变。实际批次112.355571s/1，分开组71.461052s/1；反向检查点重算的Qwen MLP分配272MiB失败，0/4优化更新，配对组未启动。
- 首日志/真实开始退出/错误/批准均已归档；TRAINING_RESULT.json是当前结果，之前RESULT与CPU检查是准备证据。没有训练checkpoint/adapter、未验证保存重载；逐微例日志在失败时未保存，完成微例数未知。BATCH completed_arms=1表示一次尝试结束，成功组0。GPU0已恢复4MiB占用/24121MiB空闲。
- 不修改本次冻结代码、输入、模型或计划，不重跑。下一步只读诊断反向激活显存与保持同知识/动作/监督token的可执行方案；另立具体可审核计划后再申请训练。独立数据与整屋任务工作继续，不能把失败推为设计或学习结论。
- 待发布阻断错误节点将一并带上普通数据增量：DLUGI两知识（外屋顶保温/抬床窗景抽屉），0新动作；EVALTASK两家族先划测试候选再读源，2草案导出0.064759s/0，无场景/模型。Casa读主文、Moabit仅作者作品搜索片段/fullpage403；明确研究人数、矩形与门窗，缺原始brief/测绘/评价阈值。
- 数据：14知识/13知识家族，22登记家族=12dev+8train_source+2test_candidate，0sealed/0正式评价ready。5执行来源决定/8版本、9回放34动作、6客观QA、3条件几何收益+1模式转换，Mini165；真人专业验收/正式核心训练准入仍0。测试候选禁止训练与其他项目检索，不加入已执行pilot。
- 最新已推ed9eafac5f7264c4fbadd0855212235d2fc2e456（训练前验收），本次失败按关键节点立即提交推送并ls-remote核对。历史LOSS30前向动作NLL.283854/.437199仅条件提示效应；MODEBASE0/4只作开发失败。无需重跑这些检查。
- 当前证据目录：runs/920/P920-PILOTPLAN-v1_20261105T094043Z、P920-DLUGI-KNOW-v1_20261105T095744Z、P920-EVALTASK-v1_20261105T100503Z；批准记录不是模板。大模型与runtime仅服务器，Git存小日志/哈希/恢复说明。
- 不碰Swiss及未跟踪scripts/920/extract_swiss_development_input.py，不使用其他GPU、不外发联系或付费。Goal持续active，不缩成机制试验。
