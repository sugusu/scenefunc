# 当前执行：有限的完整房间编辑学习试验

更新 2026-11-06T09:14:51.280794+00:00；Goal active。action8/8已验，joint5/8在原作业中继续，0评测输出。

- 用户给出六家族/两adapter各8更新各2GPUh/12房间共2GPUh方案后，两次明确“继续进行”，随后催问；AUTHORIZATION.json记录此新授权和代理对实现细节的解释。广义Goal仍不授无限训练权，旧OFFLOAD4更新已消费完。
- run `P920-EDITSTUDY-v1_20261106T074752Z`。44官方OTA编码/22动作每组/6家族/7决定位置通过；动作目标1245token，联合额外1094；实际Trainer CPU缩放误差<4e-8。研究者重建与来源原则分开，0专业整屋金标准。
- 训练session79563持续：action_only已8/8更新、176backward、6336层checkpoint、144张量改变，08:00:26.268619→08:45:07.741908UTC，2681.473314秒/exit0，峰值20.36036GiB。同进程save/zero/reload exact，CPU核FP32全部finite且final==checkpoint8。联合组08:45:07.802438UTC自动开始，目前6/8更新，最近微批152完成；两个GPU锁仍由原作业持有。不能额外启动任何训练。
- 评测队列会话27562，08:14:07UTC实际启动等待现有训练，绝不启动优化。两组save/reload验收后12cell顺序执行；每cell模型子进程600秒，总7200秒。持久状态EVALUATION_BATCH.json。若训练失败队列停。
- Pine/Milan初态均CPU恢复/五图/真实编码通过，9320/15159token；12模板共同条件核对通过。Pine初态缺第二工作位，既有参考三宽均通过；Milan共桌材料/餐位初态已三宽通过，不能算学习增益，是保持/副作用任务。两案均开发且从训练/知识排除。
- 官方think/tool_call，真实采样17/29（temperature.7/top_p.9/top_k50），同6来源知识、规划、当前实测反馈；无标准D/参考姿态。每cell12模型与工具尝试、每次1024输出。新指标允许共桌，finitepatch/座方/通路只是代理；未知舒适/客睡/设施不能成功。
- 最新可查：本run README、PREFLIGHT_RESULT、TRAINING_RESULT、EVALUATION_BATCH、STATUS、recorded微批快照。代码train_edit_learning_study/run_edit_training_batch/run_edit_evaluation_batch/run_edit_policy/edit_study_model_step/evaluate_edit_room/edit_workspace_metrics。既有冻结代码不改。
- Git当前已核远端 `8c9d0960d0f5daf6625d1e56c58392310286c070`（runs/920），包含父commit2befbd1及本轮119文件。push17.023110秒/0、ls-remote精确相同；PUBLICATION.json记录。下一普通进度提交合并到>=09:19:32UTC或下一关键验收/错误节点，无变化不提交。无关extract_swiss_development_input.py未stage。

本轮新增：summarize_edit_study.py只读初末汇总与check_edit_study_report.py八项核对（REPORT_VALIDATION.json）通过；实际参考暴露old_seat_side失效但右/后可达，严格探针损失与几何/身份违例/人类用途分别报告。未改任何冻结训练、评测、反馈代码或数据。METHODS_DRAFT重写为当前三组方法，旧106行保留METHODS_HISTORY_20261106.md。这些文件和发布补记留下一实质小时/验收批，不抢占每小时提交限制。

下一步：监控现有作业、核每组8更新与保存、12终态对照及GLB；根据初态/终态变化判断收益，不新增测试题。核公开commit与远端；队列全部结束后汇总结果表，不能把正在运行写成成功。

本轮新增ACTION_TRAINING_RESULT.json记录权重/最终checkpoint8/优化器文件服务器路径与哈希（weights不入Git）；四初态视图原样归档INITIAL_VISUAL_CHECK.json，非专业复核。CLAIM_EVIDENCE更新当前编码、初态、探针边界与第一adapter验收，19证据源。当前已达到09:19:32UTC普通合并窗口，准备推送第一组验收及后续修订（联合组6/8、0评测）。

重要评测修订V2（训练锁未改）：实际CPU fixture发现完整schema ValidationError使Milan下一输入18474>18432。run_edit_policy仅改为<=512字符模型回执，完整异常继续存RESULT/validation_error；实编码Pine12275→9361，Milan18474→15200。REJECTION_MESSAGE_PREFLIGHT和FEEDBACK_REVISION保存前后证据，旧代码code/run_edit_policy_before_feedback_fix.py/旧锁EVALUATION_EXECUTION_LOCK_V1留存。12未来PLAN_TEMPLATE更新codeSHA，EVALUATION_EXECUTION_LOCK revision2全部核对。原PREPARE_PLAN/请求不改，旧准备可在8c9d096恢复。队列27562在训练结束后会重新读取新锁，已确认仍waiting且cells为空，不重启队列。

新增VISUAL_CONTINUITY_PROTOCOL：Milan原任务含工作位—沙发几何视线，原初态报告两座候选均15/15无阻挡（朝向分别约.610/-.277）。当前prepared与原初态零状态差，复用已有证据，不重算。文件有6个终态CPU命令，待相应final_checkpoint形成后执行；只读取ray/seat字段，旧dedicated-work flag不用于新共桌验收。它不进入模型反馈，不改主指标或原任务。
