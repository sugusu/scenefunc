# 当前执行：有限的完整房间编辑学习试验

更新 2026-11-06T08:17:11.462355+00:00；Goal active。本轮已实际开训，禁止误读旧“0新训练授权”再要求确认或重启。

- 用户给出六家族/两adapter各8更新各2GPUh/12房间共2GPUh方案后，两次明确“继续进行”，随后催问；AUTHORIZATION.json记录此新授权和代理对实现细节的解释。广义Goal仍不授无限训练权，旧OFFLOAD4更新已消费完。
- run `P920-EDITSTUDY-v1_20261106T074752Z`。44官方OTA编码/22动作每组/6家族/7决定位置通过；动作目标1245token，联合额外1094；实际Trainer CPU缩放误差<4e-8。研究者重建与来源原则分开，0专业整屋金标准。
- 训练真实batch开始08:00:26UTC，action_only模型阶段08:01:06UTC；本快照2/8更新完成，0错误。GPU0，两锁；进程会话79563；下一decision_action由同批依次运行，勿另启动。每组7200秒/8次，首错停批。每次checkpoint和微批进度都在 runtime/P920-EDITSTUDY-v1_20261106T074752Z/training/。
- 评测队列会话27562，08:14:07UTC实际启动等待现有训练，绝不启动优化。两组save/reload验收后12cell顺序执行；每cell模型子进程600秒，总7200秒。持久状态EVALUATION_BATCH.json。若训练失败队列停。
- Pine/Milan初态均CPU恢复/五图/真实编码通过，9320/15159token；12模板共同条件核对通过。Pine初态缺第二工作位，既有参考三宽均通过；Milan共桌材料/餐位初态已三宽通过，不能算学习增益，是保持/副作用任务。两案均开发且从训练/知识排除。
- 官方think/tool_call，真实采样17/29（temperature.7/top_p.9/top_k50），同6来源知识、规划、当前实测反馈；无标准D/参考姿态。每cell12模型与工具尝试、每次1024输出。新指标允许共桌，finitepatch/座方/通路只是代理；未知舒适/客睡/设施不能成功。
- 最新可查：本run README、PREFLIGHT_RESULT、TRAINING_RESULT、EVALUATION_BATCH、STATUS、recorded微批快照。代码train_edit_learning_study/run_edit_training_batch/run_edit_evaluation_batch/run_edit_policy/edit_study_model_step/evaluate_edit_room/edit_workspace_metrics。既有冻结代码不改。
- Git本地父commit2befbd1b9090c626ebefb143df2221890d6255ea尚未推送；最近ls-remote仍9f75c1891fd795a4190e4174c590cfd59518dc64。保留同链，下一提交本轮实际代码/日志再push；不force/main/AutoDL。无关未跟踪extract_swiss_development_input.py绝不stage。

下一步：监控现有作业、核每组8更新与保存、12终态对照及GLB；根据初态/终态变化判断收益，不新增测试题。核公开commit与远端；队列全部结束后汇总结果表，不能把正在运行写成成功。
