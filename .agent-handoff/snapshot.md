# 当前研究状态

- 更新2026-11-05T10:31:59.696056+00:00；总体Goal active。上一Goal轮是progress：获批真实反向OOM证据与数据/任务已推94dc3b585a8e47bc2b1a7dde4bd5f86ffd234998。当前也有实质代码与CPU检查进展。当前已启动获批OFFLOAD批次：session10407、outer pid1265279，真实启动10:35:23.143749UTC；先查该句柄/回执，不重启。
- 当前run P920-OFFLOAD-v1_20261105T102528Z已获新明确批准并开始真实训练，首反向通过、整批仍运行；新PLAN.json SHAb76074bad354fb243a095d98d9312738b2b53e8cd2d328b701c226419cf59835。仅语言36层检查点输入save_on_cpu(pin_memory=False)，不改原底座精度/五图/输入/目标/源权重/LoRA/预算。新代码独立decision_pilot_offload.py、run_decision_offload_batch.py、checkpoint_input_offload.py；原失败代码/计划均未改。
- 新计划仍两组各2更新，共4；GPU0，语言q/v 3833856参数、BF16/SDPA/AdaFactor，20微例/982目标/每更新、3来源等权，1800s每组/3900s外层，首错停止。CPU临时副本最低可用主存16GiB；约2.7–3.3GiB边界尺寸只是估算，未实测节省。
- CPU两随机tiny Qwen3-VL/五合成图/32与48token，共4反向0优化，输出/loss/全部16个LoRA梯度差0；5.334462s/0。float32 CPU不证明完整真实GPU/BF16/时间；装4语言层wrapper，每例4调用，所有参数未改，无CUDA。
- 旧真批准与新false模板实际CLI门禁均拒绝（4.581191s/1、4.480485s/1），训练目录不存在。AST与输入/新代码哈希核验通过；目标函数/数据/采样完全不变。新代码在每微例前后写进度和OOM显存，弥补旧失败不存micro_log缺口。
- 用户已明确回复“批准此修订方案的4次更新”，真批准USER_APPROVAL.json绑定新计划SHA，已运行PROPOSED_COMMAND对应批次；首错停止，无自动重试。普通准备留本地待关键节点/小时批，上一推送10:20:59UTC。
- 原PILOTPLAN批次112.355571s/1，分开组71.461052s/1，反向检查点MLP OOM；0/4更新、配对未启、无checkpoint/adapter/重试。真实结束后GPU0 4MiB/24121MiB空闲。旧批准已消费，旧结果不可改写。
- 数据：14知识/13知识家族；22登记=12dev+8train_source+2test_candidate，封存/正式评价ready0。5执行来源决定/8版本、9回放34动作、6QA、3条件几何收益+1模式转换，Mini165；真人专业验收/核心训练准入0。
- 独立方向：EVALTASK两草案已先划测试候选且不入训练/检索，Casa主文核对，Moabit仅作者搜索片段；人数/矩形/开口是研究设定，不是真实brief或测绘。下一应落实一个实际初态与完整用途表，不再空增资料。DLUGI两知识不可由平移床/缩放物体冒充复现。
- 不碰Swiss及未跟踪extract_swiss_development_input.py，不用其他GPU、不对外联系/付费。正式研究、独立整屋生成与论文证据仍未完成，Goal保持active。

- 2026-11-05T10:38:21.983478+00:00 首真实反向已核，FIRST_BACKWARD含第一微例loss与进度快照；完整4更新/保存未验收，勿仅因当前成功而终止观察。先检查session10407或authorized_training/BATCH。准备首输出节点提交。
