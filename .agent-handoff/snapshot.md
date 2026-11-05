# 当前研究状态

- 更新2026-11-05T12:45:44.039373+00:00。本轮progress：Casa实际五视图52.857343s/0、共同输入准备56.303227s/0（8981token/5图/0训练）、初态路径1.670568s/0，三体宽对两目标均可达。
- **活跃作业：P920-CASA-COMP-v1_20261105T123924Z，exec session68372，外层pid1275191 start_ticks54035440，启动12:41:45.816210UTC。** runtime/P920-CASA-COMP-v1_20261105T123924Z/batch_execution/receipt.json、BATCH.json和各arm/rollout/RESULT.json为权威；先poll同一handle或核pid，不因观察超时重启。外层16800秒，每组5400秒，顺序base/separate/paired，各最多24模型/动作，0训练，首infra错误停整批。
- 三组固定同一prepared request SHA e5661fa4a7bb7cee6fa8156273cf2a02ed3489cac52ef052b54c45cc82842022，同BF16/SDPA贪心、16目录和工具/视图/需求。原底座或OFFLOAD两步adapter（不可继续训练）。共同准备t/prepare，模型Python c15_genrec，原生driver native311，审计geometry311；GPU0两锁不变。
- 首base输出真实加queen_bed_0，16.268415s/0，回执/状态绑定/图已归档first_output，作业还在继续，不能报整批完成。引擎位姿扰动由请求变换确定性seed，std位置.001m/yaw.1度；评实际返回姿态，未改算法。准备发首真实输出/首带家具节点push。
- Casa观察原RESULT furniture_instances=13误数所有objects，COUNT_CORRECTION保留原证据并按type更正9wall+4wall_mounted开口、0furniture；不改冻结脚本/场景、不重渲染。初态五图可辨认，露台外缘部分裁切，隐私/安装/容量仍未知。
- 接下来监看同一作业并完成九用途逐项评价；通行/碰撞代理不是专业或人体结论。只有一个未封存研究壳/2更新adapter，没有配对收益或泛化证据；强提示/同知识检索和动作-only对照仍待完成。

- 更新2026-11-05T12:28:42.222737+00:00；总体Goal active。上一轮修订准备为progress，本轮真实四更新与独立初态构建也是progress。旧训练/CPU sessions10407/14404/20254/80513均已终态；新活跃整屋batch见首段。
- OFFLOAD P920-OFFLOAD-v1_20261105T102528Z已获用户第二次明确批准并完成：PLAN SHA b76074bad354fb243a095d98d9312738b2b53e8cd2d328b701c226419cf59835不变；4/4优化更新、80/80反向，2组各40micro/2step。10:35:23.143749–10:56:51.940223UTC，1288.796496s/0。
- 两组进程621.210053/626.677043s/0，Trainer574.3855/579.926s，峰值19.376747/19.385646GiB；各144adapter张量变化，checkpoint1/2+final均在runtime。真实同进程保存/清零/重载精确，另CPU磁盘验收2.022283s/0：final等checkpoint2、optimizer全144状态有限且step2、两组参数不同。冷启动优化器恢复未验。
- TRAINING_RESULT、TRAINING_METRICS.csv、CHECKPOINT_MANIFEST、training/、execution/与RESTORE说明为当前证据，原RESULT是准备阶段。指标为更新前训练周期：separate2.500305759→2.503363070，paired2.428457503→2.431532676；两组略升，初始差已存在。没有最终适配器生成/评审/专业学习或泛化证据，不宣称学会设计。
- 本次四更新授权已消费，不自动追加更新/重训。模型及adapter/checkpoint留服务器；新推理评估可在既有授权内另定有限预算和共同输入，训练必须另审。下一优先独立初态视图/用途评价及base/adapters共同推理条件，不反复调这3来源10相关步。
- 原PILOTPLAN OOM0更新及其批准/日志保留，未覆盖。OFFLOAD全部原输入/监督/LoRA/优化预算保持；仅36语言检查点输入CPU暂存加记录。GPU0终验4MiB/24121MiB，无compute进程。
- 首真实反向已推f1004c7e4395b64ecf2a5f6b3d61e44500615f7c并ls-remote精确核对（44文件300513B），回执../preflight/publication_f1004c7e4395b64ecf2a5f6b3d61e44500615f7c/PUBLICATION.json。本轮最终验收已推b3e1367b0a03cb48d9537ef10a456c091ce79950并ls-remote精确核对；72文件512843B逐字节核验，回执../preflight/publication_b3e1367b0a03cb48d9537ef10a456c091ce79950/PUBLICATION.json。发布后本地回执文字并入下次实质批次。
- 独立初态 P920-CASA-INPUT-v1_20261105T104553Z：Casa测试候选6×5×2.7m研究房壳、东南1.8×2m卫浴围护、4开口/露台背景、16同目录真实网格可选，初始家具0。1次来源枚举失败后新V2构建恢复1.921717s/0；checker公共面误判修正后实际SDF边界/开口探针通过0.165338s/0。失败和原代码均保留，无渲染/模型/训练。72.6MB包留runtime，原生小状态/manifest/SDF已归档。
- Casa所有尺寸/人数/固定卫浴是研究假设，不是原始brief或测绘。门/隐私/安装/容量等未知保留，正式评价ready0。下一五视图和完整要求可行性检查后再定新生成预算。不要用原生加载成功代替整屋成功；不要从待测模型输出反调任务。
- 14知识/13知识家族；22登记=12dev+8train_source+2test_candidate；1原生候选初态、0sealed/0人评。已有5执行来源决定/8版本、9回放34动作、6QA、Mini165；本次实际适配3来源10重建步骤单列，不抹掉数据层也不等同专业偏好验收。
- 不碰Swiss及未跟踪extract_swiss_development_input.py，不用其他GPU、不擅自外联/付费。总体论文目标未完成，不标complete。
