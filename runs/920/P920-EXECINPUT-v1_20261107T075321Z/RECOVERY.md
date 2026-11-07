# 恢复与复核

代码及文本位于 runs/920 的本批目录；最终代码快照在 code/。实际命令、起止时间、首日志与退出码见 COMMANDS.md 和 receipts/。最终数据是 corpus_phase/，不是只迁移历史的 corpus/。

服务器最终张量：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-EXECINPUT-v1_20261107T075321Z/phase_encoding/tensors/`。ENCODING.json 记录每个 .pt 的 SHA256；不要提交张量、权重或完整检查点。固定模型 revision 为 63e235a0c0d6e5bc2f8f3485214df4457fd2e672，路径/配置哈希在 PLAN_v5.json。源场景与五图的逐项路径及哈希在 DATASET.json，缺少它们时不能仅凭 Git 文本重建图像或执行证据。

优先读取并核对已有张量哈希，无需重算。需要异地恢复时，先恢复 PLAN_v5 指定的来源、原始张量、模型 processor 和原生场景资源，再逐项验证哈希。当前环境使用原生 CP311 做 materialize/contract checks、模型 CP310 做 encode，不能把两套 site-packages 混用。

旧 PLAN_v5 绑定的独立 checker 已变更；原版本重建快照的 SHA 已确认匹配，见 CODE_PROVENANCE.json。RECOVERY_PLAN.json 记录最终当前依赖，供必要时在新输出目录按 COMMANDS.md 的 --mode materialize/encode 恢复，未被冒称为已执行计划。恢复时替换输出路径会改变文档路径和文件哈希，须记录新 manifest；不得覆盖本批输出。编码消费官方加载器得到的监督目标，再使用同一请求前缀；本地官方加载器是记录哈希的适配版本，不声称未修改上游。

本批没有新增 adapter、优化器状态或房间检查点。原历史训练/房间实验关闭，禁止把恢复数据解释成重跑授权。早期 materializer 的全部源码快照未保存，只保留各版计划哈希/命令/日志；不声称每个早期失败都可逐字节复演。
