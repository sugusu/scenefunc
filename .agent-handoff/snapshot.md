# 当前：完整学习试验已发布；接口诊断停止，候选前缀修正通过CPU验证

更新 2026-11-06T15:15:22.243666+00:00。研究Goal active，无活跃训练/推理/CPU作业。不重启已结束作业。

- 主试验 P920-EDITSTUDY-v1_20261106T074752Z：两adapter各8更新、12房间已结束，144调用133拒绝；未见专业决定监督收益。FINAL_REVIEW/FINAL_ACCEPTANCE为完整结论。
- 最终提交56f76fff0ac4218e91f3b11a8e4be471cb7ca335已推送并ls-remote精确核验，15:03:04UTC。首次600秒超时、重推同一提交219.7秒成功，未重跑试验。PUBLICATION记录证据。
- 只读接口审计：动作22/22、联合15/22在未监督空think后学习A；线上从助手起点生成。144调用中131query、119相邻重复、121有上次拒绝、0助手历史。没有enable_thinking模板开关，未确立因果。
- P920-SEENPROMPT-v1_20261106T150500Z原计划6调用，实际4调用90.019秒、0训练/0场景编辑；AIR三组都有think、都用非法来源资产ID。第四个Waterloo底座有think，rotation代替yaw。校验器假定完整schema/字典对象而退出1，后两adapter未调用；按规则不补跑。
- 新run的RECOVERED_ACCOUNTING.json才是实际4调用计数，原SUMMARY仅计3个完成校验的行；FAILED_RUN_STATUS/原回执保留。首次启动缺jsonschema发生在0模型调用前，原生启动器修正已记录。
- 22示例12完整schema+dict对象、10文字签名+list对象。两个抽查prompt的Memory为空历史但Task含已有对象；不能称房间为空或已证失败原因。
- 新候选scripts/920/edit_supervision.py用官方no_thinking=True去掉无D样本的未监督前缀；check_edit_supervision.py在全部44真实样本通过，37改变、原prompt/action/D/五图完全一致。0新训练，历史encoder/tensors不改，尚无候选推理收益。
- 新run REVIEW.md含结果、限制及复核命令。当前变更准备合并同一runs/920链；以实际Git和PUBLICATION核最终发布状态。
- 勿提交无关scripts/920/extract_swiss_development_input.py；权重/完整checkpoint/GLB仍只在服务器。

下一步围绕具体专业决定统一已有22示例和部署的对象表、工具schema、可调用/来源资产身份、已存家具与空历史表达，用同一生成器并CPU校验所有格式。不要因这次校验器错误再补跑6调用，也不要扩主试验或自动追加训练。训练必须单独审定，候选前缀修复只是工程前提，不是论文贡献。没有待用户提供资料的阻断。
