# P920-DATA-v1：来源与实际 SFT 加载器审计

CPU 审计已结束；0 次模型生成、0 次训练更新，未编辑场景。加载器阶段实际 12.81 秒、退出 0；来源盘点退出 0。整体研究尚未完成，专业三维训练样本门槛未通过。

- `loader/RESULT.json`：真实上游加载器的张量／监督检查、5 个工程输入及官方 dry-run 回执。
- `loader/fixtures/`：研究者工程样本，明确 training_eligible=false；不得当作专业示范。
- `source_inventory/LOCAL_SOURCE_INVENTORY.json`：本机输入哈希与有限搜索；“原柜”目录中的壳体实际是家具数为 0 的巴黎壳体。
- `execution/`：实际运行命令、开始、首日志、结束及退出状态。
- `PROTOCOL.json`：加载器运行前冻结的范围、源码与脚本哈希。
- `MANIFEST.json`：本运行小型证据的文件哈希。

详细发现、原访谈链接、项目／知识覆盖登记及恢复命令见 `research/920/data_readiness_v1/README.md`。初始对象顶层元数据和配对元数据不会自动进入官方样本；实际理由目标可以进入，但两组工程样本预算尚为 84 对 92 个监督 token。不能据此宣布训练收益或申请直接扩训。
