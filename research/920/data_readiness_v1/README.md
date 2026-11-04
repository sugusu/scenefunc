# 真实项目证据与官方加载器核验 v1

2026-11-04。本轮推进整体研究的数据和算法入口，未训练、未加载大模型、未新增场景生成。CPU 加载器核验实际耗时 12.81 秒、退出 0；运行证据在 `runs/920/P920-DATA-v1_20261104T114417Z/`。

## 可以确认什么

1. 本机 `inputs_v1/同项目原柜实验/冻结开发输入/19_remote_studio/initial.json` 与巴黎交付房壳逐字节一致，SHA256 为 `54b98aa6070a14f82c91e88e51f6f84b5df4b192f179dba206e515ba156d2b5f`，其中家具数为 0。目录名称不能证明它是育儿房原柜场景。
2. 在指定本机目录和当前 GitHub main 树中，未定位到附件描述的完整专业设计数据包。当前 main 的 229 个路径只有巴黎交付及根说明；这不证明其他机器或路径没有数据。7 张知识卡、165 张图、29 间房／545 件家具和 99 条二维评审仍是待核验的历史描述。用户追问后，扩查 `/data/wyh` 与当前附件的 384452 个路径，排除环境／缓存目录，未找到当前实验以外的相关文件名；附件文字仍指向 Windows 桌面。原交接 payload 清单及哈希核验确认只传入上述四个巴黎文件，见 `SOURCE_TRANSFER_CHECK.json`。没有解压无关归档，不能据此断言任意名称或其他机器下不存在该包。
3. [育儿房原访谈](https://www.architecturaldigest.com/story/how-writer-erika-veurink-built-an-impressionist-inspired-nyc-nursery-for-5000-dollars)可支持现有候选的保留与外观改造理由；登记见 DECISION_CANDIDATES.json。原始三维状态、原柜 ID、回放结果、权限及标签复核仍缺失。没有从照片推断坐标，也没有把手绘饰面或容量标为已实现。
4. DATA-v1 执行时登记了两项目家族、一个来源支持的候选，均不具备训练资格。后续来源审阅将登记扩展为 **9 个开发项目家族、2 个既有决定候选；另有 30 条论文来源记录待逐项资格审阅，合格专业训练决定仍为 0**。PROJECT_REGISTRY.json 与 KNOWLEDGE_COVERAGE.json 分别记录项目和七类知识空缺；这些已审阅项目不能充当封存测试。

后续来源恢复：从 Function2Scene 官网内嵌数据取得了三维参考，其中 29 条带需求的最终布局合计 545 件家具，与历史计数一致，见 [来源恢复说明](../source_recovery_v1/README.md)。这不是从 920 找到旧包，而是重新取得公开来源；Windows 包字节一致性、知识卡、二维评审图和原柜工程结果仍未核验。生成参考不升级为专业训练真值。

## 实际加载器的发现

使用固定 [Scene-Designer 源码 a0565e5](https://github.com/ZJU-OmniAI/Scene-Designer/tree/a0565e50ed7d32c0e09132b9311c0b9b91165405)，7 个文件均核对 Git blob 与 SHA256。源码只下载到服务器，不在本仓库重新分发；恢复清单见 OFFICIAL_SOURCE_LOCK.json。

输入通过官方 `StepwiseSFTDataset.__getitem__` 和固定 Scene-Builder processor，另实际运行了官方 `python -m src.train_sft --dry-run --use-cpu`。使用原巴黎 A 的五张已保存视图和对象表，只检查数据传递。理由与动作是明确标注的研究者工程样本，动作未执行，不构成专业示范。

| 检查 | 实际结果 | 对后续实现的要求 |
| --- | --- | --- |
| 仅改变 project_pair_id／paired_reason 元数据 | 全部输入张量及 labels 相同 | 配对必须进入实际上下文或监督输出，仅改标签无效。 |
| 将相同理由放到 assistant.content | 监督 token 从 55 变为 84，理由出现在解码目标中 | 正确配对有可执行的数据入口，但还须使用可信同项目理由。 |
| 分开学习相同理由与动作 | 动作 55 tokens，加 QA 理由 37 tokens，共 92 | 与配对组 84 不匹配；当前样本数亦为 2 对 1。须匹配更新步数和监督预算，不能直接开训比较。 |
| 只添加顶层 initial_state | 张量不变，原生 memory 仍为空 | 不能假设加载器自动读取初始对象表。该工程样本已将真实对象表明确写入 task_context，故其对象仍可见；下一步应规范状态导入，而非伪造历史工具回执。 |
| 输入和监督边界 | prompt 是完整序列的精确前缀；前缀全部为 -100，助手部分保留 | 当前样本的文字与图像输入均不计 loss。评分专用哨兵元数据不进入序列。 |
| 五视图与批处理 | 保留 5 视图，缺图／错序拒绝，padding 不计 loss | 保留上游真实校验路径，不能以单张图静默替代。 |

此轮未检验真实专业轨迹的工具回执回放、OTC 专业评审质量、LoRA 参数更新或任何生成收益。没有把“输入确实不同”写成“正确配对一定更好”。**加载器工程检查通过，可信专业样本的训练前门槛仍未通过。**

## 恢复与复核

源码缺失时，在仓库根使用小文件下载器和固定清单；第三方源码保留服务器。随后可在新的输出目录运行审计，禁止覆盖既有记录。

```bash
python3 scripts/920/fetch_upstream.py \
  --lock research/920/data_readiness_v1/OFFICIAL_SOURCE_LOCK.json \
  --data /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/recovered_sft \
  --receipt /tmp/scenefunc_sft_source_recovery.json

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /data/wyh/conda/envs/c15_genrec/bin/python scripts/920/audit_sft_loader.py \
  --root /data/wyh/scene_cvpr_20261020/design_decisions_920 \
  --source /data/wyh/scene_cvpr_20261020/design_decisions_920/resources/recovered_sft/upstream/Scene-Designer/a0565e50ed7d32c0e09132b9311c0b9b91165405 \
  --lock research/920/data_readiness_v1/OFFICIAL_SOURCE_LOCK.json \
  --out /tmp/scenefunc_sft_loader_recheck_new
```

前置依赖为固定模型 processor 小文件和既有 P920-DD-v1 A 草稿五视图／checkpoint；本轮未重新运行模型、构造房间或复制大型资源。实际命令、开始／结束时间、首日志和退出状态均在运行目录 `execution/`。源数据盘点可用 `audit_project_sources.py --root <实验根> --out <新目录>` 重算，搜索范围和局限写入结果。

后续 [状态接口核验](../STATE_INTERFACE.md) 已用四个保存回合确认显式初态／真实既有回执进入被屏蔽输入，且状态差异有据可查。[评审与资格清单核验](../critic_contract_v1/README.md) 已完成实际加载器检查，并拒绝将工程样本送入训练目录。下一步是可信同项目资料、通用房间输入及整屋验收；批准训练之前，仍须有合格样本、预算匹配方案及单独审定。

后续公平预算核验：[P920-PAIR-v1](../../../runs/920/P920-PAIR-v1_20261104T174408Z/README.md)在真实官方加载器上得到两组各 83 个相同目标 token、2 条样本、10 次图像输入，17 项检查通过。该新格式没有训练，未证明运行时成本相等，也未完成官方训练入口集成；旧表中的 84/92 是先前格式的真实结果，保留不改。
