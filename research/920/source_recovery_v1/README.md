# 三维开发参考的公开来源恢复

2026-11-04 从 [Function2Scene 官方 viewer](https://function2scene.github.io/viewer.html) 的内嵌 JSON 恢复了布局参考。**其中同时有适配需求文本和 final.json 的 29 条，最终家具数合计 545，与旧交接描述的数字一致。** 这提供了可读取的公开来源，但未取得 Windows 原包逐文件比对，不能宣称整个旧专业数据包已恢复。

| 核验项 | 实际数量／状态 |
| --- | --- |
| 页面内嵌目录 | 33 |
| 场景快照 | 336 |
| 有最终布局的条目 | 30，合计 572 件生成家具 |
| 同时有需求文本和最终布局 | 29，合计 545 件生成家具 |
| 从需求文本提取的不同原文 URL | 29；不等同于已核验的独立项目数 |
| Windows 包逐文件一致性 | 未核验 |
| 7 张知识卡、165 张住宅评审图 | 此次未恢复 |
| Windows 原柜工程对照 result.json | 此次未恢复 |
| 合格专业训练决定 | 仍为 0 |

页面 SHA256 为 `4108acb36cef2ecbaaa39c385ef8da6a6fd64988b852c2a3b96c58a1523f23d0`，共 2814877 字节。下载和恢复均有实际回执；CPU 恢复耗时 0.67 秒，退出 0。清单见 [REFERENCE_INVENTORY.json](../../../runs/920/P920-SOURCE-v1_20261104T122131Z/recovery/REFERENCE_INVENTORY.json)，命令见 [运行说明](../../../runs/920/P920-SOURCE-v1_20261104T122131Z/README.md)。没有执行网页脚本、下载远程媒体或运行模型。

完整 HTML 和解析后的原数据只保存在服务器：

```text
/data/wyh/scene_cvpr_20261020/design_decisions_920/resources/research_support/public_source_inventory_20261104/function2scene_viewer.html
/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-SOURCE-v1_20261104T122131Z/recovery/embedded_data.json
```

Git 保存来源 URL、检索时间、哈希、文件清单、计数与恢复程序；未重新分发全部需求文本或布局。研究使用和再分发权限仍须核实。

## 与已知项目的对应

`19_remote_studio` 指向巴黎开间原文，`26_writer_nursary` 指向此前核验的育儿房访谈。见 [SOURCE_LINK_AUDIT.json](SOURCE_LINK_AUDIT.json)。后者在初始生成布局中有一个名为 `tall_storage_cabinet` 的对象，但没有原生对象 ID、资产 ID 或工具回执；其最终 x 坐标与初始还相差 0.01 米。这个名字不能证明保留了住户原柜，恢复的 JSON 也不能代替 Windows 工程实验。

这批数据属于生成的参考布局与适配需求。原论文将专业来源文本用于生成及功能检查，不能据此把生成终态称为设计师签认方案。未来接入时须分别标明原文事实、作者适配、研究者设定的尺寸、模型输出和实际执行结果；不能将 final.json 内的事后解释写入设计开始时的需求。

## 来源关联与项目划分

后续已阅读原文并记录 [SOURCE_RELATIONS.json](SOURCE_RELATIONS.json)：共用文章的两个儿童房是不同物理项目；巴黎项目另有跨文章重复。网址和文件夹名均不能替代项目身份。项目登记分别保留物理家族与共享文章关联；已审阅目标设计的项目只能用于开发，不能再算封存测试。尚未审阅的来源仍保持未分配。

`10_dancer-studio` 保存布局、`10_dancer_studio` 保存需求。两者的代理尺寸、房壳描述和活动内容一致，因此记录为高度一致的暂定关联，尚无作者确认；没有自动合并原始目录或改写 29／545 的原始计数。公开适配需求中的精确年龄、尺寸、活动位置和时间限制逐项区别于原文事实，不能冒充住户原话。

新增一个来源支持的活动切换决定候选，登记于 [DECISION_CANDIDATES.json](../data_readiness_v1/DECISION_CANDIDATES.json)。历史执行器固定巴黎；现已另接真实原生会话和协作式恢复，见 [入口说明](../project_input_v1/README.md)。两个候选仍未在该入口回放。后续需先明确代理场景的身份、资产和假设，再核动作及整屋后果；没有把生成终态或未核尺寸转换为监督真值。

这项恢复解除的是三维参考的可读取性问题。专业理由与实际动作的同项目对应、真实对象身份、许可和独立复核仍需补齐。

两个决定的具体输入边界与重建方案见 [审阅包](../decision_packets_v1/README.md)。新审计确认生成初稿／终稿不能直接充当专业动作轨迹；可以分配明确标注的重建代理 ID，不必以历史实物软件 UUID 为前提。
