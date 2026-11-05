# 有来源的真实项目资料索引

本批取得并整理 [Gronostajska 与 Szczegielniak 的 2021 年论文](https://doi.org/10.3390/buildings11120654)，登记 30 条项目记录、面积等报告数值、附录页码和逐条原文哈希。原文正文采用 CC BY 4.0，附录说明平面示意由作者根据来源绘制。PDF、未改写的文本片段及提取图留在服务器；本仓库保存事实索引和来源定位，不复制外部摄影或把论文方案直接转成训练答案。

[TABLE1_INDEX.json](TABLE1_INDEX.json) 是看过原 PDF 表格后的事实转录。[SOURCE_INDEX.json](SOURCE_INDEX.json) 将它与表 2、附录 30 条描述和参考文献绑定。它冻结的是入库时的隔离状态，当前项目归属另见项目登记及 SELECTED_PROJECTS.json。公开数值存在分组边界和小的算术差异，原值保留，不强行修正。

[FAMILY_REVIEW.json](FAMILY_REVIEW.json)记录三个机器提示的人工式来源审阅：两处 Apartment C 只是英文短语前缀误匹配；Batiik 的小阁楼和较大花园住宅是同事务所的不同项目。Mili、Sylvie Cahen 和同篇论文的关联仍保留。30 条记录不等于已确认的 30 个独立物理家族；没有分配封存测试。

首个重建候选是 **Geneva Flat / FREAKS / 2014**，已先登记为开发家族，再产生新的输入草稿。其[建筑师提供的说明](https://www.archdaily.com/543734/geneva-flat-freaks-freearchitects)可交叉核对委托目标。它无需先依赖折叠床或夹层，因而适合优先核整体房间接入；透明分隔和完整资产供给仍待验证。

- [BUILDER_INPUT_DRAFT.json](geneva/BUILDER_INPUT_DRAFT.json)：只有改写需求、房壳及未知项，不含发表方案的设备分区、家具坐标或事后理由。
- [ENVELOPE_DIGITIZATION.json](geneva/ENVELOPE_DIGITIZATION.json)：保存原图 xref、哈希、像素选点和标尺换算。约 7.58 × 4.78 m 的矩形为 36.23 m²，与文献 35.3 m²、建筑师报道 35 m²不同；没有缩放凑面积。它是依据完成后示意图的开发房壳，不能冒充改造前测绘。
- [SOURCE_DECISION.json](geneva/SOURCE_DECISION.json)：单独保存理由来源和动作映射候选，不交给 Builder。家具集中布置可进一步核现有工具；透明分隔尚不能计作已实现。

**当前已有带声明假设的原生房壳、16项资产目录和 WHOLE／FEEDBACK 真实模型运行；两组完整用途均未通过。** 初次来源登记时的待接入事项属于历史状态。尚未完成的是与上述专业理由绑定的研究者动作示范及整屋后果复核，已有模型输出不能充当专业目标。详见 ../decision_packets_v1/SOURCED_DECISIONS.json。原始住户原话与独立专业复核仍为空。

此来源集中于紧凑住宅，且存在设计名作选择偏差，不能代表家庭住宅、儿童房或全部七类设计知识。它是数据的一部分，后续仍须补保留／材料、收纳、照明等不同项目及独立房间评价；不能用另一个小开间替代整体系统目标。

执行证据：CORPUS-v1 在启动导入处因封装器禁用用户包而退出1，未解析资料；v2明确记录现有 Python/PyMuPDF 环境后，0.8193秒退出0，完成30/30来源绑定。失败和环境说明均保留，不重跑场景、模型或训练。代码为 scripts/920/index_microapartments.py；实际命令与首日志在两个 run 的 execution/。

[完整房间能力核查](geneva/NATIVE_CAPABILITY_AUDIT.json)对既有 436 项资产锁和本机缓存作只读检查：起居、睡眠、厨房、收纳已有部分哈希验证网格；此锁内未命中马桶和洗浴类别，不代表整个 HSSD 无此资产。类别和文件完整性不等于功能验收。原生上游有目录资产添加工具，但当时会话尚未公开；后续CATALOG-v3已在独立范围接入并验收，旧会话范围不变。复核脚本和真实 0.2168 秒／退出0回执均已保存。

扩查固定 HSSD-HAB 的 semantics/objects.csv 后，取得 17,503 条元数据（4,405,002字节，Git blob 与 SHA256均核验）。精确类别有51条马桶、63条淋浴间、97条浴缸；不是全部已下载或适用。两项初选及淋浴门不可直接假定可活动的限制见 [FIXTURE_SHORTLIST.json](geneva/FIXTURE_SHORTLIST.json)。原436锁的缺口保留为历史事实，后续从完整目录补齐；两次元数据请求均有实际回执。
