# 两条专业知识记录与表达范围

本批新增两条建筑师有明确解释的知识记录；没有新增三维动作、专业评价或训练。来源、研究解释、接口限制分别保存在 NEW_KNOWLEDGE.json、PRIMARY_SOURCE_AUDIT.json 和 TOOL_SCOPE_AUDIT.json。

| 项目 | 条件与决定 | 当前可用层 | 未验证机制 |
|---|---|---|---|
| [29sqm / Ewa Czerny](https://www.archdaily.com/387467/29sqm-3xa) | 紧凑住宅及有限层高，采用局部夹层并按活动分配净高 | 有归属的知识解释、检索候选 | 承重、楼梯/姿态、分层通行与净高；报道高度不是通用标准 |
| [Appartement Spectral](https://www.archdaily.com/369075/appartement-spectral-betillon-dorval-nil-bory) | 自然光不足，按作者判断的活动辨色需求组织不同光源 | 有归属的知识解释、检索候选 | 光谱、显色、实际光分布和体验；RGB着色不构成验证 |

原作者页面两次访问失败后，采用明确注明建筑师供稿的 ArchDaily 项目说明。原网页未整页缓存；JSON 哈希只标识本研究提取记录。既有论文片段文件哈希已核，面积及作者别名差异保留。未复制原照片；知识来源成立不自动解决全部后续复用范围。

接口审计仅检查固定版工具 schema、实际家具 move 包装器和现有平面通行程序，不声称穷尽 SceneSmith 或 Blender 的功能。两条 QA 是研究者根据来源编排的文本解释候选，没有实际加载器验收，没有模拟专家评分。零新动作、零模型调用、零训练。

本批后合计11条专业知识、5条部分可表达、4个已执行来源决定（7个重建版本）；其中3个仅支持条件几何收益。外部 Mini 的165条评价资料与本研究候选的真人评价0分别统计。纠正的旧计数不作为本批新增成果。17登记家族中12开发、5未来训练来源、0封存测试；这不表示17个独立可训练项目。

复核入口：RESULT.json、AUDIT_PLAN.json、FINAL_CHECKS.json、COMMANDS.json；实际启动与退出见 execution/receipt.json。已有 STORAGE-v2 与 STEPPAIR 的前后场景、逐步输入和真实加载器证据继续沿原路径保留，本次不重跑。

完整性检查首次发现 AIR-v2 的当前明细漏记条件收益标志；依据原发布结果补标，原失败与依据见 FINAL_CHECKS_ATTEMPT_1.json 和 COUNT_RECONCILIATION.json。最终15项一致性检查通过，没有重跑场景。
