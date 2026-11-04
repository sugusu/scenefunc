# 自行寻找可用项目与专业依据

用户目前没有可提供的数据包或设计师复核资源，要求执行者自行寻找。本轮已找到一份明确许可的专业研究正文及三个可交叉核对的住宅候选；没有等待用户提供资料，也没有开始训练或联系作者。

Arkan Demirörs 与 Keresteci 的 [2025 年研究](https://dergipark.org.tr/en/pub/ks/article/1789863)在首页明确采用 CC BY-NC 4.0，表 1 给出三案的功能切换分析。已下载 17 页 PDF，直接查看许可页、分析表和参考文献页。该分析可作为有署名的外部研究依据；它没有评审我们的三维重建。第三方图片和图纸的再分发权限单列，原 PDF 和预览只保存在服务器。

交叉核验发现引用编号错位：Bukit Batok 案正文的 URL 2 实际指向米兰；米兰的 URL 3 指向 Newton；Newton 的 URL 4 指向 Bukit Batok。以下按原项目报道重新对应，原论文不改写：

| 候选项目 | 已核来源 | 接入判断 |
| --- | --- | --- |
| Bukit Batok / Interim Spaces | [设计师访谈](https://www.nevertoosmall.com/post/flexible-home-for-a-singapore-social-worker)：38 m²，工作、书画及起居需求，可移动餐桌／工作台。 | 优先审查桌椅移动；折叠床尚未接入。 |
| Milan / TAAAC | [原报道](https://www.nevertoosmall.com/post/taaac-milan)：30 m²，事务所署名 ATOMAA。 | 活动墙及折叠部件超出已验证工具；个人署名暂不确认。 |
| Newton / Design Pair | [住户兼设计师访谈](https://www.nevertoosmall.com/post/one-room-three-ways-in-a-convertible-singapore-apartment)：58 m²，家庭变化与可变分隔。 | 窗帘与折叠床需要真实工具能力，不能用删家具代替。 |

这些项目已经阅读过成品资料，只能进入开发来源审查。已对 33 条公开参考元数据核查来源 URL 和项目别名，三案无精确匹配，并与原五家族区分，登记为八个开发家族；这不排除未发现的别名。Bukit 的前置事实、方案内容、无尺寸平面和工具缺口见 [BUKIT_EVIDENCE.json](BUKIT_EVIDENCE.json)。原始任务书、精确三维初态和对我们候选的独立评审仍未获得；所有候选均未加入训练。

另核验了 [WikiArch](https://github.com/danruili/ArchLogicRAG/tree/e35bc92127d67c783718916cba945b4f49a78e35)的四个文件，Git blob SHA 全部一致。70 条项目、182 个图片链接是 CC0 元数据；不包含正文或图片。其人工标注针对检索，论文说明多数标注者是建筑学生，不能当整屋专业评分。未读取基准答案或启动付费 API。

[马拉加住宅修复研究](https://doi.org/10.1108/JCHMSD-10-2025-0387)有 CC BY 4.0 正文和保留、复用方面的说明，可继续补知识覆盖；目前没有核到可直接回放的家具前后状态。RenoLayout、OptiScene、照明数据、Tesseract 等来源的用途与缺口见 [SOURCE_AUDIT.json](SOURCE_AUDIT.json)，不把合成数据或只有检索标签的材料冒充真实项目决定。

下载命令、真实时间与退出状态在 `execution/`；本地原件、哈希和恢复地址在 `SERVER_ARTIFACTS.json`。PDF 下载 3.932 秒退出 0，1236691 字节。公开仓库只保存来源判断和必要记录。

后续又取得两份原文，详见 [扩展审计](expansion_20261104T1752Z/AUDIT.json)：[Inside a Microapartment](https://www.mdpi.com/2075-5309/11/12/654) 的 CC BY 正文提供 30 案的资料入口，附录有作者绘制的带比例平面示意，适合优先审阅同项目证据；[Bollo 的研究](https://onlinelibrary.wiley.com/doi/10.1111/joid.12235)分析 24 套住房，专家参与的是指标审查，采用 BY-NC-ND，当前只用作方法参考。两者都没有评审我们的候选，不能把出现频率或空间属性直接当作住户满意度。原文已留服务器并记录哈希，尚未派生训练数据。

后续已完成 [30 案来源索引](../microapartments_v1/README.md)，Geneva 先登记为第九个开发家族，再派生输入草稿。原三案去重结果保留；30 条索引不是 30 个已核独立家族。马拉加的后续审阅和原文数值矛盾见 [MALAGA_EVIDENCE.json](MALAGA_EVIDENCE.json)，仍不准入训练。
