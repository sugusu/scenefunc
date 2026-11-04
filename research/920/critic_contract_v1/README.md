# 专业评审入口与训练数据资格检查

2026-11-04。固定 Scene-Designer `a0565e50ed7d32c0e09132b9311c0b9b91165405`。新增读取的 4 个官方文件均核对 Git blob／SHA256，清单见 `OFFICIAL_CRITIC_SOURCE_LOCK.json`；源码只保存在服务器。实际 CPU 核验 `P920-REVIEW-v1_20261104T123915Z` 已完成，9.30 秒、退出 0，18 项检查通过。使用已有开发场景和明确标记的工程判断，没有调用模型、修改场景或训练。

| 本研究要求 | 固定官方实现 | 接入决定 |
| --- | --- | --- |
| 按需求评审候选 | SFT `critic` 从 `prompt` 取问题，将有序图片列表作为候选 A/B/C/D；输出 reasoning 与 answer，没有额外 system turn | 需求、候选对象和可用观察须进入 prompt，不能只放元数据。 |
| 候选位置与答案对应 | 图片列表顺序直接决定字母含义；加载器不替调用者验证语义对应 | 从稳定 candidate ID 生成槽位映射；图片换序时同步重映射答案，缺图拒绝。 |
| 允许并列／无法判断 | `src_critic/validate_sft.py` 的严格排序解析拒绝等号和 tie；pair judge 主要要求 A 或 B | 明确偏好走 critic；并列／证据不足走同一官方 SFT 的 QA，结构化答案可完整读回，unknown 没有被转为胜负。 |
| 观察完整性 | 实测官方 QA 缺图时静默跳过，10 图变成 9 图；critic 缺图会报错 | 导出前核验场景、渲染回执、视图顺序和文件哈希，拒绝缺图及跨候选错图。 |
| 项目家族隔离 | 官方 `split_dataset.py` 按 `scene_dir` 的目录名分组 | 增加来源／项目家族登记，先划分再派生；目录名不能单独证明项目独立。 |
| 工程样本不进入训练 | 实际加载器不会按 `training_eligible` 元数据过滤 | 实现显式 manifest 资格检查和名单复制。当前工程样本被拒绝，未建立训练目录；专用审计目录只含指定两条样本，相邻 JSON 未被实际加载器读取。 |

可追溯源码：[SFT 加载器](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/sft_dataset.py)、[排序校验](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src_critic/validate_sft.py)、[划分](https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src_critic/split_dataset.py)。

## 公开数据的用途边界

[Scene-OTC-Bench 数据卡](https://huggingface.co/datasets/XuanyuLiu/Scene-OTC-Bench/blob/e50788bdbe74644d98e62929cd24218c9110e812/README.md)明确发布的是评测集：94 个双候选、55 个四候选问题。此次只读取公开元数据和说明，没有下载答案 JSON 或图片；不能用它补足训练样本。元数据没有给出 license 字段，未将公开可访问解释成新增训练许可。

[RenoLayout 论文附录 A](https://arxiv.org/html/2610.21366v1)说明其训练语料无法公开，200 个评测案例计划经 OpenReview 提供。此次未确认可下载的正式发布入口，不把论文中的规模记成本项目已拥有的数据。Function2Scene 的公开生成参考已另行恢复，见 [来源恢复](../source_recovery_v1/README.md)，同样不能代替专业配对动作。

## 本次核验说明什么

[实际结果](../../../runs/920/P920-REVIEW-v1_20261104T123915Z/audit/RESULT.json)和[运行命令](../../../runs/920/P920-REVIEW-v1_20261104T123915Z/README.md)保存四种输入：明确偏好、并列、证据不足，以及换序后的明确偏好。critic 使用每候选一张真实俯视图，QA 使用每候选五张真实视图。输入和图片均被 loss mask 屏蔽；评审标注只出现在监督目标。交换候选顺序确实改变图像张量，答案字母同步换位，解析后的稳定 candidate ID 不变。

`scripts/920/review_samples.py` 校验对象、共同起点、原需求与观察来源，`training_manifest.py` 检查项目登记、划分早于派生、文件哈希和审定记录，只复制名单内样本。当前工程样本、同家族跨划分、封存测试检索和篡改哈希均被拒绝。审定记录检查的是声明与引用的一致性，不会自动证明许可、签名身份或专业判断真实有效。没有真实获审定样本，正向专业数据准入尚未验收。

工程判断只衡量指定原 ID 是否保留：并列表示该单项相等，不能解释成整屋质量相等。现有视图没有原生对象 ID 标注，不能据此声称专业视觉定位已验证。QA 结构化结论也不是官方严格排序评测的分数；还需独立语义核验及人类评审。合格专业训练决定仍为 0，训练授权仍须单独取得。
