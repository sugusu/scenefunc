# 已批准评测续行：最终结果

12/12 终态已完成，实际退出码 0，产物与成本核验通过；本次 0 次新训练。三组均没有取得“活动达标且不新增已测损害”的终态。本配置没有显示专业决定监督的额外设计收益。

## 同批比较

| 方法 | 终态数 | 活动达标 | 达标且无新增损害 | 有新增损害 | 拒绝／模型调用 | 原生执行 | 状态改变 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 底座＋同知识／反馈 | 4 | 3 | 0 | 4 | 7/24 | 17 | 14 |
| 动作微调 | 4 | 2 | 0 | 4 | 5/24 | 19 | 16 |
| 决定＋动作微调 | 4 | 2 | 0 | 4 | 4/24 | 20 | 16 |

活动达标只对应冻结代理指标；不能抵消删除保留对象等副作用。所有格均耗尽 6 次调用预算，没有模型主动完成。共 56 次原生执行、46 次状态改变，不能把这次全部失败解释为接口没有执行。联合组少几次拒绝不等于设计更好。

## 十二份终态

| 用途 | 种子 | 方法 | 活动达标 | 新增失败检查 |
|---|---:|---|---|---|
| clear_floor | 17 | 底座＋同知识／反馈 | pass | all_original_instances_and_assets |
| clear_floor | 17 | 动作微调 | pass | all_original_instances_and_assets |
| clear_floor | 17 | 决定＋动作微调 | pass | all_original_instances_and_assets, no_recorded_envelope_overlap_or_boundary_violation |
| clear_floor | 29 | 底座＋同知识／反馈 | pass | all_original_instances_and_assets |
| clear_floor | 29 | 动作微调 | pass | all_original_instances_and_assets, no_recorded_envelope_overlap_or_boundary_violation |
| clear_floor | 29 | 决定＋动作微调 | pass | all_original_instances_and_assets, no_recorded_envelope_overlap_or_boundary_violation |
| desk_work | 17 | 底座＋同知识／反馈 | fail | all_original_instances_and_assets, access_kitchen_front_width_0.4, access_kitchen_front_width_0.5, access_kitchen_front_width_0.6 |
| desk_work | 17 | 动作微调 | fail | no_recorded_envelope_overlap_or_boundary_violation, access_kitchen_front_width_0.4, access_kitchen_front_width_0.5, access_kitchen_front_width_0.6 |
| desk_work | 17 | 决定＋动作微调 | fail | no_recorded_envelope_overlap_or_boundary_violation, desk_knee_volume_clear |
| desk_work | 29 | 底座＋同知识／反馈 | pass | all_original_instances_and_assets |
| desk_work | 29 | 动作微调 | fail | desk_knee_volume_clear |
| desk_work | 29 | 决定＋动作微调 | fail | no_recorded_envelope_overlap_or_boundary_violation, desk_knee_volume_clear |

检查名释义：`all_original_instances_and_assets` 为原对象实例及资产保留；`no_recorded_envelope_overlap_or_boundary_violation` 为记录的包围几何相交／越界；`access_kitchen_front_width_*` 为声明身体宽度的厨房接近；`desk_knee_volume_clear` 为工作位膝部代理体积。原有 0.6 m 卫浴接近失败仍保留在绝对缺陷中，不计新增损害。实际字段与逐格终态见 [验收 JSON](review/FINAL_ACCEPTANCE.json) 和 [CSV](LEARNING_RESULTS.csv)。

清空活动区的六格全部达到活动指标，也全部破坏原实例保留。办公任务只有底座种子 29 达到活动指标，但同样破坏保留要求；其他格包含工作位、几何或接近失败。这里报告的是执行后可计算后果，不把研究者代理当作真实住户舒适或专业综合评价。

## 成本、来源与恢复

- 真实开始：2026-11-07T16:50:06.580393+00:00；结束：2026-11-07T18:03:55.742569+00:00；外层经过 4429.162 秒，exit 0。
- 新增模型调用 71，新增 GPU 子进程 1372.130446 秒；承接原首拒绝 1 次／24.048121 秒。累计评测 1396.178566 秒，未超过批准的 7200 秒。
- 既有训练 5512.523019 秒，两个 final-step-8 适配器复用；训练加评测累计 6908.701585 秒。本次无新增反向或更新。
- 原 STOPDATA 基础设施失败、首原始输出及其成本原样留存。续行没有把失败记成新试验或清零预算。
- [授权](AUTHORIZATION.json)、[冻结方案](STUDY_PLAN.json)、[真实执行命令和首日志](execution_receipt/receipt.json)、[首原始输出承接](first_new_output/)、[终态及恢复索引](SCENES.md)。
- 底座 Scene-Builder 固定版本 `63e235a0c0d6e5bc2f8f3485214df4457fd2e672`；适配器路径和 SHA256 在验收 JSON 的 training 字段。checkpoint、权重留在服务器。

复核命令（只读取已有产物，使用新的输出目录；不重启训练或评测）：
```bash
python3 scripts/920/review_resumed_evaluation.py \
  --public runs/920/P920-EVALFIX-v1_20261107T130916Z \
  --out /tmp/evalfix-independent-review
```

## 结论和下一步取舍

这次完成了有共同输入、实际执行和完整终态的有限比较，未观察到决定＋动作监督相对动作微调或强底座的有效改善。因此不自动扩训或追加同一配置的评测。下一步优先利用已保存的动作、回执和训练记录，区分条件选择、约束保留和训练拟合问题；形成可验证的机制解释后，才提出新的有限学习方案。

本试验不能证明专业知识学习无效，也不能证明模型已经充分学会训练任务。训练日志下降不是独立最终拟合验收；动作组和联合组的原始损失监督内容不同，不能直接比较数值高低。

## 适用范围

- 仅一个已暴露开发项目族 `paris_bien_vivre_16m2_nabucet_baillargeon`，训练／检索隔离，但不是封存测试；2 用途 ×2 种子不是 4 个独立项目。
- 房间是研究者构造的 4×4 m 代理，不是原住宅完整建筑重建；视图和导出保留这种限制。
- 既有训练每组 24 样本（22 工具步骤＋2 研究者停止控制）、6 来源族；8 个决定位置来自 5 原则。共同控制目标 1647 token，联合另有 1332 决定 token，因此不是等材料配对机制证明。
- 每组 8 次更新、少量相关样本，不能由本轮否定所有监督算法；没有独立专业偏好、人体验证或未见项目泛化证据。
- 未知的容量、舒适、搬运可行性与整屋真实功能不计成功。不同任务的旧 Pine/Milan 等结果不能拿来作本批因果对照。
