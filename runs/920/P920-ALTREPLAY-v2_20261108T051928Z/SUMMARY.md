# 已有替代动作的实际后果

对训练内已有自由输出做事后单步回放：7条合法移动对应6个不同状态/动作，重复项共享一次执行；没有新生成或训练。固定对象违例、未提供工具及格式错误不修补；6条合法查询不按移动失败计分，也未继续生成。

| 状态 | 方法 | 引擎回执 | 当前用途之前→之后 | 新增失败项 |
|---|---|---|---|---|
| waterloo_edit_02 | base/action_only | no_movement | fail → fail | 无 |
| waterloo_edit_03 | base | accepted | fail → fail | no_recorded_envelope_overlap_or_boundary_violation, assigned_seat_alignment_recon_chair_b_0 |
| waterloo_initial_work_stop | base | accepted | pass → fail | no_recorded_envelope_overlap_or_boundary_violation, separate_table_gap, assigned_seat_alignment_recon_chair_b_0 |
| waterloo_initial_work_stop | action_only | accepted | pass → fail | assigned_seat_alignment_recon_chair_b_0 |
| waterloo_edit_03 | decision_action | accepted | fail → fail | 无 |
| waterloo_initial_work_stop | decision_action | no_movement | pass → pass | 无 |

6次原生调用只有4次实际变化；另2次返回 no_movement。联合组在初态办公任务的 pass→pass 是请求原位移动被引擎拒绝、场景未变，并非成功停止或新增用途收益。底座/动作组在待客任务的相同提案亦为原位移动，既有缺口未解。

评价采用既有 conditional-endpoint-v1 的已验收 V3 实现，参考对象身份不变，收益/损害相对于各自原始状态计算。yaw 原值按原生度数单位执行，保留引擎位姿噪声；并非按示范坐标相同与否评分。

一次动作失败不等于后续不能修复；一次保持不等于学到专业设计。这里只有一个曝光项目的3个编辑起点，不能证明跨项目收益。坐席、间距、方形身体通行是声明代理，真实舒适、搬运和完整房间质量仍未知。

累计执行耗时 289.689 秒（包含首批错误），续行退出码 0。首批重复注册服务在第2条动作前停止；首条结果复用，只续行剩余5条。完整检查向量、前后状态、原生回执和俯视图见 review/；服务器检查点及哈希见 RECOVERY.json。

## 已有示范中间态，事后解释用

读取历史检查点，没有重跑示范。待客示范下一步通过全部43项；恢复办公示范下一步先拉开桌子，此时椅子未跟上，仍有1项坐席失败，后续才移动椅子。因此不能要求每个中间动作立即完成整屋。模型恢复办公提案中，底座新增桌体重叠，联合组保留分桌间距不足；它们的问题依据实际几何，不依据精确坐标匹配。初态办公的参考停止保持全部34项。详见 REFERENCE_CONTEXT.json。
