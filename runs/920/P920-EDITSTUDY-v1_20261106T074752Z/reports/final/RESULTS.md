# 固定开发试验：初态—终态对照

每一行是一份预定终态；未运行或未测量保留为空，不能算成功或失败。代理通过不等于整屋人类使用通过。

| 项目 | 种子 | 方法 | 状态 | 用途代理（初→末，0.5m） | 新增几何/身份违例 | 新失效接近探针 | 原角色探针可达 | 编辑状态变化次数 | GPU子进程秒 |
|---|---:|---|---|---|---|---|---|---:|---:|
| Pine | 17 | frozen_base_knowledge_feedback | policy_completed_pending_comparison | 0→0 (still_missing) | False | 无 | 4/4 | 0 | 168.21 |
| Pine | 29 | frozen_base_knowledge_feedback | policy_completed_pending_comparison | 0→0 (still_missing) | False | 无 | 4/4 | 0 | 168.56 |
| Pine | 17 | action_only | policy_completed_pending_comparison | 0→0 (still_missing) | False | 无 | 4/4 | 3 | 184.96 |
| Pine | 29 | action_only | policy_completed_pending_comparison | 0→0 (still_missing) | False | 无 | 4/4 | 0 | 178.58 |
| Pine | 17 | decision_action | policy_completed_pending_comparison | 0→0 (still_missing) | False | 无 | 4/4 | 0 | 179.44 |
| Pine | 29 | decision_action | policy_completed_pending_comparison | 0→0 (still_missing) | False | 无 | 4/4 | 0 | 176.89 |
| Milan | 17 | frozen_base_knowledge_feedback | policy_completed_pending_comparison | 1→1 (preserved) | True | shelf_front | 13/14 | 2 | 198.76 |
| Milan | 29 | frozen_base_knowledge_feedback | policy_completed_pending_comparison | 1→1 (preserved) | True | toilet_front,shelf_front | 12/14 | 2 | 239.75 |
| Milan | 17 | action_only | policy_completed_pending_comparison | 1→1 (preserved) | False | 无 | 14/14 | 0 | 202.43 |
| Milan | 29 | action_only | policy_completed_pending_comparison | 1→1 (preserved) | False | 无 | 14/14 | 0 | 202.3 |
| Milan | 17 | decision_action | policy_completed_pending_comparison | 1→1 (preserved) | True | shelf_front | 13/14 | 2 | 207.84 |
| Milan | 29 | decision_action | policy_completed_pending_comparison | 1→1 (preserved) | True | toilet_front,shelf_front | 12/14 | 2 | 248.04 |

Milan 初态已容纳分开的工作材料与两餐位，其终态通过属于保持；Pine 初态缺第二工作位。0.4/0.6m敏感性、初态原有失败、新增相交/越界、身份与原角色损失分别保存在 SUMMARY.json。

预设某侧接近点失效与对象完全不可使用分开报告；其他可达侧及座位代理在JSON中保留。严格探针失败不被删除，也不单独证明人类用途受损。

本表尚未经过最终场景目视和证据审阅，不自动输出“论文方法成立”。
