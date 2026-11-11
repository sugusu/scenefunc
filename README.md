# scenefunc

[巴黎开间 A/B 实验：代码、固定依赖、实际记录与 920 复现说明](paris/README.md)

[示范真实编码与实例分配审计（2026-11-11）](runs/autodl/20261111_matched_pine_example/stage_report.html)：实际处理器已编码同输入、同53个动作token的A/DA控制；DA另有40个决定token，未加载权重或训练。新核算同时发现Pine相同替代资产下的实例分配歧义：同样终态几何对应不同旧物移动量。原示范仍可监督用途组织，不能把指定旧/新实例位置当成唯一正确功能答案。

[旧学习对照的144条响应逐条复核（2026-11-11）](runs/autodl/20261111_existing_edit_rejection_census/stage_report.html)：133次拒绝中，131次来自无参数查询携带多余参数，2次是截断输出。普通规范化已经实现并用原验证器回放；新增可接收项都是查询，不是设计改进。原有11次家具调用、旧场景结果和负面结论不改写，无新增模型/训练。

[核定一项可监督的设计选择：Pine（2026-11-11）](runs/autodl/20261111_pine_decision_supervision/stage_report.html)：已核对初始五视图、真实重建动作、部署输入和旧22步训练回执。正向示范可以用于定义模仿目标，不要求先凑两份成功偏好方案；该项目仍保留开发集排除，未新增训练。来源中的住户/设计师决定与研究者坐标、操作顺序分别标明。

[全书提取文字与SceneExplorer源码审查（2026-11-11）](runs/autodl/20261111_sceneexplorer_source_review/stage_report.html)：书312页提取文字的覆盖记录、作者接受稿和关键生成/调度源码审阅已补齐；不等于全部图表、引用论文或系统均已复现。

[整屋用途转换的实际对照（2026-11-11）](runs/autodl/20261111_lifecycle_case_comparison_v2/stage_report.html)：在另一套研究重建住宅中，普通搜索与案例条件排序都在第3个原生候选完成了声明的办公室转换，保留其余28个对象记录，终态及GLB相同；原有淋浴接近警告仍保留。两组各24次失败的前版也完整保存。新增桌椅、原生冷恢复、整屋预览和[逐组结果](runs/autodl/20261111_lifecycle_case_comparison_v2/summary.csv)已实际运行，零新增模型调用/训练；这是条件几何与用途分配证据，不是从空房完整生成成功或案例Agent创新。许可几何的绝对路径见报告内清单，不随Git发布。

[最新训练目标与真实轨迹审计（2026-11-11）](runs/autodl/20261111_trajectory_training_admission/stage_report.html)：两条已有模型轨迹、八个删除子集、24 次原生移动回放；普通规则解释了本例全部安全压缩收益，未新增完整场景成功、模型调用或训练。算法、训练目标和证明边界见[完整说明](runs/autodl/20261111_trajectory_training_admission/algorithm_and_theory.txt)，逐组结果见[summary.csv](runs/autodl/20261111_trajectory_training_admission/summary.csv)。研究目标仍未完成，这些是开发证据，不是已验证的 CVPR 方法。
