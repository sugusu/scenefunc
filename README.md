# scenefunc

[巴黎开间 A/B 实验：代码、固定依赖、实际记录与 920 复现说明](paris/README.md)

[整屋用途转换的实际对照（2026-11-11）](runs/autodl/20261111_lifecycle_case_comparison_v2/stage_report.html)：在另一套研究重建住宅中，普通搜索与案例条件排序都在第3个原生候选完成了声明的办公室转换，保留其余28个对象记录，终态及GLB相同；原有淋浴接近警告仍保留。两组各24次失败的前版也完整保存。新增桌椅、原生冷恢复、整屋预览和[逐组结果](runs/autodl/20261111_lifecycle_case_comparison_v2/summary.csv)已实际运行，零新增模型调用/训练；这是条件几何与用途分配证据，不是从空房完整生成成功或案例Agent创新。许可几何的绝对路径见报告内清单，不随Git发布。

[最新训练目标与真实轨迹审计（2026-11-11）](runs/autodl/20261111_trajectory_training_admission/stage_report.html)：两条已有模型轨迹、八个删除子集、24 次原生移动回放；普通规则解释了本例全部安全压缩收益，未新增完整场景成功、模型调用或训练。算法、训练目标和证明边界见[完整说明](runs/autodl/20261111_trajectory_training_admission/algorithm_and_theory.txt)，逐组结果见[summary.csv](runs/autodl/20261111_trajectory_training_admission/summary.csv)。研究目标仍未完成，这些是开发证据，不是已验证的 CVPR 方法。
