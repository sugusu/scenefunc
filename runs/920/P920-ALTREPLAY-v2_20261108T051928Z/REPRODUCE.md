# 核验与恢复

执行命令和真实时间在 `execution/receipt.json`；冻结输入、代码哈希和续行预算在 `PLAN.json`。原批失败和首结果见相邻ALTREPLAY-v1目录；不要重跑原批。

在仓库根目录，仅重新汇总已有输出（0模型调用、0场景编辑）：

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/summarize_final_fit_alternatives.py --run runs/920/P920-ALTREPLAY-v2_20261108T051928Z
```

汇总器核对原HTTP动作、前后状态、checkpoint对应关系，并由保存的检查向量重算转移。历史参考检查点可由同目录 `measure_saved_reference_context.py` 只读重测。汇总会刷新报告时间与清单；不改变原始执行产物。

`RECOVERY.json`列出六个服务器checkpoint目录、manifest哈希和原动作。用冻结Scene-Designer的session/existing入口打开对应checkpoint；大型网格及五视图原件留在服务器。Git内保存逐项量测、状态、回执和每例俯视图。原生移动含位姿噪声，重发动作不是逐字节恢复；检查点才是本次结果的恢复依据。
