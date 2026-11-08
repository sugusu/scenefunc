# 实际 CPU 命令与恢复

各阶段均已完成，不应重启。实际命令来自不可覆盖的执行回执：

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/balance_native_conditions.py --plan runs/920/P920-MATCHEDINPUT-v1_20261108T055328Z/PLAN.json
```

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/materialize_native_completion.py --plan runs/920/P920-MATCHEDINPUT-v1_20261108T055328Z/PLAN.json --phase encode
```

完整外层命令采用 `python3 scripts/920/run_stage.py --receipt-dir <该阶段receipt目录> --timeout 600 -- <上列命令>`。

运行目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-MATCHEDINPUT-v1_20261108T055328Z`；张量目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-MATCHEDINPUT-v1_20261108T055328Z/encoding`。所有张量 SHA256 见 ENCODING.json，底座固定版本及官方源码哈希见 PLAN.json。权重、原始图像和网格沿源清单留在服务器。

若迁移后恢复，先按源清单恢复固定资源，逐项核验哈希；在新的恢复目录与新计划中运行，不覆盖本次结果。编码需本机 c15_genrec 环境；准备需 scene_designer311_native。路径变更必须保留映射并重新审计。张量容器文件字节未必跨 PyTorch 版本一致，应同时核目标 token、像素和掩码。
