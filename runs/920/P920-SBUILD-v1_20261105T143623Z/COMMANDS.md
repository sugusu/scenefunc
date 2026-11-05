# 实际命令与退出

工作目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc`。以下来自已执行 receipt，不是请求重跑；重复审阅只读现有结果。

## archive_execution

启动 2026-11-05T14:58:31.798494+00:00，结束 2026-11-05T14:58:31.964120+00:00，退出 1。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/archive_source_construction.py --public runs/920/P920-SBUILD-v1_20261105T143623Z --runtime ../runtime/P920-SBUILD-v1_20261105T143623Z
```

## archive_v2_execution

启动 2026-11-05T14:59:33.351085+00:00，结束 2026-11-05T14:59:34.419983+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/archive_source_construction.py --public runs/920/P920-SBUILD-v1_20261105T143623Z --runtime ../runtime/P920-SBUILD-v1_20261105T143623Z
```

## audit_execution

启动 2026-11-05T14:49:46.227597+00:00，结束 2026-11-05T14:49:47.597181+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/audit_source_construction.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/AUDIT_PLAN.json --bundle ../runtime/P920-SBUILD-v1_20261105T143623Z/native/checkpoint --out ../runtime/P920-SBUILD-v1_20261105T143623Z/FINAL_AUDIT.json
```

## build_execution

启动 2026-11-05T14:36:48.541955+00:00，结束 2026-11-05T14:36:49.108753+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/prepare_source_construction.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/BUILD_PLAN.json --out ../runtime/P920-SBUILD-v1_20261105T143623Z/build
```

## encoding_execution

启动 2026-11-05T14:49:46.286171+00:00，结束 2026-11-05T14:49:56.688774+00:00，退出 0。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/encode_construction_candidates.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/ENCODING_PLAN_V2.json --out ../runtime/P920-SBUILD-v1_20261105T143623Z/ENCODING.json
```

## endpoint_execution

启动 2026-11-05T14:56:38.991112+00:00，结束 2026-11-05T14:56:39.808927+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/measure_construction_endpoint.py --public runs/920/P920-SBUILD-v1_20261105T143623Z --runtime ../runtime/P920-SBUILD-v1_20261105T143623Z
```

## export_execution

启动 2026-11-05T14:49:46.254722+00:00，结束 2026-11-05T14:49:49.079165+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/export_catalog_geometry.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/EXPORT_PLAN.json --out ../runtime/P920-SBUILD-v1_20261105T143623Z/export
```

## native_execution

启动 2026-11-05T14:38:14.277033+00:00，结束 2026-11-05T14:45:21.488322+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/construct_source_room.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/EXECUTION_PLAN.json --phase execute --out ../runtime/P920-SBUILD-v1_20261105T143623Z/native
```

## package_execution

启动 2026-11-05T14:47:07.534283+00:00，结束 2026-11-05T14:47:07.649326+00:00，退出 1。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/package_source_construction.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/PACKAGE_PLAN.json --out ../runtime/P920-SBUILD-v1_20261105T143623Z/dataset
```

## package_v2_execution

启动 2026-11-05T14:48:42.726820+00:00，结束 2026-11-05T14:48:43.042817+00:00，退出 0。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/package_source_construction.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/PACKAGE_PLAN_V2.json --out ../runtime/P920-SBUILD-v1_20261105T143623Z/dataset_v2
```

## preflight_execution

启动 2026-11-05T14:37:29.730947+00:00，结束 2026-11-05T14:37:30.347964+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/construct_source_room.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/EXECUTION_PLAN.json --phase preflight --out ../runtime/P920-SBUILD-v1_20261105T143623Z/preflight
```

## restore_execution

启动 2026-11-05T14:49:51.600781+00:00，结束 2026-11-05T14:50:00.545927+00:00，退出 0。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/restore_terminal_catalog.py --plan runs/920/P920-SBUILD-v1_20261105T143623Z/RESTORE_PLAN.json --out ../runtime/P920-SBUILD-v1_20261105T143623Z/cold_restore
```
