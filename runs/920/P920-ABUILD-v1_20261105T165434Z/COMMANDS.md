# 实际命令与退出

工作目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc`。回执记录已执行命令，重阅结果无需重跑。

## archive_execution

2026-11-05T17:06:17.930505+00:00 → 2026-11-05T17:06:17.995420+00:00；退出 1，0.064930 秒。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/archive_scoped_construction.py --public runs/920/P920-ABUILD-v1_20261105T165434Z --runtime ../runtime/P920-ABUILD-v1_20261105T165434Z
```

## archive_v2_execution

2026-11-05T17:08:18.419007+00:00 → 2026-11-05T17:08:19.286802+00:00；退出 0，0.867816 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/archive_scoped_construction.py --public runs/920/P920-ABUILD-v1_20261105T165434Z --runtime ../runtime/P920-ABUILD-v1_20261105T165434Z --allowed-failed-stage archive_execution
```

## audit_execution

2026-11-05T17:02:38.971210+00:00 → 2026-11-05T17:02:40.391095+00:00；退出 0，1.419907 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/audit_source_construction.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/AUDIT_PLAN.json --bundle ../runtime/P920-ABUILD-v1_20261105T165434Z/native/checkpoint --out ../runtime/P920-ABUILD-v1_20261105T165434Z/FINAL_AUDIT.json
```

## build_execution

2026-11-05T16:54:48.568306+00:00 → 2026-11-05T16:54:49.235664+00:00；退出 0，0.667383 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/prepare_scoped_construction.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/BUILD_PLAN.json --out ../runtime/P920-ABUILD-v1_20261105T165434Z/build
```

## endpoint_execution

2026-11-05T17:02:38.950782+00:00 → 2026-11-05T17:02:39.819279+00:00；退出 0，0.868513 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/measure_clothing_construction.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/ENDPOINT_PLAN.json --out ../runtime/P920-ABUILD-v1_20261105T165434Z/SOURCE_ENDPOINT.json
```

## export_execution

2026-11-05T17:02:38.920785+00:00 → 2026-11-05T17:02:41.746201+00:00；退出 0，2.825437 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/export_catalog_geometry.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/EXPORT_PLAN.json --out ../runtime/P920-ABUILD-v1_20261105T165434Z/export
```

## native_execution

2026-11-05T16:55:35.905648+00:00 → 2026-11-05T17:01:23.363154+00:00；退出 0，347.457527 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/construct_scoped_room.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/EXECUTION_PLAN.json --phase execute --out ../runtime/P920-ABUILD-v1_20261105T165434Z/native
```

## package_execution

2026-11-05T17:02:38.991289+00:00 → 2026-11-05T17:02:39.257145+00:00；退出 0，0.265875 秒。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/package_scoped_construction.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/PACKAGE_PLAN.json --out ../runtime/P920-ABUILD-v1_20261105T165434Z/dataset
```

## preflight_execution

2026-11-05T16:55:15.054669+00:00 → 2026-11-05T16:55:15.671534+00:00；退出 0，0.616886 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/construct_scoped_room.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/EXECUTION_PLAN.json --phase preflight --out ../runtime/P920-ABUILD-v1_20261105T165434Z/preflight
```

## restore_execution

2026-11-05T17:03:11.793058+00:00 → 2026-11-05T17:03:20.287624+00:00；退出 0，8.494589 秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/restore_terminal_catalog.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/RESTORE_PLAN.json --out ../runtime/P920-ABUILD-v1_20261105T165434Z/cold_restore
```

## storage_package_execution

2026-11-05T16:59:24.468689+00:00 → 2026-11-05T16:59:24.784764+00:00；退出 0，0.316091 秒。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/package_scoped_construction.py --plan runs/920/P920-ABUILD-v1_20261105T165434Z/storage_source_only/PACKAGE_PLAN.json --out ../runtime/P920-ABUILD-v1_20261105T165434Z/storage_dataset
```
