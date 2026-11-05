# 已执行命令

工作目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc`。此处为回执，查看结果无需重跑。

## archive_execution

2026-11-05T18:54:48.220102+00:00 → 2026-11-05T18:54:48.786980+00:00；退出0，0.566900秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/archive_mode_construction.py --public runs/920/P920-PAPER-v2_20261105T182424Z --runtime ../runtime/P920-PAPER-v2_20261105T182424Z
```

## audit_execution

2026-11-05T18:53:45.342987+00:00 → 2026-11-05T18:53:46.763626+00:00；退出0，1.420656秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/audit_source_construction.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/AUDIT_PLAN.json --bundle ../runtime/P920-PAPER-v2_20261105T182424Z/native/checkpoint --out ../runtime/P920-PAPER-v2_20261105T182424Z/FINAL_AUDIT.json
```

## export_execution

2026-11-05T18:53:45.474618+00:00 → 2026-11-05T18:53:47.647320+00:00；退出0，2.172724秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/export_catalog_geometry.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/EXPORT_PLAN.json --out ../runtime/P920-PAPER-v2_20261105T182424Z/export
```

## mode_execution

2026-11-05T18:53:45.428475+00:00 → 2026-11-05T18:53:48.404145+00:00；退出0，2.975692秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/measure_paper_modes.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/MODE_PLAN.json --states ../runtime/P920-PAPER-v2_20261105T182424Z/native --phase native --bundle ../runtime/P920-PAPER-v2_20261105T182424Z/native/checkpoint --out ../runtime/P920-PAPER-v2_20261105T182424Z/MODE_MEASUREMENT.json
```

## mode_preflight_execution

2026-11-05T18:32:30.877789+00:00 → 2026-11-05T18:32:33.753209+00:00；退出0，2.875440秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/measure_paper_modes.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/MODE_PLAN.json --states ../runtime/P920-PAPER-v2_20261105T182424Z/preflight --phase preflight --bundle ../runtime/P920-PAPER-v1_20261105T180747Z/build/bundle --out ../runtime/P920-PAPER-v2_20261105T182424Z/MODE_PREFLIGHT.json
```

## motion_limits_execution

2026-11-05T18:53:45.393052+00:00 → 2026-11-05T18:53:45.458005+00:00；退出0，0.064964秒。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/record_native_motion_limits.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/MODE_PLAN.json --native ../runtime/P920-PAPER-v2_20261105T182424Z/native --out ../runtime/P920-PAPER-v2_20261105T182424Z/TRANSLATION_MEASUREMENT.json
```

## native_execution

2026-11-05T18:39:13.638373+00:00 → 2026-11-05T18:45:58.740641+00:00；退出0，405.102292秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/construct_and_switch_room.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/EXECUTION_PLAN.json --phase execute --out ../runtime/P920-PAPER-v2_20261105T182424Z/native
```

## package_execution

2026-11-05T18:53:45.460460+00:00 → 2026-11-05T18:53:45.776472+00:00；退出0，0.316034秒。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/package_mode_construction.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/PACKAGE_PLAN.json --out ../runtime/P920-PAPER-v2_20261105T182424Z/dataset
```

## preflight_execution

2026-11-05T18:28:34.086741+00:00 → 2026-11-05T18:28:34.804089+00:00；退出0，0.717369秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/construct_and_switch_room.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/EXECUTION_PLAN.json --phase preflight --out ../runtime/P920-PAPER-v2_20261105T182424Z/preflight
```

## restore_execution

2026-11-05T18:54:10.273801+00:00 → 2026-11-05T18:54:18.517270+00:00；退出0，8.243491秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/restore_terminal_catalog.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/RESTORE_PLAN.json --out ../runtime/P920-PAPER-v2_20261105T182424Z/cold_restore
```

## translation_preflight_execution

2026-11-05T18:37:11.533256+00:00 → 2026-11-05T18:37:11.949780+00:00；退出0，0.416548秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/check_translation_sweeps.py --plan runs/920/P920-PAPER-v2_20261105T182424Z/MODE_PLAN.json --states ../runtime/P920-PAPER-v2_20261105T182424Z/preflight --phase preflight --out ../runtime/P920-PAPER-v2_20261105T182424Z/TRANSLATION_PREFLIGHT.json
```
