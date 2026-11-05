# 已执行命令

工作目录：`/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc`。此处为回执，查看结果无需重跑。

## build_execution

2026-11-05T18:13:05.533841+00:00 → 2026-11-05T18:13:06.100749+00:00；退出0，0.566930秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/prepare_scoped_construction.py --plan runs/920/P920-PAPER-v1_20261105T180747Z/BUILD_PLAN.json --out ../runtime/P920-PAPER-v1_20261105T180747Z/build
```

## mode_preflight_execution

2026-11-05T18:17:54.616032+00:00 → 2026-11-05T18:17:57.541239+00:00；退出0，2.925229秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/measure_paper_modes.py --plan runs/920/P920-PAPER-v1_20261105T180747Z/MODE_PLAN.json --states ../runtime/P920-PAPER-v1_20261105T180747Z/preflight --phase preflight --bundle ../runtime/P920-PAPER-v1_20261105T180747Z/build/bundle --out ../runtime/P920-PAPER-v1_20261105T180747Z/MODE_PREFLIGHT.json
```

## preflight_execution

2026-11-05T18:15:04.367055+00:00 → 2026-11-05T18:15:05.084947+00:00；退出0，0.717915秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/construct_and_switch_room.py --plan runs/920/P920-PAPER-v1_20261105T180747Z/EXECUTION_PLAN.json --phase preflight --out ../runtime/P920-PAPER-v1_20261105T180747Z/preflight
```

## prepare_execution

2026-11-05T18:09:21.857077+00:00 → 2026-11-05T18:09:22.324120+00:00；退出0，0.467066秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/geometry311/bin/python scripts/920/prepare_paper_modes.py --public runs/920/P920-PAPER-v1_20261105T180747Z --runtime ../runtime/P920-PAPER-v1_20261105T180747Z
```

## reference_execution

2026-11-05T18:10:58.248622+00:00 → 2026-11-05T18:11:02.829746+00:00；退出0，4.581145秒。

```bash
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/build_reference_room.py --plan runs/920/P920-PAPER-v1_20261105T180747Z/REFERENCE_BUILD_PLAN.json --scenesmith ../resources/full_runtime_20261104/scenesmith --out ../runtime/P920-PAPER-v1_20261105T180747Z/reference
```
