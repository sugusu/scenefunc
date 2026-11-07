# Actual commands and receipts
Run from repository root. These are historical executed commands, not a queue to rerun. Runtime directories are intentionally non-overwriting.

## contract_checks_receipt

Started 2026-11-07T07:59:47.611290+00:00; ended 2026-11-07T07:59:48.278552+00:00; exit 0; 0.667281s.

```sh
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/check_executed_edit_corpus.py --run runs/920/P920-EXECINPUT-v1_20261107T075321Z --out runs/920/P920-EXECINPUT-v1_20261107T075321Z/CONTRACT_CHECKS.json
```

## encoding_receipt

Started 2026-11-07T07:55:43.516982+00:00; ended 2026-11-07T07:56:05.212270+00:00; exit 0; 21.695307s.

```sh
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/materialize_executed_edit_corpus.py --plan runs/920/P920-EXECINPUT-v1_20261107T075321Z/PLAN_v3.json --mode encode --corpus runs/920/P920-EXECINPUT-v1_20261107T075321Z/corpus --out /data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-EXECINPUT-v1_20261107T075321Z/encoding
```

## materialize_phase_receipt

Started 2026-11-07T07:58:03.709071+00:00; ended 2026-11-07T07:58:04.977509+00:00; exit 0; 1.268455s.

```sh
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/materialize_executed_edit_corpus.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/PLAN_v4.json --mode materialize --corpus /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/corpus_phase
```

## materialize_receipt

Started 2026-11-07T07:53:21.347440+00:00; ended 2026-11-07T07:53:21.462676+00:00; exit 1; 0.115255s.

```sh
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/materialize_executed_edit_corpus.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/PLAN.json --mode materialize --corpus /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/corpus
```

## materialize_v2_receipt

Started 2026-11-07T07:54:06.045301+00:00; ended 2026-11-07T07:54:06.160515+00:00; exit 1; 0.115236s.

```sh
/data/wyh/conda/envs/c15_genrec/bin/python /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/scripts/920/materialize_executed_edit_corpus.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/PLAN_v2.json --mode materialize --corpus /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/corpus
```

## materialize_v3_receipt

Started 2026-11-07T07:55:17.502568+00:00; ended 2026-11-07T07:55:18.722046+00:00; exit 0; 1.219499s.

```sh
/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/scripts/920/materialize_executed_edit_corpus.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/PLAN_v3.json --mode materialize --corpus /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/corpus
```

## phase_encoding_receipt

Started 2026-11-07T07:59:25.332322+00:00; ended 2026-11-07T07:59:47.224887+00:00; exit 0; 21.892586s.

```sh
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/materialize_executed_edit_corpus.py --plan /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/PLAN_v5.json --mode encode --corpus /data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/runs/920/P920-EXECINPUT-v1_20261107T075321Z/corpus_phase --out /data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-EXECINPUT-v1_20261107T075321Z/phase_encoding
```
