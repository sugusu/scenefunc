# Recover the recorded scenes without generating new outputs

Runtime root: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGEGATE-v1_20261107T071621Z`. All full checkpoints and GLB files remain on this server. Git contains small checkpoint records, receipts, state bindings and shareable views; it is not a self-contained redistribution of the asset library.

The three `RESTORE_PLAN.json` files and `COMMANDS.json` record the cold-restores already executed. They used `verify_storage_gate_terminal.py` to load and compare task, identities and receipts without edits or model calls. Quaternion roundtrips at approximately machine precision are explicitly listed in each `REVIEW_ROW.json`. The GLBs are viewing geometry; use the native checkpoints for editable state.

## frozen_base_knowledge_feedback

- Native checkpoint: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGEGATE-v1_20261107T071621Z/frozen_base_knowledge_feedback/final_checkpoint`
- Manifest SHA256: `401e7f39afd45c2639b8ccf75a2944243874a191f612f22f22109ee637a804e8`
- Canonical state SHA256: `4b880a23de71e22cd1b52628933fe03eb2e0c48084cb2c28a0c1064c5f63601b`
- GLB: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGEGATE-v1_20261107T071621Z/frozen_base_knowledge_feedback_export/room_geometry.glb`
- GLB SHA256: `b2eaaaa31ab1a10af24c9aa718b2f15199e1e0823e5edb570d70a0d6dd5cf8ec`
- Actual restore/export commands: `COMMANDS.json`, `frozen_base_knowledge_feedback/RESTORE_PLAN.json`, `frozen_base_knowledge_feedback/EXPORT_PLAN.json`.

## action_only

- Native checkpoint: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGEGATE-v1_20261107T071621Z/action_only/final_checkpoint`
- Manifest SHA256: `ca4091481d3b830d2b4e67ac7343e0ff7e6ad35229fbf772c6a3d3f31ad061be`
- Canonical state SHA256: `cfd7db7b6fef9dfbdfc88272b0f25cf8ac9ee2c1b4b1488429ad62e9391d62ba`
- GLB: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGEGATE-v1_20261107T071621Z/action_only_export/room_geometry.glb`
- GLB SHA256: `7056f92dbd79c669aa06e967c385ecf85426106882e204d5161150a6fa6081f3`
- Actual restore/export commands: `COMMANDS.json`, `action_only/RESTORE_PLAN.json`, `action_only/EXPORT_PLAN.json`.

Existing adapter: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-EDITSTUDY-v1_20261106T074752Z/training/action_only/adapter`. Hashes: `{"adapter_config.json": "e821461377ae445b18e013caeac90ecc57bc3dba1a94d811f12e2f3ada037ed1", "adapter_model.safetensors": "33e66d7edf0267ea51a83bc2c833644d714f2494f281d2130f100c0bc37ffed6"}`.

## decision_action

- Native checkpoint: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGEGATE-v1_20261107T071621Z/decision_action/final_checkpoint`
- Manifest SHA256: `d46f4d0e51105d1533b7d28a066fba9c332c309b917967ebbaba6cf785590f13`
- Canonical state SHA256: `1c552eb779c316dfc466cd807d0198c6eb44e3fa7283a7fc0ea65df46bf22b23`
- GLB: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGEGATE-v1_20261107T071621Z/decision_action_export/room_geometry.glb`
- GLB SHA256: `44c056f9d6a27741a7f68a2bcdac05cb1a313e112d51a700d297ca5214486571`
- Actual restore/export commands: `COMMANDS.json`, `decision_action/RESTORE_PLAN.json`, `decision_action/EXPORT_PLAN.json`.

Existing adapter: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-EDITSTUDY-v1_20261106T074752Z/training/decision_action/adapter`. Hashes: `{"adapter_config.json": "ac6e4729894ee84f3fcdb80b04634fa71521223c25069775d11610fa2527eec1", "adapter_model.safetensors": "1d0030bf441a4fb5b12fc48b4fb0637c2714c0d7237ee72eb2e7e99db457be7f"}`.

## Fixed resources

Base model: `XuanyuLiu/Scene-Builder-Qwen3-VL-8B-Instruct`, revision `63e235a0c0d6e5bc2f8f3485214df4457fd2e672`. The four shard URLs, sizes and SHA256 values are recorded in every `PLAN.json:model_weight_files`; tokenizer/config hashes are in `model_configuration_sha256`. Locally: `/data/wyh/scene_cvpr_20261020/design_decisions_920/resources/models/Scene-Builder-Qwen3-VL-8B-Instruct/63e235a0c0d6e5bc2f8f3485214df4457fd2e672`.

Initial asset bundle: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-STORAGE-v2_20261105T062810Z/build/bundle/manifest.json` (manifest SHA256 `dcfdab7f8a151608457bcf0036a68ea67456d8247fe95c1bea12a77e6da71f7d`). Full resource references and runtime fingerprints are retained by each native checkpoint. Preserve or restore these resources before attempting a scene recovery; archived state JSON alone cannot replace meshes. Respect original asset access rights.

Code used for inference was frozen in the plans and published in commits `94c7ed5ffcfda6869b4ac7570fada6b800693f12` and `63b0f663089ce283f7ecd32e65ea88f500ac103f`. Later closeout scripts only verify and report these outputs. Use the recorded runtime code fingerprints for recovery, rather than silently substituting a new upstream version.

Do not execute `execute_storage_learning_gate.py` again for this run. Network recovery only requires pushing the preserved commit chain.
