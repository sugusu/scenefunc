Native launch (one bounded batch):

```sh
python3 scripts/920/run_stage.py --receipt-dir runs/920/P920-ALTREPLAY-v1_20261108T051517Z/execution --timeout 1200 -- /data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python scripts/920/replay_final_fit_alternatives.py --plan runs/920/P920-ALTREPLAY-v1_20261108T051517Z/PLAN.json
```

Preparation: `--validate-only` passed source object tables against all original checkpoints, unchanged raw actions, six unique/seven arm coverage and allowed objects. An earlier read-only draft check rejected obsolete ENDPOINT hashes; PREPARATION_DRAFT.json retained, final plan uses previously accepted V3 evaluator with identical activity program. No native/model calls occurred before final validation.
