# Storage execution gate — closed, no demonstrated learning benefit

The fixed base and two existing EDITSTUDY adapters each made six model calls on the same furnished Guilbault reconstruction. All three preserved the room and produced recoverable terminal scenes, but **none passed the prospectively frozen geometric rule**. This was a training-exposed execution gate, not an independent test of professional design or the pairing mechanism. No training was performed.

| Model | Model calls | Native calls | Actual state changes | Rejections | Fixed access /7 | Cabinet-front access /15 | Living free-area gain, m² | AABB pairs | Rule passed |
|---|---:|---:|---:|---:|---|---|---:|---:|---|
| Fixed base + knowledge + feedback | 6 | 4 | 3 | 2 | 6 / 4 / 3 | 5 / 5 / 5 | 0.31926 | 1 | No |
| Action-only adapter | 6 | 2 | 2 | 4 | 4 / 4 / 4 | 7 / 6 / 5 | 0.32000 | 2 | No |
| Decision + action adapter | 6 | 1 | 1 | 5 | 6 / 6 / 6 | 5 / 5 / 5 | 0.32000 | 1 | No |

Access triples use declared 0.4, 0.5 and 0.6 m square bodies. The initial fixed-access counts were 6/6/6 and cabinet-front counts 10/5/5. An already executed historical reference passes 7/7/7 and 15/15/15, gains 0.82 m², and has no positive AABB pair or outside-wall failure. It was checked offline only; its coordinates were not supplied to the models or replayed in this run. Room dimensions, proxy furniture and probe definitions are researcher assumptions.

All 26 object IDs, all 23 non-target object records and cabinet identities/dimensions were preserved. No terminal object extends outside the declared wall inner faces. The base leaves cabinet A overlapping the kitchen-counter AABB; action-only overlaps the counter and cooktop; decision+action overlaps the bed. These are conservative box intersections, not mesh penetration depths or proofs of human usability. The base's 0.00654 m overlap extent on one axis must not be described as physical penetration depth. Clearing floor area alone does not meet the task.

## What failed, and what this does not identify

Both adapters claim at round 001 that all three cabinets have moved successfully, although only A was moved and B/C remain unchanged. The exact text is preserved in each adapter's `evidence/round_001/model/raw_output.txt`; native calls, states and receipts substantiate the discrepancy (see `SUMMARY.json:false_execution_claims`). Thus plausible design explanations are not reliable execution evidence. The first actual moves already incur geometric damage, before the rejected query continuations.

This gate advertised only `move_furniture_tool`. Action-only requested unadvertised `check_furniture_tool` four times; decision+action requested unadvertised `get_current_scene_state` three times and produced two malformed final outputs. The base had two malformed outputs. Full current state and measured feedback were supplied, but withholding familiar queries may affect continuation. **This interface confound prevents a clean attribution of all failures to professional-knowledge learning.** Rejection rates are not a causal improvement comparison with the historical EDITSTUDY interface.

The three groups had the same first request (7063 tokens, five images), six source-knowledge records, views, tools, feedback format and limits. Subsequent trajectories differ. All models could emit planning text; no test designer explanation or reference action was inserted. One source exposed during training and seed sequence 17+turn do not support significance or generalization claims. The common current input also differs from the adapters' old training representation; candidate_v3 has not been trained.

## Execution and evidence

Preparation began 2026-11-07 07:17:04 UTC. The three-arm execution finished 07:31:44 UTC with process exit 0 and 18/18 calls accounted for: 7 native calls, 6 state changes, 11 rejected outputs. Every arm exhausted its six-call budget without an explicit completion. Total inference-subprocess elapsed time was 361.483 seconds, including loading and encoding; this is not measured GPU utilization time. There were no reruns or extra updates. Process success and recovery success do not mean design success.

`SUMMARY.json` contains exact results and limitations; `RESULT_TABLE.csv` is the compact table. `VALIDATION.json` verifies 243 archive copies, 10 observation/state bindings, 90 model-image hashes, frozen input/code references and three restored/exported terminal scenes. The 819 code checks are 273 frozen files checked for each arm, not 819 distinct scripts. Initial and three terminal topdowns were visually inspected; no independent designer evaluation was obtained. `RECOVERY.md` and `COMMANDS.json` locate the server checkpoints, model versions and actual execution commands.

## Research decision

Close this finite gate without retries. The existing adapters have not shown enough execution reliability to justify broader room evaluation, and this run establishes no advantage for decision supervision. It does not establish that professional knowledge is useless.

Before proposing another learning run, use the existing corpus and deployment code to define one matched contract for observed state, available tools, one executed action and its receipt. Keep professional applicability decisions as the scientific target and score actual room consequences; accurate explanation alone is insufficient. A new study must separate tool/state-following competence from conditional strategy choice, use strong equal-knowledge baselines and predeclare its stopping criterion. New training requires its own concrete review; the completed EDITSTUDY authorization is consumed. Do not replace this prerequisite with more ad hoc sources, rooms or retries of closed probes.
