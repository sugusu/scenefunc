# Matched complete-trajectory diagnostic

| Condition | Weights | Initial → final measured status | Valid completion | Gate | Calls |
| --- | --- | --- | --- | --- | --- |
| waterloo_initial_work_stop | frozen_base | pass → pass | False | False | 4 |
| waterloo_initial_work_stop | action_only | pass → fail | False | False | 4 |
| waterloo_initial_work_stop | decision_action | pass → pass | True | True | 2 |
| waterloo_edit_01 | frozen_base | fail → fail | True | False | 2 |
| waterloo_edit_01 | action_only | fail → fail | False | False | 4 |
| waterloo_edit_01 | decision_action | fail → fail | True | False | 4 |
| waterloo_joined_host_stop | frozen_base | pass → pass | True | True | 4 |
| waterloo_joined_host_stop | action_only | pass → pass | True | True | 2 |
| waterloo_joined_host_stop | decision_action | pass → pass | True | True | 1 |
| waterloo_edit_03 | frozen_base | fail → fail | True | False | 4 |
| waterloo_edit_03 | action_only | fail → fail | True | False | 2 |
| waterloo_edit_03 | decision_action | fail → fail | True | False | 2 |

| Weights | Already-satisfied starts: gate passed | Deficit starts: measured repair | New hard failures |
| --- | --- | --- | --- |
| frozen_base | 1/2 | 0/2 | 0 |
| action_only | 1/2 | 0/2 | 1 |
| decision_action | 2/2 | 0/2 | 0 |

Actual calls: 35; GPU subprocess seconds: 872.032; new training updates: 0.

Gate requires declared activity pass, no new declared hard failure, original object identities and valid native completion. Satisfied starts additionally require no state changes. A refusal, exhausted budget or explanation alone is not completion.

One training-exposed project, four starts, one seed; these are not twelve independent projects. All methods use the same current inputs and feedback. Old adapters predate the matched/native-stop input changes. No new-training, professional quality, generalization or pairing-mechanism gain is established. Full-room comfort, physical moveability and other unknowns remain unknown.

The frozen completion gate is stricter than the official rollout parser: malformed no-tool emissions can terminate upstream but are rejected here. See PARSER_SENSITIVITY.json for a post hoc, zero-generation audit. It does not replace primary scores or assert an alternative trajectory was executed.
