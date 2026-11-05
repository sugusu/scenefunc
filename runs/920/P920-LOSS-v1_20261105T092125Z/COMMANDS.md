# Actual commands and restoration

Working directory: `/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc`.
Exact commands, real start/end times, first logs and exits: `EXECUTION.json` and `execution/*/receipt.json`.

The first preparation failed because `--out ../runtime/...` was relative. `AMENDMENT_01.json` records the additional CPU preparation before it ran; the successful command uses an absolute output path. Do not repeat the failed relative invocation when restoring.

`prepare_decision_loss.py` consumes the frozen `PLAN.json` and the original three run folders. It recreates thirty JSON components and thirty unique tensor batches via the hash-pinned upstream SFT loader. Five original images per step must be present at the paths locked by PLAN. The archived episodes are the exact inputs used, not substituted source-authored trajectories.

`run_decision_loss.py` consumes PLAN and the resulting preparation RESULT; output must be a fresh directory. It verifies weight/config/script hashes and both existing GPU0 locks before model loading. It uses BF16/SDPA, no gradients/optimizer, thirty source forwards and two short formula checks. Run through `run_stage.py` to preserve real process status. Current results are complete; network recovery needs only pushing the existing commit.

The final evidence validation also recomputed pooled NLL from token-level values, verified all original input/image hashes, parsed public JSON, checked the actual empty-think prefixes and source counts. See VALIDATION.json. It did not run backward/Trainer, perform new scene actions or create professional preference labels.
