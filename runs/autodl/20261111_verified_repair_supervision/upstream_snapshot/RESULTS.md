# Results and provenance

This file connects the results reported in
[Privileged Self-Distillation](https://www.canvas.inc/research/privileged-self-distillation)
to the experiment artifacts in this repository. The machine-readable ledger is
[`results/psd_blog_score_manifest.json`](results/psd_blog_score_manifest.json).

## Experimental settings

PSD is one method evaluated at two model scales:

| Model | Training infrastructure | Tool harness | Status |
|---|---|---|---|
| Qwen3.5-9B | Tinker | BFCL prompt mode | Published study |
| Qwen3.6-35B-A3B-FP8 | River | Qwen native tool calling | Larger-model study in progress |

The infrastructure provider is not the experimental variable. Results must be
compared within a model and harness: the 35B study is not expected to reproduce
the 9B scores numerically.

## Published evaluation protocol

- Benchmark: BFCL multi-turn tool use
- Training tasks: 360
- Held-out tasks: 240
- Held-out categories: `multi_turn_base`, `multi_turn_miss_func`,
  `multi_turn_miss_param` (80 tasks each)
- Metric: greedy pass@1, reported as tasks passed out of 240
- Repeats: repeated evaluations of one selected checkpoint, not independent
  training seeds

Observed ranges therefore measure evaluation and serving variability. Training
seed variance was not measured.

## Constructing supervision from failure

| Method | Held-out score | Evaluation totals | Status |
|---|---:|---|---|
| Base Qwen3.5-9B | 59.0 | 60, 58 | Confirmed |
| Verified-retry SFT | 77.2 | 78, 76, 78, 77 | Confirmed |
| OPD (path grading; natural coverage) | 80.0 | 81, 79, 80 | Confirmed |
| PSD, self-generated hints | 86.3 | 86, 86, 87 | Confirmed |
| PSD, frontier single-pass hints | 88.0 | 88, 87, 89 | Confirmed |
| PSD, frontier agentic search | 105.7 | 104, 108, 105 | Confirmed |

The article's OPD score of 80 is the natural-coverage path-grading arm. A
separate matched-state experiment using Qwen3.5-397B-A17B as teacher scored
77.8 (79, 77, 77, 78); it is retained in the manifest as supplementary
evidence and should not be substituted for the headline OPD arm.

## Learning from all-fail tasks

The shared first collection sampled eight rollouts for each of 360 tasks:

| Group outcome | Tasks |
|---|---:|
| All fail | 183 |
| Mixed rewards | 166 |
| All pass | 11 |

Agentic PSD produced 506 accepted repair rows covering 296 unique tasks. Of
the 183 all-fail tasks, 141 yielded an accepted repair. These counts are
recorded by:

- `results/runs/psd/train_grpo/grpo_lr1e5_screen/summary.json`
- `results/runs/psd/train_turn_kl/qwen35_uniform_r1c_lr4e5_rank32_b32_e5/summary.json`

The training summary confirms 506 accepted repair examples. The unique-task
and all-fail intersections were computed from archived row-level inputs, which
are intentionally not stored in Git.

The article reports approximately 104 for training with corrective supervision
from the all-fail slice plus preservation. That ablation still needs a
dedicated canonical run/checkpoint entry in the ledger; the value should remain
approximate until that provenance is recovered.

## Repeated campaigns

| Round | PSD | GRPO |
|---:|---:|---:|
| 0 (base) | 59.0 | 59.0 |
| 1 | 105.7 | 109.0 |
| 2 | 109.7 | 99.7 |
| 3 | 118.0 | 111.3 |

PSD reduced the number of all-fail tasks from 183 to 43 across the observed
campaign. The corresponding GRPO campaign remained near 140 all-fail tasks.

## Repair warm-starts GRPO

One all-fail-only PSD round followed by two guarded GRPO rounds reached
**146.5**. The selected final checkpoint scored 147 and 146 on two held-out
evaluations:

- `results/runs/psd/train_grpo/arm_s_r2_guarded/summary.json`
- `results/runs/psd/train_grpo/arm_s_r2_rollouts/summary.json`
- `results/runs/direct/bfcl/arm_s_r2_final*_multi_turn_*_heldout/summary.json`

The previously recorded 132.5 (134, 131) was an intermediate result after one
GRPO round. It is preserved in the manifest as a superseded score and is not
the result reported by the current article.

A separate three-round PSD plus success-imitation arm reached 129.3
(128, 131, 129). It is supplementary to the article's repair-to-GRPO result.

## Published PSD optimization recipe

The round-one agentic PSD training summary records:

| Field | Value |
|---|---:|
| Model / teacher | Qwen3.5-9B / frozen Qwen3.5-9B |
| Repair examples / targets | 506 / 506 |
| Preservation examples / targets | 356 / 815 |
| Total targets | 1,321 |
| Top-k | 20 |
| Learning rate | 4e-5 |
| LoRA rank | 32 |
| Unique targets per optimizer step | 32 |
| Epochs / optimizer steps | 5 / 210 |

Source:
`results/runs/psd/train_turn_kl/qwen35_uniform_r1c_lr4e5_rank32_b32_e5/summary.json`.

Both CLI coefficients were `1.0`, but the implementation did **not** rebalance
the aggregate repair and preservation masses. Each accepted repair contributed
one local target at weight 1.0. Preservation contributed one target per
retained assistant step, also at weight 1.0, producing 506 repair targets and
815 preservation targets before token-length effects. This is the exact
weighting semantics that the larger-model study should match.

## Qwen3.6-35B study

The Qwen3.6-35B experiment runs the same PSD method and BFCL task through
River because Tinker does not host that model. Qwen3.6 uses its native
tool-calling format, so comparisons are made only against baselines trained
and evaluated under the same model and harness.

The earlier 35B PSD and GRPO arms were not matched: GRPO received a larger
effective batch, a different learning rate, more data coverage, and a broader
tuning sweep. Those scores are diagnostic only. The next valid comparison
requires:

- native top-k cross-entropy (`river-client >= 0.5.1`);
- 32 unique targets per optimizer step;
- learning rate 4e-5;
- five complete epochs;
- all eligible preservation targets at weight 1.0 per target;
- matched checkpoint cadence and evaluation protocol.
