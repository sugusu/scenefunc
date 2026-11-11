# Privileged Self-Distillation

Official code for
**[Privileged Self-Distillation](https://www.canvas.inc/research/privileged-self-distillation)**.

PSD turns failed model rollouts into verified local training targets. It finds a
recoverable mistake, constructs a short hint that fixes it, verifies the repair
in the environment, and distills the corrected behavior into a policy that
never sees the hint.

## Method

1. **Roll out** the current policy and collect failures.
2. **Repair** a failed decision with a hint, retaining only verifier-passing
   continuations.
3. **Distill** the hinted teacher distribution into the unhinted policy at the
   repaired decision.

The experiments use BFCL multi-turn tool use. The published Qwen3.5-9B study
trains through Tinker; the larger Qwen3.6-35B study trains through River.
Tinker and River are infrastructure backends for the same method.

## Results

Agentic PSD repairs 141 of 183 all-fail training tasks. Three PSD rounds reach
118.0/240 on the held-out suite, compared with 111.3 for GRPO. One PSD round
followed by two GRPO rounds reaches 146.5.

![Held-out BFCL performance across three training rounds](results/figures/psd_campaign.svg)

See [`RESULTS.md`](RESULTS.md) for the full scorecard and provenance.

## Research artifacts

- **Dataset:** [essamsleiman/psd-bfcl](https://huggingface.co/datasets/essamsleiman/psd-bfcl)
  contains verified repairs, preservation targets, top-20 soft targets, rollout
  artifacts, and 264 sanitized before/search/after repair traces.
- **Repair browser:** [essamsleiman/psd-repair-browser](https://huggingface.co/spaces/essamsleiman/psd-repair-browser)
  provides an interactive view of all 375 cleaned River-native repairs,
  including 423 verifier-guided reruns and the final passing trajectories.

## Installation

Python 3.12 is required.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[tinker,agentic]"  # Qwen3.5-9B
# or
.venv/bin/pip install -e ".[river]"          # Qwen3.6-35B

cp .env.example .env
```

Add the API key for the backend you are using to `.env`.

## Usage

```bash
# Evaluate a model on BFCL
python -m psd --runner direct --benchmark bfcl --limit 1

# Inspect rollout collection and GRPO options
python -m psd.training.grpo --help

# Inspect PSD training options
python -m psd.training.turn_kl --help
python -m psd.training.river_psd --help
```

Exact experiment settings are recorded in [`experiments/bfcl/configs/`](experiments/bfcl/configs/).

Raw `results/runs/` outputs and rollout banks remain gitignored. Small run summaries
and result provenance are tracked here; curated training artifacts and
sanitized repair traces live in the Hugging Face dataset above. Raw Claude
streams and agentic execution logs are intentionally omitted.

## Repository layout

```text
psd/                 # installable method package
  backends/          # Tinker / River model backends
  evaluation/        # BFCL adapter and eval runners
  repair/            # turn recovery, canonical repair, hint toolbox
  targets/           # OPD / preservation / top-k target builders
  training/          # PSD turn-KL, GRPO, hybrid, River PSD
experiments/bfcl/    # configs, splits, harnesses, drivers, agentic CLIs
results/             # figures, score provenance, frozen run summaries
docs/                # implementation notes
release/             # HF dataset metadata and repair browser source
scripts/             # validation and release utilities
```

## Citation

```bibtex
@misc{sleiman2026psd,
  author = {Essam Sleiman},
  title = {Privileged Self-Distillation},
  year = {2026},
  note = {https://www.canvas.inc/research/privileged-self-distillation},
}
```

## License

Code is released under the [MIT License](LICENSE). Dataset artifacts use the
license declared on their Hugging Face dataset card.
