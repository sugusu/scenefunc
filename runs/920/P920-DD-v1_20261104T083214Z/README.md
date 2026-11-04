# P920-DD-v1 on 920

This run tests whether measured functional-use feedback improves one revision over geometry feedback from the same unrevised A/B drafts. The baseline is the merged AutoDL delivery `bdabc219772112ff2aee0b4a14ffd86809f48cb1`; see `BASELINE.json`. Historical AutoDL measurements remain reference data, not 920 results.

Current machine root: `/data/wyh/scene_cvpr_20261020/design_decisions_920`.
The authoritative portable progress files are `runs/920/latest.json` and this directory's `STATUS.json`.

The exact model revision and all four weight shards are verified. The original requirement card hash matches. Native first-round inputs encode to A=2237 and B=2241 tokens without truncation, matching the handoff. The engineering generation has a 32-token cap and executes no tools; it cannot establish 2048-token or five-view capacity, furnish a room, or answer the G/U research question.

## Resource recovery and actual commands

Commands run from this repository; set the machine root explicitly:

```bash
export P920_RUN_ROOT=/data/wyh/scene_cvpr_20261020/design_decisions_920
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Weight provenance and approved 12-file allowlist are in `evidence/MODEL_DOWNLOAD_PLAN.json`; verified full-file hashes and resumed transfer evidence are in `evidence/MODEL_DOWNLOAD_RECEIPT.json`. The repository's original 11-file weight lock also matches; the approved extra file is the 385-byte video preprocessor config. The actual resumable downloader is `scripts/920/download_fixed_model.py`; it expects the plan beside the script in the machine root. No second weight cache is required. Existing complete files are checked and reused, partial files resume, and the fixed revision never changes.

The 25 necessary upstream files are fetched using:

```bash
python3 scripts/920/fetch_upstream.py \
  --lock runs/920/P920-DD-v1_20261104T083214Z/upstream.920.lock.json \
  --data "$P920_RUN_ROOT/resources/scenefunc" \
  --receipt "$P920_RUN_ROOT/preflight/UPSTREAM_DOWNLOAD_920.json"
```

One official source file differs from the AutoDL lock. Both independent GitHub reads agree on the actual fixed-commit bytes. The original lock is retained, the explicit 920 lock is separate, and `evidence/UPSTREAM_LOCK_CORRECTION.json` records the discrepancy. Historical executor byte equivalence is not claimed.

The runtime was created by `paris/scripts/materialize.py` with `--run "$P920_RUN_ROOT/runtime/P920-DD-v1_20261104T083214Z"`, `--data "$P920_RUN_ROOT/resources/scenefunc"`, model/render Python `/data/wyh/conda/envs/c15_genrec/bin/python`, and geometry Python `$P920_RUN_ROOT/envs/geometry/bin/python`. Its source hashes and resolved prompt mapping remain in the local `run_config.json`; upstream prompt text is reconstructed from the pinned provider files, not copied into this run's Git evidence. The render interpreter is a placeholder until integration with the existing Blender executable is verified.

CPU model validation and the bounded guarded forward:

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/validate_model_cpu.py
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/model_forward_preflight.py
```

The GPU guard uses the existing shared GPU0 lock plus an experiment lock, checks the authorized UUID, rejects occupied cards, and records start time, PID/start ticks, log and actual child exit. The original V100 26-GiB guard remains unchanged. The 920 engineering guard covers only this exact input and 32 output tokens, with an explicit weight/KV/workspace estimate. It does not authorize a full experiment round. Prior attempts are never overwritten.

## Remaining experiment gate

HSSD returned HTTP 401 `GatedRepo`. The user must accept the provider conditions using their own account and configure a read token locally. Only then can the 23 selected real meshes plus coordinate files (62,522,166 bytes total) be downloaded and checked. Tokens, weights, meshes, and full checkpoints are excluded from Git.

After access is available: reconstruct both historical **unrevised** drafts, verify source meshes, object identities/transforms and fresh geometric audits, then validate the five-view budget. Only trusted reconstructions qualify for four same-start G/U revision calls. Otherwise record why new A/B drafts are required. No finished scene, four-cell use score, training gain, or design improvement is claimed by this preflight.
