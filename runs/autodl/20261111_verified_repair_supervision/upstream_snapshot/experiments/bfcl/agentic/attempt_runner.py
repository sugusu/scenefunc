#!/usr/bin/env python3
"""Backend for the sandbox `run_attempt` shim.

Runs one verified slate rollout in the real repo, then copies a sanitized
artifact (no token ids / logprobs) into the sandbox as attempt_NN.json and
prints a compact verdict for the agent.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from psd.paths import REPO_ROOT


def sanitize_step(step: dict) -> dict:
    return {
        k: v
        for k, v in step.items()
        if k not in ("sampled_token_ids", "sampled_logprobs", "latency_sec", "prompt_chars")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-path", help="tinker:// sampler for the actor (round >= 2).")
    parser.add_argument(
        "--sampling-backend",
        choices=["tinker", "river"],
        default="tinker",
    )
    parser.add_argument("--model")
    parser.add_argument("--rollouts", help="Collection jsonl the failed episode comes from.")
    args = parser.parse_args()
    sandbox = Path(args.sandbox)

    cmd = [
        sys.executable, "-m", "experiments.bfcl.repair.slate",
        "--task-id", args.task_id,
        "--run-name", args.run_name,
        "--manual-slate", str(sandbox / "slate.json"),
        "--sampling-backend", args.sampling_backend,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if args.rollouts:
        cmd += ["--rollouts", args.rollouts]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (sandbox / ".last_run.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)

    run_dir = REPO_ROOT / "results/runs/psd/pilot_slate" / args.run_name
    artifacts = sorted(run_dir.glob("manual_*.json")) if run_dir.exists() else []
    if not artifacts:
        print("ERROR: rollout produced no artifact; see .last_run.log")
        return 2
    record = json.loads(artifacts[-1].read_text())

    n_prev = len(list(sandbox.glob("attempt_*.json")))
    out = {
        "attempt": n_prev + 1,
        "slate": record.get("slate"),
        "task_valid": record.get("task_valid"),
        "first_bad_turn": record.get("first_bad_turn"),
        "first_bad_score": record.get("first_bad_score"),
        "checker": record.get("checker"),
        "turns": [
            {
                "turn_idx": t.get("turn_idx"),
                "user_messages": t.get("user_messages"),
                "hinted": bool(t.get("hint")),
                "force_quit": t.get("force_quit"),
                "steps": [sanitize_step(s) for s in t.get("steps", [])],
            }
            for t in record.get("turns", [])
        ],
    }
    dest = sandbox / f"attempt_{n_prev + 1:02d}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))

    if record.get("task_valid"):
        print(f"VERDICT: FULL TASK PASSED. Artifact: {dest.name}")
    else:
        score = record.get("first_bad_score") or record.get("checker") or {}
        print(
            f"VERDICT: FAILED. First invalid turn = {record.get('first_bad_turn')} "
            f"({score.get('error_type')}): {str(score.get('error_message'))[:220]} "
            f"Artifact: {dest.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
