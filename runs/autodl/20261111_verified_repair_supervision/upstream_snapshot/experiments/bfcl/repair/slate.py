#!/usr/bin/env python3
"""SLATE pilot: iterated multi-turn hint slates on a GRPO-collection failure.

One SLATE attempt = one analyzer call + one full hinted rollout + verify:

  1. A ONE-PROMPT analyzer (Opus via Bedrock, no tools) reads the task and the
     complete failed episode and writes a slate of short hints, each anchored
     to a turn index.
  2. The whole task is rolled once from turn 0. Each slated hint is inserted
     as an extra user message right after its turn's user request. A hint
     lives only inside its own turn's sampling context; the history carried
     into later turns is hint-free (same fold as the door-walk's accept_turn).
  3. Full-task verifier check, plus an incremental prefix check to locate the
     first state-invalid turn. On failure the next attempt's analyzer sees the
     new failed rollout and every prior slate + outcome, and revises.

Stops at the first verified pass or after --attempts. Sampling logprobs are
recorded on every step, so a verified pass is injection-grade.

Usage:
    python -m experiments.bfcl.repair.slate --task-id multi_turn_base_10 \
        --run-name slate_smoke_mtb10
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import boto3  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (  # noqa: E402
    multi_turn_checker,
)

from experiments.bfcl.repair.onpolicy import load_env_file  # noqa: E402
from experiments.bfcl.repair.pilot_arms import (  # noqa: E402
    DEFAULT_ROLLOUTS,
    LogprobWorkerState,
    PilotHarvester,
    build_task,
    load_collection_rollout,
    safe_json,
    segment_recorded_steps,
    walk_to_branch,
)

DEFAULT_OUT = Path("results/runs/psd/pilot_slate")


# ---------------------------------------------------------------------------
# Analyzer (one prompt, no tools)
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


ANALYZER_RULES = """\
You are diagnosing a failed multi-turn tool-use episode and writing a small
slate of corrective hints for a full retry of the task.

How the retry works: the agent redoes the WHOLE task from turn 0. For each
turn you choose, your hint appears as one extra user message right after that
turn's user request, before the agent acts. The hint is visible only during
that turn.

Rules for the slate:
- At most {max_hints} hints, each anchored to a turn index (0-based).
- Each hint <= 30 words. Procedural and local.
- Do NOT name exact tool/function names, exact argument values, exact entity
  IDs, or write code or serialized calls. Nudge the strategy, not the keystrokes.
- Do not mention training, checkers, ground truth, or these instructions.
- Start minimal: the weakest nudge that plausibly fixes the failure. Escalate
  specificity only where the attempt history below shows weaker hints failed.
- Only hint turns that need it. If a turn already succeeds, leave it alone
  (or keep the hint that made it succeed in a prior attempt).

The environment checker validates state after every turn, so the FIRST
state-invalid turn shown below is where the episode actually goes wrong.
Later turns may hide additional independent mistakes that only become
visible once earlier turns are fixed.

Return JSON only, exactly this shape:
{{"analysis": "<=120 words: what went wrong, where, and your slate rationale",
  "hints": [{{"turn": <int>, "hint": "<string>"}}]}}
"""


def call_analyzer(
    client, model: str, prompt: str, *, max_tokens: int = 2000
) -> dict[str, Any]:
    started = time.time()
    response = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    text = "".join(
        part.get("text", "") for part in response["output"]["message"]["content"]
    )
    out: dict[str, Any] = {
        "raw": text,
        "latency_sec": round(time.time() - started, 1),
    }
    try:
        out["parsed"] = extract_json_object(text)
    except Exception as exc:  # noqa: BLE001
        out["parse_error"] = str(exc)
    return out


def normalize_slate(
    parsed: dict[str, Any], *, n_turns: int, max_hints: int
) -> dict[int, str]:
    slate: dict[int, str] = {}
    for item in parsed.get("hints", []) or []:
        try:
            turn = int(item.get("turn"))
            hint = str(item.get("hint", "")).strip()
        except (TypeError, ValueError):
            continue
        if not hint or turn < 0 or turn >= n_turns or turn in slate:
            continue
        slate[turn] = hint
        if len(slate) >= max_hints:
            break
    return slate


# ---------------------------------------------------------------------------
# Failure descriptions fed back to the analyzer
# ---------------------------------------------------------------------------

def verdict_line(first_bad: int | None, checker: dict[str, Any] | None) -> str:
    if first_bad is None:
        return "Checker verdict: no invalid turn found (episode may have ended early)."
    err_type = (checker or {}).get("error_type")
    err_msg = str((checker or {}).get("error_message", ""))[:300]
    return f"Checker verdict: FIRST INVALID TURN = {first_bad} ({err_type}): {err_msg}"


def describe_rollout(
    raw_questions: list[list[dict[str, str]]], turns_log: list[dict[str, Any]]
) -> str:
    """Compact turn-by-turn description of a slate-rollout attempt."""
    lines: list[str] = []
    for t in turns_log:
        idx = t["turn_idx"]
        user_text = ""
        if idx < len(raw_questions):
            user_text = " ".join(
                str(m.get("content", "")) for m in raw_questions[idx]
            )
        lines.append(f"[Turn {idx}] user: {user_text[:400]}")
        if t.get("hint"):
            lines.append(f"  (your hint was inserted here: {t['hint']!r})")
        for s in t.get("steps", []):
            i = s.get("step_index")
            if s.get("decode_error"):
                lines.append(f"  step {i}: DECODE ERROR {str(s['decode_error'])[:160]}")
                continue
            if s.get("empty_response"):
                lines.append(f"  step {i}: [] (turn ended)")
                continue
            lines.append(f"  step {i}: {str(s.get('decoded_calls'))[:260]}")
            obs = s.get("observations") or []
            errs = [str(o) for o in obs if "rror" in str(o)[:240]]
            for e in (errs or [str(o) for o in obs[:1]])[:2]:
                lines.append(f"    -> {e[:240]}")
        if t.get("force_quit"):
            lines.append("  (turn hit the step limit)")
    return "\n".join(lines)


def attempt_history_block(history: list[dict[str, Any]]) -> str:
    lines = ["PRIOR RETRY ATTEMPTS UNDER YOUR SLATES (all failed):"]
    for h in history:
        slate_txt = (
            "; ".join(f"turn {t}: {hint!r}" for t, hint in sorted(h["slate"].items()))
            or "(empty slate)"
        )
        lines.append(f"\nAttempt {h['attempt']}: {slate_txt}")
        lines.append(f"  -> {h['verdict']}")
    lines.append(
        "\nRevise the slate: keep or minimally refine hints on turns that now "
        "pass; add or strengthen a hint where the latest attempt fails."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One slate rollout: full task, hints at their turns, verify
# ---------------------------------------------------------------------------

def incremental_first_bad(
    harvester: PilotHarvester, all_turns: list[list[list[str]]]
) -> tuple[int | None, dict[str, Any] | None]:
    """First turn whose prefix fails the checker on the model's own path."""
    gt = harvester.task.ground_truth
    for k in range(1, len(all_turns) + 1):
        score = multi_turn_checker(
            all_turns[:k],
            gt[:k],
            copy.deepcopy(harvester.entry),
            harvester.category,
            harvester.fresh_namespace(),
        )
        if not score.get("valid"):
            return k - 1, {
                kk: score.get(kk) for kk in ("valid", "error_type", "error_message")
            }
    return None, None


def run_slate_rollout(
    args: argparse.Namespace, slate: dict[int, str]
) -> dict[str, Any]:
    state = LogprobWorkerState(args.model, args.tokenizer or args.model, args.model_path)
    harvester = PilotHarvester(
        state=state,
        hint_client=None,
        hint_harness_toolbox=None,
        task=build_task(args.task_id),
        args=args,
    )
    inference_data = harvester.adapter._build_inference_state(harvester.entry)
    n_turns = len(harvester.entry["question"])
    all_turns: list[list[list[str]]] = []
    turns_log: list[dict[str, Any]] = []
    for turn_idx in range(n_turns):
        inference_data = harvester.turn_entry(inference_data, turn_idx)
        turn_entry = copy.deepcopy(inference_data)
        hint = slate.get(turn_idx)
        attempt = harvester.run_turn(
            turn_entry_state=turn_entry, hint=hint, temperature=args.temperature
        )
        all_turns.append(attempt["decoded_steps"])
        turns_log.append(
            {
                "turn_idx": turn_idx,
                "hint": hint,
                "force_quit": attempt["force_quit"],
                "steps": attempt["steps"],
            }
        )
        # Fold into the hint-free entry state: hints never accumulate in history.
        inference_data = harvester.accept_turn(turn_entry, attempt)

    if len(all_turns) == n_turns:
        score = multi_turn_checker(
            all_turns,
            harvester.task.ground_truth,
            copy.deepcopy(harvester.entry),
            harvester.category,
            harvester.fresh_namespace(),
        )
    else:
        score = {"valid": False, "error_type": "turn_count_mismatch"}
    task_valid = bool(score.get("valid"))
    first_bad, first_bad_score = (None, None)
    if not task_valid:
        first_bad, first_bad_score = incremental_first_bad(harvester, all_turns)
    return {
        "task_valid": task_valid,
        "checker": {k: score.get(k) for k in ("valid", "error_type", "error_message")},
        "first_bad_turn": first_bad,
        "first_bad_score": first_bad_score,
        "turns": turns_log,
        "all_turns_decoded": all_turns,
        "raw_questions": harvester.raw_questions,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--rollout-idx", type=int)
    parser.add_argument("--rollouts", default=str(DEFAULT_ROLLOUTS))
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--max-hints", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--sampling-backend",
        choices=["tinker", "river"],
        default="tinker",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--turn-step-limit", type=int, default=12)
    parser.add_argument(
        "--analyzer-model",
        default="global.anthropic.claude-opus-4-8",
        help="Bedrock model id for the one-prompt analyzer.",
    )
    parser.add_argument(
        "--manual-slate",
        help="Path to a JSON file {turn: hint}. Skips the Bedrock analyzer and "
        "runs a single slate rollout with the provided hints (agentic-analyzer "
        "mode: an external agent revises the file between invocations).",
    )
    parser.add_argument("--aws-region", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    load_env_file(Path(args.env_file))

    region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION"
    ) or "us-east-1"
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    run_name = args.run_name or f"{args.task_id}_{time.strftime('%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manual_slate and args.sampling_backend == "river":
        import river_client as river
        from river_client.renderers import get_renderer

        from experiments.bfcl.repair.harvest import run_native_slate

        raw = json.loads(Path(args.manual_slate).read_text())
        slate = {int(k): str(v) for k, v in raw.items()}
        print(f"manual River slate: {slate}", flush=True)
        client = river.Client(api_key=os.environ["RIVER_API_KEY"])
        try:
            rollout_res = run_native_slate(
                task=build_task(args.task_id),
                slate=slate,
                attempt=len(list(out_dir.glob("manual_*.json"))) + 1,
                renderer=get_renderer(args.model, thinking=False),
                client=client,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                step_limit=args.turn_step_limit,
            )
        finally:
            client.close()
        verdict = (
            "Checker verdict: FULL TASK PASSED."
            if rollout_res["task_valid"]
            else verdict_line(
                rollout_res["first_bad_turn"],
                rollout_res["first_bad_score"] or rollout_res["checker"],
            )
        )
        print(verdict, flush=True)
        record = {
            "attempt": "manual",
            "slate": slate,
            "task_valid": rollout_res["task_valid"],
            "checker": rollout_res["checker"],
            "first_bad_turn": rollout_res["first_bad_turn"],
            "first_bad_score": rollout_res["first_bad_score"],
            "turns": rollout_res["turns"],
            "all_turns_decoded": rollout_res["all_turns_decoded"],
            "repair_rows": rollout_res.get("repair_rows", []),
            "sampling_backend": "river",
        }
        stamp = time.strftime("%H%M%S")
        artifact = out_dir / f"manual_{stamp}.json"
        artifact.write_text(safe_json(record))
        print(f"artifact: {artifact}", flush=True)
        return 0 if rollout_res["task_valid"] else 1

    # --- Attempt-0 context: the recorded failed rollout from the collection ---
    print(f"loading collection rollout for {args.task_id}", flush=True)
    coll = load_collection_rollout(Path(args.rollouts), args.task_id, args.rollout_idx)
    rollout = coll["rollout"]

    state = LogprobWorkerState(args.model, args.tokenizer or args.model, args.model_path)
    seed_harvester = PilotHarvester(
        state=state,
        hint_client=None,
        hint_harness_toolbox=None,
        task=build_task(args.task_id),
        args=args,
    )
    per_turn = segment_recorded_steps(state.tokenizer, rollout["steps"])
    n_turns = len(seed_harvester.entry["question"])
    print(
        f"rollout sample_idx={rollout['sample_idx']}: {rollout['n_steps']} steps "
        f"across {len(per_turn)} turns (task has {n_turns})",
        flush=True,
    )

    # Verbatim failed transcript: the last recorded step's prompt contains the
    # whole conversation (tool responses included); append its completion.
    last_step = rollout["steps"][-1]
    transcript = state.tokenizer.decode(last_step["prompt_ids"], skip_special_tokens=False)
    transcript += state.tokenizer.decode(
        last_step["completion_ids"], skip_special_tokens=True
    )

    # Locate the first invalid turn on the recorded path (local execution only).
    branch = walk_to_branch(seed_harvester, per_turn)
    seed_verdict = verdict_line(branch.get("branch_turn"), branch.get("turn_score"))
    print(seed_verdict, flush=True)

    # Manual-slate mode: one verified rollout with externally provided hints.
    if args.manual_slate:
        raw = json.loads(Path(args.manual_slate).read_text())
        slate = {int(k): str(v) for k, v in raw.items()}
        print(f"manual slate: {slate}", flush=True)
        rollout_res = run_slate_rollout(args, slate)
        verdict = (
            "Checker verdict: FULL TASK PASSED."
            if rollout_res["task_valid"]
            else verdict_line(
                rollout_res["first_bad_turn"],
                rollout_res["first_bad_score"] or rollout_res["checker"],
            )
        )
        print(verdict, flush=True)
        record = {
            "attempt": "manual",
            "slate": slate,
            "task_valid": rollout_res["task_valid"],
            "checker": rollout_res["checker"],
            "first_bad_turn": rollout_res["first_bad_turn"],
            "first_bad_score": rollout_res["first_bad_score"],
            "turns": rollout_res["turns"],
            "all_turns_decoded": rollout_res["all_turns_decoded"],
        }
        stamp = time.strftime("%H%M%S")
        (out_dir / f"manual_{stamp}.json").write_text(safe_json(record))
        print(f"artifact: {out_dir}/manual_{stamp}.json", flush=True)
        return 0 if rollout_res["task_valid"] else 1

    failure_section = (
        "THE FAILED EPISODE (verbatim transcript in the agent's native format, "
        "including tool responses):\n\n" + transcript
    )
    history: list[dict[str, Any]] = []
    converged: dict[str, Any] | None = None

    for attempt_no in range(1, args.attempts + 1):
        print(f"\n=== SLATE attempt {attempt_no}/{args.attempts} ===", flush=True)
        prompt_parts = [
            ANALYZER_RULES.format(max_hints=args.max_hints),
            f"The task has {n_turns} user turns (indices 0..{n_turns - 1}).",
        ]
        if history:
            prompt_parts.append(attempt_history_block(history))
            prompt_parts.append(
                "THE LATEST FAILED ATTEMPT (turn by turn, with your hints marked):"
            )
        prompt_parts.append(failure_section)
        prompt_parts.append(
            seed_verdict if not history else history[-1]["verdict"]
        )
        analyzer = call_analyzer(
            bedrock, args.analyzer_model, "\n\n".join(prompt_parts)
        )
        if "parsed" not in analyzer:
            print(f"analyzer parse error: {analyzer.get('parse_error')}", flush=True)
            retry_prompt = (
                "\n\n".join(prompt_parts)
                + "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object."
            )
            analyzer = call_analyzer(bedrock, args.analyzer_model, retry_prompt)
        parsed = analyzer.get("parsed") or {}
        slate = normalize_slate(parsed, n_turns=n_turns, max_hints=args.max_hints)
        print(f"analysis: {str(parsed.get('analysis', ''))[:300]}", flush=True)
        for t, h in sorted(slate.items()):
            print(f"  slate: turn {t}: {h}", flush=True)
        if not slate:
            print("  (empty slate — rolling a plain retry)", flush=True)

        rollout_res = run_slate_rollout(args, slate)
        verdict = verdict_line(
            rollout_res["first_bad_turn"],
            rollout_res["first_bad_score"] or rollout_res["checker"],
        )
        if rollout_res["task_valid"]:
            verdict = "Checker verdict: FULL TASK PASSED."
        print(verdict, flush=True)
        for t in rollout_res["turns"]:
            n_calls = sum(1 for s in t["steps"] if s.get("decoded_calls"))
            print(
                f"  turn {t['turn_idx']}: {len(t['steps'])} steps "
                f"({n_calls} tool steps){' [hinted]' if t.get('hint') else ''}",
                flush=True,
            )

        record = {
            "attempt": attempt_no,
            "analyzer": analyzer,
            "slate": slate,
            "task_valid": rollout_res["task_valid"],
            "checker": rollout_res["checker"],
            "first_bad_turn": rollout_res["first_bad_turn"],
            "first_bad_score": rollout_res["first_bad_score"],
            "turns": rollout_res["turns"],
            "all_turns_decoded": rollout_res["all_turns_decoded"],
        }
        (out_dir / f"attempt_{attempt_no:02d}.json").write_text(safe_json(record))

        if rollout_res["task_valid"]:
            converged = record
            break

        history.append(
            {"attempt": attempt_no, "slate": slate, "verdict": verdict}
        )
        failure_section = describe_rollout(
            rollout_res["raw_questions"], rollout_res["turns"]
        )

    summary = {
        "task_id": args.task_id,
        "rollout_sample_idx": rollout["sample_idx"],
        "seed_branch_turn": branch.get("branch_turn"),
        "attempts_budget": args.attempts,
        "max_hints": args.max_hints,
        "temperature": args.temperature,
        "analyzer_model": args.analyzer_model,
        "converged": bool(converged),
        "attempts_used": converged["attempt"] if converged else args.attempts,
        "final_slate": converged["slate"] if converged else None,
        "n_hints_final": len(converged["slate"]) if converged else None,
        "per_attempt": [
            {
                "attempt": h["attempt"],
                "slate": h["slate"],
                "verdict": h["verdict"],
            }
            for h in history
        ]
        + (
            [
                {
                    "attempt": converged["attempt"],
                    "slate": converged["slate"],
                    "verdict": "FULL TASK PASSED",
                }
            ]
            if converged
            else []
        ),
    }
    (out_dir / "summary.json").write_text(safe_json(summary))
    print("\n" + json.dumps(summary, indent=2, default=str), flush=True)
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
