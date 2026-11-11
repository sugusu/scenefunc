#!/usr/bin/env python3
"""Agentic verified-repair harvest for River's native Qwen tool harness."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
import river_client as river
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from river_client.renderers import get_renderer

from psd.evaluation.bfcl import (
    NATIVE_TOOL_SYSTEM_PROMPT,
    BFCLAdapter,
)
from psd.paths import REPO_ROOT
from experiments.bfcl.repair.slate import (
    ANALYZER_RULES,
    attempt_history_block,
    call_analyzer,
    describe_rollout,
    normalize_slate,
    verdict_line,
)

DEFAULT_BANK = Path("results/river_qwen36_native_r1_full360_g8_rollouts.jsonl.gz")
DEFAULT_OUT = Path("results/runs/psd/river_native_repairs")
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


def load_env(path: Path) -> None:
    for raw in path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def iter_groups(path: Path):
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt") if path.suffix == ".gz" else opener("r") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_all_fail_tasks(path: Path, per_category: int) -> list[str]:
    selected: dict[str, list[str]] = {
        "multi_turn_base": [],
        "multi_turn_miss_func": [],
        "multi_turn_miss_param": [],
    }
    for group in iter_groups(path):
        if not all(float(reward) == 0.0 for reward in group["rewards"]):
            continue
        category = str(group["task_id"]).rsplit("_", 1)[0]
        if category in selected and len(selected[category]) < per_category:
            selected[category].append(str(group["task_id"]))
        if all(len(task_ids) >= per_category for task_ids in selected.values()):
            break
    return [task_id for category in selected.values() for task_id in category]


def load_failed_rollout(path: Path, task_id: str) -> dict[str, Any]:
    for group in iter_groups(path):
        if group["task_id"] != task_id:
            continue
        failed = [rollout for rollout in group["rollouts"] if rollout["reward"] == 0.0]
        if not failed:
            raise ValueError(f"{task_id} has no failed rollouts")
        return max(failed, key=lambda rollout: rollout["n_steps"])
    raise ValueError(f"{task_id} not found in {path}")


def decoded_seed_transcript(renderer: Any, task: Any, rollout: dict[str, Any]) -> str:
    raw_questions = task.payload["question"]
    lines: list[str] = []
    turn_index = -1
    for step in rollout["steps"]:
        prompt = renderer.tokenizer.decode(
            step["prompt_ids"],
            skip_special_tokens=False,
        )
        users = prompt.count("<|im_start|>user\n")
        tool_responses = prompt.count("<tool_response>")
        inferred_turn = users - tool_responses - 1
        if inferred_turn != turn_index:
            turn_index = inferred_turn
            user_text = ""
            if 0 <= turn_index < len(raw_questions):
                user_text = " ".join(
                    str(message.get("content", ""))
                    for message in raw_questions[turn_index]
                )
            lines.append(f"[Turn {turn_index}] user: {user_text}")
        completion = renderer.tokenizer.decode(
            step["completion_ids"],
            skip_special_tokens=True,
        )
        lines.append(f"  assistant: {completion[:1200]}")
    return "\n".join(lines)[-50000:]


def incremental_first_bad(
    task: Any,
    entry: dict[str, Any],
    category: str,
    turns: list[list[list[str]]],
    namespace: str,
) -> tuple[int | None, dict[str, Any] | None]:
    for count in range(1, len(turns) + 1):
        score = multi_turn_checker(
            turns[:count],
            task.ground_truth[:count],
            copy.deepcopy(entry),
            category,
            f"{namespace}_prefix_{count}",
        )
        if not score.get("valid"):
            return count - 1, {
                key: score.get(key)
                for key in ("valid", "error_type", "error_message")
            }
    return None, None


def run_native_slate(
    *,
    task: Any,
    slate: dict[int, str],
    attempt: int,
    renderer: Any,
    client: river.Client,
    temperature: float,
    max_tokens: int,
    step_limit: int,
) -> dict[str, Any]:
    adapter = BFCLAdapter()
    entry = copy.deepcopy(task.payload)
    category = task.id.rsplit("_", 1)[0]
    functions = list(entry["function"])
    tools = adapter._compile_native_tools(functions)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": NATIVE_TOOL_SYSTEM_PROMPT}
    ]
    all_turns: list[list[list[str]]] = []
    turns_log: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    namespace = f"river_repair_{task.id}_a{attempt}_{time.time_ns()}"

    for turn_index, source_user_messages in enumerate(entry["question"]):
        user_messages = source_user_messages
        if str(turn_index) in entry.get("missed_function", {}):
            functions.extend(entry["missed_function"][str(turn_index)])
            tools = adapter._compile_native_tools(functions)
            user_messages = [
                {
                    "role": "user",
                    "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                }
            ]

        turn_entry_messages = [*messages, *(dict(message) for message in user_messages)]
        sampling_messages = copy.deepcopy(turn_entry_messages)
        hint = slate.get(turn_index)
        if hint:
            sampling_messages.append({"role": "user", "content": hint})
        canonical_additions: list[dict[str, Any]] = []
        decoded_steps: list[list[str]] = []
        steps: list[dict[str, Any]] = []
        force_quit = False

        for step_index in range(step_limit + 1):
            prompt = renderer.build_prompt_str(sampling_messages, tools=tools)
            sample = client.sample(
                prompt,
                base_model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=renderer.get_stop_strings(),
                tokenizer=renderer.tokenizer,
            )[0]
            parsed = renderer.parse_response(sample.text)
            message = parsed.message
            tool_calls = message.get("tool_calls", [])
            sampling_messages.append(message)
            canonical_additions.append(message)
            step: dict[str, Any] = {
                "step_index": step_index,
                "model_text": sample.text,
                "decoded_calls": [],
                "observations": [],
            }
            if hint and step_index == 0:
                repair_rows.append(
                    {
                        "task_id": task.id,
                        "turn_index": turn_index,
                        "hint": hint,
                        "student_messages": turn_entry_messages,
                        "teacher_messages": [
                            *turn_entry_messages,
                            {"role": "user", "content": hint},
                        ],
                        "target_message": message,
                        "tools": tools,
                        "completion_ids": list(sample.tokens),
                        "sample_logprobs": list(sample.logprobs),
                    }
                )
            if not tool_calls:
                steps.append(step)
                break

            decoded_calls: list[str] = []
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                raw_arguments = tool_call["function"].get("arguments") or "{}"
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
                arguments = adapter._coerce_native_arguments(
                    function_name,
                    arguments,
                    tools,
                )
                decoded_calls.append(
                    adapter._format_native_call(function_name, arguments)
                )
            observations, _ = execute_multi_turn_func_call(
                decoded_calls,
                entry.get("initial_config", {}),
                entry["involved_classes"],
                namespace,
                task.id,
                long_context=False,
                is_evaL_run=False,
            )
            step["decoded_calls"] = decoded_calls
            step["observations"] = observations
            decoded_steps.append(decoded_calls)
            steps.append(step)
            for observation in observations:
                tool_message = {
                    "role": "tool",
                    "content": str(observation),
                    "tool_call_id": "",
                }
                sampling_messages.append(tool_message)
                canonical_additions.append(tool_message)
        else:
            force_quit = True

        messages = [*turn_entry_messages, *canonical_additions]
        all_turns.append(decoded_steps)
        turns_log.append(
            {
                "turn_idx": turn_index,
                "user_messages": user_messages,
                "hint": hint,
                "force_quit": force_quit,
                "steps": steps,
            }
        )
        if force_quit:
            break

    if len(all_turns) == len(task.ground_truth):
        score = multi_turn_checker(
            all_turns,
            task.ground_truth,
            copy.deepcopy(entry),
            category,
            f"{namespace}_final",
        )
    else:
        score = {"valid": False, "error_type": "turn_count_mismatch"}
    task_valid = bool(score.get("valid"))
    first_bad, first_bad_score = (None, None)
    if not task_valid:
        first_bad, first_bad_score = incremental_first_bad(
            task,
            entry,
            category,
            all_turns,
            namespace,
        )
    return {
        "task_valid": task_valid,
        "checker": {
            key: score.get(key)
            for key in ("valid", "error_type", "error_message")
        },
        "first_bad_turn": first_bad,
        "first_bad_score": first_bad_score,
        "turns": turns_log,
        "all_turns_decoded": all_turns,
        "raw_questions": entry["question"],
        "repair_rows": repair_rows if task_valid else [],
    }


def harvest_task(
    *,
    task_id: str,
    bank: Path,
    attempts: int,
    max_hints: int,
    temperature: float,
    max_tokens: int,
    step_limit: int,
    analyzer_model: str,
    renderer: Any,
    out_dir: Path,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    adapter = BFCLAdapter()
    task = adapter.load_tasks(
        category=task_id.rsplit("_", 1)[0],
        limit=1,
        offset=0,
        task_ids=[task_id],
    )[0]
    rollout = load_failed_rollout(bank, task_id)
    transcript = decoded_seed_transcript(renderer, task, rollout)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    client = river.Client(api_key=os.environ["RIVER_API_KEY"])
    history: list[dict[str, Any]] = []
    converged: dict[str, Any] | None = None
    task_dir = out_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(1, attempts + 1):
            prompt_parts = [
                ANALYZER_RULES.format(max_hints=max_hints),
                (
                    f"The task has {len(task.payload['question'])} user turns "
                    f"(indices 0..{len(task.payload['question']) - 1})."
                ),
            ]
            if history:
                prompt_parts.append(attempt_history_block(history))
            prompt_parts.extend(
                [
                    "THE FAILED EPISODE:\n\n" + transcript,
                    (
                        history[-1]["verdict"]
                        if history
                        else "Checker verdict: the full task failed."
                    ),
                ]
            )
            analyzer = call_analyzer(
                bedrock,
                analyzer_model,
                "\n\n".join(prompt_parts),
            )
            parsed = analyzer.get("parsed") or {}
            proposed_slate = normalize_slate(
                parsed,
                n_turns=len(task.payload["question"]),
                max_hints=max_hints,
            )
            tool_names = {
                function["name"].lower() for function in task.payload["function"]
            }
            slate = {
                turn: hint
                for turn, hint in proposed_slate.items()
                if not any(name in hint.lower() for name in tool_names)
            }
            rejected = {
                turn: hint
                for turn, hint in proposed_slate.items()
                if turn not in slate
            }
            if rejected:
                rewrite_prompt = (
                    "\n\n".join(prompt_parts)
                    + "\n\nYour proposed hints were rejected because they named exact "
                    "function/tool identifiers:\n"
                    + json.dumps(rejected)
                    + "\nRewrite them as procedural strategy hints without any exact "
                    "tool names, function names, argument values, IDs, or code. Return "
                    "only the required JSON object."
                )
                rewritten = call_analyzer(
                    bedrock,
                    analyzer_model,
                    rewrite_prompt,
                )
                rewritten_parsed = rewritten.get("parsed") or {}
                rewritten_slate = normalize_slate(
                    rewritten_parsed,
                    n_turns=len(task.payload["question"]),
                    max_hints=max_hints,
                )
                slate = {
                    turn: hint
                    for turn, hint in rewritten_slate.items()
                    if not any(name in hint.lower() for name in tool_names)
                }
                analyzer = {
                    "initial": analyzer,
                    "rewrite": rewritten,
                    "rejected_hints": rejected,
                }
            result = run_native_slate(
                task=task,
                slate=slate,
                attempt=attempt,
                renderer=renderer,
                client=client,
                temperature=temperature,
                max_tokens=max_tokens,
                step_limit=step_limit,
            )
            verdict = (
                "Checker verdict: FULL TASK PASSED."
                if result["task_valid"]
                else verdict_line(
                    result["first_bad_turn"],
                    result["first_bad_score"] or result["checker"],
                )
            )
            record = {
                "attempt": attempt,
                "analyzer": analyzer,
                "slate": slate,
                "verdict": verdict,
                **result,
            }
            (task_dir / f"attempt_{attempt:02d}.json").write_text(
                json.dumps(record, indent=2, default=str)
            )
            if result["task_valid"]:
                converged = record
                break
            history.append(
                {"attempt": attempt, "slate": slate, "verdict": verdict}
            )
            transcript = describe_rollout(
                result["raw_questions"],
                result["turns"],
            )
    finally:
        client.close()
    summary = {
        "task_id": task_id,
        "converged": bool(converged),
        "attempts_used": converged["attempt"] if converged else attempts,
        "final_slate": converged["slate"] if converged else None,
        "repair_rows": converged["repair_rows"] if converged else [],
    }
    (task_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with print_lock:
        print(
            f"REPAIR_TASK_DONE {task_id} converged={summary['converged']} "
            f"attempts={summary['attempts_used']} rows={len(summary['repair_rows'])}",
            flush=True,
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=str(DEFAULT_BANK))
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-hints", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--step-limit", type=int, default=20)
    parser.add_argument(
        "--analyzer-model",
        default="global.anthropic.claude-opus-4-8",
    )
    parser.add_argument("--run-name", default="canary12")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    load_env(REPO_ROOT / args.env_file)
    bank = REPO_ROOT / args.bank
    out_dir = REPO_ROOT / args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = args.task_ids or select_all_fail_tasks(bank, args.per_category)
    (out_dir / "task_ids.json").write_text(json.dumps(task_ids, indent=2))
    renderer = get_renderer(MODEL, thinking=False)
    print_lock = threading.Lock()
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                harvest_task,
                task_id=task_id,
                bank=bank,
                attempts=args.attempts,
                max_hints=args.max_hints,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                step_limit=args.step_limit,
                analyzer_model=args.analyzer_model,
                renderer=renderer,
                out_dir=out_dir,
                print_lock=print_lock,
            )
            for task_id in task_ids
        ]
        for future in as_completed(futures):
            summaries.append(future.result())
    summaries.sort(key=lambda summary: task_ids.index(summary["task_id"]))
    passed = sum(bool(summary["converged"]) for summary in summaries)
    rows = [
        row
        for summary in summaries
        for row in summary.get("repair_rows", [])
    ]
    result = {
        "model": MODEL,
        "analyzer_model": args.analyzer_model,
        "n_tasks": len(summaries),
        "converged": passed,
        "repair_rate": passed / len(summaries),
        "n_repair_rows": len(rows),
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "repair_rows.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows)
    )
    print(
        f"RIVER_REPAIR_CANARY_DONE passed={passed}/{len(summaries)} "
        f"rows={len(rows)} output={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
