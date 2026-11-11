#!/usr/bin/env python3
"""Consolidate verified River-native agentic repairs into PSD rows.

The Claude Code repair harness stores a small sanitized attempt in its sandbox,
but the corresponding ``results/runs/psd/pilot_slate`` artifact retains the native
messages, tools, and first repaired assistant message needed for PSD. This
script uses each run summary as the source of truth, selects one passing
artifact per task, validates its native row structure, and writes JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from psd.paths import REPO_ROOT

DEFAULT_SUMMARY_GLOB = (
    "results/runs/psd/agentic_repair/agentic_cc_*_rivercc*/summary.json"
)
DEFAULT_SCOPE = (
    "results/runs/psd/agentic_repair/river_native_failed_tasks_full.json"
)
DEFAULT_OUT = (
    "results/runs/psd/agentic_repair/river_native_repair_rows.jsonl"
)
DEFAULT_REPORT = (
    "results/runs/psd/agentic_repair/river_native_repair_rows_report.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def artifact_path(summary: dict[str, Any]) -> Path:
    return (
        REPO_ROOT
        / "results/runs/psd/pilot_slate"
        / str(summary["run_name"])
        / str(summary["passed_artifact"])
    )


def validate_native_rows(
    *,
    task_id: str,
    record: dict[str, Any],
    artifact: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not record.get("task_valid") or not (record.get("checker") or {}).get("valid"):
        errors.append(f"{task_id}: selected artifact is not verifier-passing")
    if record.get("sampling_backend") != "river":
        errors.append(f"{task_id}: selected artifact is not River-native")

    slate = {
        int(turn): str(hint)
        for turn, hint in (record.get("slate") or {}).items()
        if str(hint).strip()
    }
    native_rows = record.get("repair_rows") or []
    seen_turns: set[int] = set()
    rows: list[dict[str, Any]] = []

    for index, native in enumerate(native_rows):
        prefix = f"{task_id}: repair_rows[{index}]"
        if native.get("task_id") != task_id:
            errors.append(f"{prefix}: task ID mismatch")
        turn_idx = int(native.get("turn_index", -1))
        if turn_idx in seen_turns:
            errors.append(f"{prefix}: duplicate turn {turn_idx}")
        seen_turns.add(turn_idx)

        hint = str(native.get("hint") or "")
        if not hint or slate.get(turn_idx) != hint:
            errors.append(f"{prefix}: hint does not match passing slate")

        student_messages = native.get("student_messages")
        teacher_messages = native.get("teacher_messages")
        if not isinstance(student_messages, list) or not isinstance(
            teacher_messages, list
        ):
            errors.append(f"{prefix}: missing native student/teacher messages")
            continue
        expected_teacher = [
            *student_messages,
            {"role": "user", "content": hint},
        ]
        if teacher_messages != expected_teacher:
            errors.append(f"{prefix}: teacher context is not student context + hint")

        target = native.get("target_message")
        if not isinstance(target, dict) or target.get("role") != "assistant":
            errors.append(f"{prefix}: malformed assistant target message")
        if not isinstance(native.get("tools"), list) or not native["tools"]:
            errors.append(f"{prefix}: missing native tool schema")

        completion_ids = native.get("completion_ids")
        sample_logprobs = native.get("sample_logprobs")
        if (
            not isinstance(completion_ids, list)
            or not completion_ids
            or not isinstance(sample_logprobs, list)
            or len(completion_ids) != len(sample_logprobs)
        ):
            errors.append(f"{prefix}: invalid completion token/logprob record")
        elif not all(math.isfinite(float(value)) for value in sample_logprobs):
            errors.append(f"{prefix}: non-finite sample logprob")

        rows.append(
            {
                "source": "river_native_agentic_slate",
                "task_id": task_id,
                "turn_idx": turn_idx,
                "accepted": True,
                "verified_full_task": True,
                "hint": hint,
                "hint_provider": "agentic_cc_opus48",
                "student_messages": student_messages,
                "teacher_messages": teacher_messages,
                "target_message": target,
                "tools": native.get("tools"),
                "completion_ids": completion_ids,
                "sample_logprobs": sample_logprobs,
                "ce_one_hot": False,
                "tier": "causal_task_pass",
                "artifact": str(artifact.relative_to(REPO_ROOT)),
            }
        )

    expected_turns = set(slate)
    if seen_turns != expected_turns:
        errors.append(
            f"{task_id}: repaired turns {sorted(seen_turns)} do not match "
            f"passing slate turns {sorted(expected_turns)}"
        )
    if slate and not rows:
        errors.append(f"{task_id}: passing artifact contains no repair rows")
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-glob", default=DEFAULT_SUMMARY_GLOB)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    scope_path = REPO_ROOT / args.scope
    scope = set(load_json(scope_path))
    summaries_by_task: dict[str, list[tuple[float, Path, dict[str, Any]]]] = (
        defaultdict(list)
    )
    passing_by_task: dict[
        str, list[tuple[float, Path, dict[str, Any], Path, dict[str, Any]]]
    ] = defaultdict(list)
    missing_artifacts: list[str] = []

    for summary_path in REPO_ROOT.glob(args.summary_glob):
        summary = load_json(summary_path)
        task_id = str(summary.get("task_id") or "")
        if task_id not in scope:
            continue
        modified = summary_path.stat().st_mtime
        summaries_by_task[task_id].append((modified, summary_path, summary))
        if not summary.get("converted_ground_truth") or not summary.get(
            "passed_artifact"
        ):
            continue
        artifact = artifact_path(summary)
        if not artifact.exists():
            missing_artifacts.append(str(artifact.relative_to(REPO_ROOT)))
            continue
        record = load_json(artifact)
        passing_by_task[task_id].append(
            (modified, summary_path, summary, artifact, record)
        )

    all_rows: list[dict[str, Any]] = []
    selected_runs: dict[str, str] = {}
    empty_slate_passes: list[str] = []
    integrity_errors: list[str] = []
    duplicate_passes: dict[str, int] = {}
    for task_id in sorted(passing_by_task):
        candidates = passing_by_task[task_id]
        if len(candidates) > 1:
            duplicate_passes[task_id] = len(candidates)
        _, _, summary, artifact, record = max(candidates, key=lambda item: item[0])
        rows, errors = validate_native_rows(
            task_id=task_id,
            record=record,
            artifact=artifact,
        )
        selected_runs[task_id] = str(summary["run_name"])
        all_rows.extend(rows)
        integrity_errors.extend(errors)
        if not rows and not (record.get("slate") or {}):
            empty_slate_passes.append(task_id)

    all_rows.sort(key=lambda row: (row["task_id"], row["turn_idx"]))
    duplicate_keys = [
        list(key)
        for key, count in Counter(
            (row["task_id"], row["turn_idx"]) for row in all_rows
        ).items()
        if count > 1
    ]
    if duplicate_keys:
        integrity_errors.append(f"duplicate output task/turn keys: {duplicate_keys}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    verified_passing_tasks = set(selected_runs)
    repaired_tasks = {row["task_id"] for row in all_rows}
    attempted_tasks = set(summaries_by_task)
    report = {
        "scope_path": args.scope,
        "scope_tasks": len(scope),
        "tasks_with_any_summary": len(attempted_tasks),
        "verified_passing_tasks": len(verified_passing_tasks),
        "corrective_repair_tasks": len(repaired_tasks),
        "empty_slate_passes": sorted(empty_slate_passes),
        "unrepaired_tasks": sorted(scope - verified_passing_tasks),
        "tasks_without_correction": sorted(scope - repaired_tasks),
        "repair_rows": len(all_rows),
        "rows_by_category": dict(
            sorted(
                Counter(
                    row["task_id"].rsplit("_", 1)[0] for row in all_rows
                ).items()
            )
        ),
        "duplicate_passing_candidates": duplicate_passes,
        "selected_runs": selected_runs,
        "missing_artifacts": missing_artifacts,
        "integrity_errors": integrity_errors,
        "rows_path": args.out,
    }
    report_path = REPO_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in report.items() if key not in {
        "unrepaired_tasks",
        "duplicate_passing_candidates",
        "selected_runs",
    }}, indent=2))
    print(f"unrepaired task count: {len(report['unrepaired_tasks'])}")
    print(f"report: {report_path}")
    return 1 if (missing_artifacts or integrity_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
