#!/usr/bin/env python3
"""Leak audit for slate repair rows (same standard as the round-1 audit).

For every repair row, compare the hint against the task's ground-truth calls:
  - candidate name leak: a ground-truth function name appears in the hint
  - candidate value leak: a ground-truth argument value appears in the hint
  - TRUE leak: the matched string also does NOT appear anywhere in the
    student prompt (episode context), i.e. it is not derivable from what the
    student sees — strictest standard; those rows are dropped.

Usage:
    python -m experiments.bfcl.agentic.leak_audit \
        --rows results/runs/psd/agentic_repair/slate_rows_ur1.jsonl \
        --out-clean results/runs/psd/agentic_repair/slate_rows_ur1_final.jsonl \
        --report results/runs/psd/agentic_repair/slate_rows_ur1_leak_audit.json
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from experiments.bfcl.repair.pilot_arms import build_task  # noqa: E402

CALL_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
STRING_VAL_RE = re.compile(r"['\"]([^'\"]{3,})['\"]")
COMMON = {
    "the", "and", "for", "with", "file", "files", "true", "false", "none",
    "path", "name", "text", "user", "list", "value", "values", "content",
}


def gt_strings(task_id: str) -> tuple[set[str], set[str]]:
    """Function names and argument values from the task's ground-truth calls."""
    gt = build_task(task_id).ground_truth
    names: set[str] = set()
    values: set[str] = set()
    for turn_calls in gt:
        for call in turn_calls:
            call = str(call)
            for m in CALL_NAME_RE.finditer(call):
                n = m.group(1).lower()
                if len(n) >= 3 and n not in COMMON:
                    names.add(n)
            for m in STRING_VAL_RE.finditer(call):
                v = m.group(1).strip().lower()
                if len(v) >= 3 and v not in COMMON and not v.isdigit():
                    values.add(v)
    return names, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out-clean", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    gt_cache: dict[str, tuple[set[str], set[str]]] = {}
    name_leaks, value_leaks, true_leaks = [], [], []
    clean: list[dict] = []

    for row in rows:
        if row.get("hint") is None:  # pass-imitation rows carry no hint
            clean.append(row)
            continue
        tid = row["task_id"]
        if tid not in gt_cache:
            gt_cache[tid] = gt_strings(tid)
        names, values = gt_cache[tid]
        hint = str(row["hint"]).lower()
        if row.get("student_prompt") is not None:
            student = str(row["student_prompt"]).lower()
        else:
            # Native rows keep structured messages and tool schemas instead of
            # a pre-rendered prompt. Tool names are visible to the student, so
            # include both structures when deciding whether a match is
            # derivable from its context.
            student = json.dumps(
                {
                    "messages": row.get("student_messages") or [],
                    "tools": row.get("tools") or [],
                },
                ensure_ascii=False,
                default=str,
            ).lower()
        turn_idx = row.get("turn_idx", row.get("turn_index"))
        key = [tid, turn_idx]

        hit_names = [n for n in names if n in hint]
        hit_values = [v for v in values if v in hint]
        if hit_names:
            name_leaks.append(key)
        if hit_values:
            value_leaks.append(key)
        # True leak: hint content matching ground truth that the student's own
        # context does not contain anywhere.
        underivable = [s for s in (hit_names + hit_values) if s not in student]
        if underivable:
            true_leaks.append([tid, turn_idx, underivable])
        else:
            clean.append(row)

    Path(args.report).write_text(json.dumps({
        "n_rows": len(rows),
        "name_leak_keys": name_leaks,
        "value_leak_keys": value_leaks,
        "true_leaks": true_leaks,
        "n_dropped": len(rows) - len(clean),
    }, indent=1))
    with Path(args.out_clean).open("w") as f:
        for row in clean:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"rows in: {len(rows)}")
    print(f"candidate name-leak sites: {len(name_leaks)} | value-leak sites: {len(value_leaks)}")
    print(f"TRUE leaks (dropped): {len(true_leaks)}")
    for t in true_leaks:
        print(f"  {t[0]} t{t[1]}: {t[2]}")
    print(f"clean rows out: {len(clean)} -> {args.out_clean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
