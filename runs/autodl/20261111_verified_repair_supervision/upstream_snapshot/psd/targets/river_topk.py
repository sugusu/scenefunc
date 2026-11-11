#!/usr/bin/env python3
"""Build resumable top-k River PSD targets for repair and preservation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import river_client as river
from river_client.renderers import get_renderer

from psd.paths import REPO_ROOT

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
CONTEXT_WINDOW = 262_144
DEFAULT_REPAIRS = (
    "results/runs/psd/agentic_repair/river_native_repair_rows_clean.jsonl"
)
DEFAULT_PRESERVATION = (
    "results/runs/psd/preservation/river_qwen36_native_base_greedy/"
    "preservation_rows.jsonl"
)
DEFAULT_OUT_DIR = "results/runs/psd/river_native_psd_r1"


@dataclass
class TargetSpec:
    key: str
    kind: str
    task_id: str
    turn_idx: int
    step_idx: int
    row_weight: float
    student_prompt_ids: list[int]
    teacher_input_ids: list[int]
    completion_ids: list[int]

    @property
    def teacher_positions(self) -> list[int]:
        start = len(self.teacher_input_ids) - len(self.completion_ids)
        return list(range(start, len(self.teacher_input_ids)))


def load_env(path: Path) -> None:
    for raw in path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def category(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def select_preservation_rows(
    rows: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    by_category_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_category_task[category(row["task_id"])][row["task_id"]].append(row)
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for cat, quota in sorted(quotas.items()):
        task_rows = by_category_task.get(cat, {})
        for values in task_rows.values():
            values.sort(key=lambda row: int(row["turn_idx"]))
        task_ids = sorted(task_rows)
        rng.shuffle(task_ids)
        cat_selected: list[dict[str, Any]] = []
        depth = 0
        while len(cat_selected) < quota:
            added = 0
            for task_id in task_ids:
                values = task_rows[task_id]
                if depth >= len(values):
                    continue
                cat_selected.append(values[depth])
                added += 1
                if len(cat_selected) == quota:
                    break
            if not added:
                break
            depth += 1
        if len(cat_selected) != quota:
            raise ValueError(
                f"preservation category {cat} has {len(cat_selected)} rows, "
                f"needs {quota}"
            )
        selected.extend(cat_selected)
    return selected


def prepare_repair_target(renderer: Any, row: dict[str, Any]) -> TargetSpec:
    student_prompt = renderer.build_prompt_str(
        row["student_messages"],
        tools=row["tools"],
    )
    teacher_prompt = renderer.build_prompt_str(
        row["teacher_messages"],
        tools=row["tools"],
    )
    student_prompt_ids = renderer.tokenizer.encode(
        student_prompt,
        add_special_tokens=False,
    )
    teacher_prompt_ids = renderer.tokenizer.encode(
        teacher_prompt,
        add_special_tokens=False,
    )
    completion_ids = [int(token) for token in row["completion_ids"]]
    if not completion_ids:
        raise ValueError("empty repair completion")
    if max(
        len(student_prompt_ids) + len(completion_ids),
        len(teacher_prompt_ids) + len(completion_ids),
    ) > CONTEXT_WINDOW:
        raise ValueError("repair sequence exceeds context window")
    task_id = str(row["task_id"])
    turn_idx = int(row["turn_idx"])
    return TargetSpec(
        key=f"repair:{task_id}:t{turn_idx}:s0",
        kind="repair",
        task_id=task_id,
        turn_idx=turn_idx,
        step_idx=0,
        row_weight=1.0,
        student_prompt_ids=student_prompt_ids,
        teacher_input_ids=[*teacher_prompt_ids, *completion_ids],
        completion_ids=completion_ids,
    )


def prepare_preservation_targets(row: dict[str, Any]) -> list[TargetSpec]:
    steps = (row.get("preserve_rollout") or {}).get("steps") or []
    if not steps:
        raise ValueError("preservation row has no steps")
    task_id = str(row["task_id"])
    turn_idx = int(row["turn_idx"])
    row_weight = 1.0 / len(steps)
    targets: list[TargetSpec] = []
    for step in steps:
        step_idx = int(step["step_idx"])
        student_prompt_ids = [
            int(token) for token in step["student_prompt_ids"]
        ]
        teacher_prompt_ids = [
            int(token) for token in step["teacher_prompt_ids"]
        ]
        completion_ids = [int(token) for token in step["completion_ids"]]
        if student_prompt_ids != teacher_prompt_ids:
            raise ValueError("preservation teacher/student prompts differ")
        if not completion_ids:
            raise ValueError("empty preservation completion")
        if len(student_prompt_ids) + len(completion_ids) > CONTEXT_WINDOW:
            raise ValueError("preservation sequence exceeds context window")
        targets.append(
            TargetSpec(
                key=f"preserve:{task_id}:t{turn_idx}:s{step_idx}",
                kind="preserve",
                task_id=task_id,
                turn_idx=turn_idx,
                step_idx=step_idx,
                row_weight=row_weight,
                student_prompt_ids=student_prompt_ids,
                teacher_input_ids=[*teacher_prompt_ids, *completion_ids],
                completion_ids=completion_ids,
            )
        )
    return targets


def selection_fingerprint(targets: list[TargetSpec]) -> str:
    digest = hashlib.sha256()
    for target in targets:
        record = {
            "key": target.key,
            "row_weight": target.row_weight,
            "student_prompt_sha256": hashlib.sha256(
                json.dumps(target.student_prompt_ids).encode()
            ).hexdigest(),
            "teacher_sha256": hashlib.sha256(
                json.dumps(target.teacher_input_ids).encode()
            ).hexdigest(),
            "completion_sha256": hashlib.sha256(
                json.dumps(target.completion_ids).encode()
            ).hexdigest(),
        }
        digest.update(json.dumps(record, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def normalized_topk(entries: list[Any], topk: int) -> list[list[float | int]]:
    chosen = entries[:topk]
    if len(chosen) != topk:
        raise ValueError(f"expected {topk} candidates, got {len(chosen)}")
    token_ids = [int(entry.token_id) for entry in chosen]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("teacher returned duplicate top-k token IDs")
    logprobs = [float(entry.logprob) for entry in chosen]
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("teacher returned non-finite top-k logprob")
    maximum = max(logprobs)
    denominator = sum(math.exp(value - maximum) for value in logprobs)
    normalized = [
        [token_id, math.exp(logprob - maximum) / denominator]
        for token_id, logprob in zip(token_ids, logprobs, strict=True)
    ]
    if not math.isclose(
        sum(float(item[1]) for item in normalized),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("normalized top-k probabilities do not sum to one")
    return normalized


def cache_record(
    target: TargetSpec,
    prompt_top: list[list[Any]],
    topk: int,
) -> dict[str, Any]:
    distributions: list[list[list[float | int]]] = []
    for position in target.teacher_positions:
        if position >= len(prompt_top):
            raise ValueError(
                f"{target.key}: missing teacher position {position}/"
                f"{len(prompt_top)}"
            )
        distributions.append(normalized_topk(prompt_top[position], topk))
    if len(distributions) != len(target.completion_ids):
        raise ValueError(f"{target.key}: completion/distribution mismatch")
    return {
        "key": target.key,
        "kind": target.kind,
        "task_id": target.task_id,
        "turn_idx": target.turn_idx,
        "step_idx": target.step_idx,
        "row_weight": target.row_weight,
        "student_prompt_ids": target.student_prompt_ids,
        "completion_ids": target.completion_ids,
        "topk_by_position": distributions,
    }


def query_batch(
    client: river.Client,
    renderer: Any,
    targets: list[TargetSpec],
    topk: int,
    retries: int,
) -> list[dict[str, Any]]:
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            samples = client.sample(
                prompt_token_ids=[
                    target.teacher_input_ids for target in targets
                ],
                base_model=MODEL,
                max_tokens=1,
                temperature=0.0,
                return_prompt_logprobs=True,
                logprobs=topk,
                tokenizer=renderer.tokenizer,
                timeout=1800,
            )
            if len(samples) != len(targets):
                raise ValueError(
                    f"batch returned {len(samples)} samples for "
                    f"{len(targets)} targets"
                )
            return [
                cache_record(
                    target,
                    sample.prompt_top_logprobs or [],
                    topk,
                )
                for target, sample in zip(targets, samples, strict=True)
            ]
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt == retries:
                break
            time.sleep(2**attempt)
    assert error is not None
    raise error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repairs", default=DEFAULT_REPAIRS)
    parser.add_argument("--preservation", default=DEFAULT_PRESERVATION)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    renderer = get_renderer(MODEL, thinking=False)
    repair_rows = read_jsonl(REPO_ROOT / args.repairs)
    preservation_rows = read_jsonl(REPO_ROOT / args.preservation)
    repair_counts = Counter(category(row["task_id"]) for row in repair_rows)
    selected_preservation = select_preservation_rows(
        preservation_rows,
        dict(repair_counts),
        args.seed,
    )
    if len(selected_preservation) != len(repair_rows):
        raise ValueError("repair/preservation row counts do not match")

    targets: list[TargetSpec] = []
    for row in repair_rows:
        targets.append(prepare_repair_target(renderer, row))
    for row in selected_preservation:
        targets.extend(prepare_preservation_targets(row))
    targets.sort(key=lambda target: target.key)
    duplicate_keys = [
        key for key, count in Counter(target.key for target in targets).items()
        if count > 1
    ]
    if duplicate_keys:
        raise ValueError(f"duplicate target keys: {duplicate_keys[:20]}")

    source_weights = {
        kind: sum(
            target.row_weight for target in targets if target.kind == kind
        )
        for kind in ("repair", "preserve")
    }
    if not math.isclose(
        source_weights["repair"],
        source_weights["preserve"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"source weights are not 1:1: {source_weights}")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "top20_targets.jsonl"
    manifest_path = out_dir / "target_manifest.json"
    selection_path = out_dir / "selection.json"
    fingerprint = selection_fingerprint(targets)

    selection = {
        "seed": args.seed,
        "repair_rows": len(repair_rows),
        "repair_rows_by_category": dict(sorted(repair_counts.items())),
        "preservation_rows": len(selected_preservation),
        "preservation_rows_by_category": dict(
            sorted(
                Counter(
                    category(row["task_id"])
                    for row in selected_preservation
                ).items()
            )
        ),
        "selected_preservation_keys": [
            [row["task_id"], int(row["turn_idx"])]
            for row in selected_preservation
        ],
        "repair_target_completions": sum(
            target.kind == "repair" for target in targets
        ),
        "preservation_target_completions": sum(
            target.kind == "preserve" for target in targets
        ),
        "source_total_weights": source_weights,
        "target_fingerprint": fingerprint,
    }
    selection_path.write_text(json.dumps(selection, indent=2))

    completed: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            completed[str(row["key"])] = row
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text())
        old_fingerprint = old_manifest.get("target_fingerprint")
        if old_fingerprint and old_fingerprint != fingerprint:
            raise ValueError("existing target cache belongs to another selection")

    pending = [target for target in targets if target.key not in completed]
    manifest = {
        **selection,
        "model": MODEL,
        "topk": args.topk,
        "target_completions": len(targets),
        "completed_targets": len(completed),
        "status": "querying" if pending else "complete",
        "cache_path": str(cache_path.relative_to(REPO_ROOT)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        "RIVER_PSD_TARGET_PLAN "
        f"repair_rows={len(repair_rows)} "
        f"preserve_rows={len(selected_preservation)} "
        f"targets={len(targets)} pending={len(pending)} "
        f"weights={source_weights}",
        flush=True,
    )
    if args.plan_only:
        manifest["status"] = "planned"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return 0

    client = river.Client(api_key=os.environ["RIVER_API_KEY"])
    started = time.time()
    try:
        with cache_path.open("a") as handle:
            for offset in range(0, len(pending), args.query_batch_size):
                batch = pending[offset : offset + args.query_batch_size]
                records = query_batch(
                    client,
                    renderer,
                    batch,
                    args.topk,
                    args.retries,
                )
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed[record["key"]] = record
                handle.flush()
                manifest["completed_targets"] = len(completed)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(
                    "RIVER_PSD_TARGET_PROGRESS "
                    f"{len(completed)}/{len(targets)} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    finally:
        client.close()

    missing = sorted(set(target.key for target in targets) - set(completed))
    extra = sorted(set(completed) - set(target.key for target in targets))
    manifest.update(
        {
            "completed_targets": len(completed),
            "missing_targets": missing,
            "extra_targets": extra,
            "status": "complete" if not missing and not extra else "invalid",
            "elapsed_query_sec": time.time() - started,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"RIVER_PSD_TARGETS_DONE {json.dumps(manifest)}", flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
