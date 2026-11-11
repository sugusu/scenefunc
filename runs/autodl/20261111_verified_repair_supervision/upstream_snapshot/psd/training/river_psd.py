#!/usr/bin/env python3
"""Train one River-native PSD learning-rate arm from a validated top-k cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import river_client as river

from psd.paths import REPO_ROOT

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
CONTEXT_WINDOW = 262_144
DEFAULT_TARGET_DIR = "results/runs/psd/river_native_psd_r1"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_target(row: dict[str, Any], topk: int) -> None:
    prompt_ids = row.get("student_prompt_ids")
    completion_ids = row.get("completion_ids")
    distributions = row.get("topk_by_position")
    row_weight = float(row.get("row_weight", 0.0))
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError(f"{row.get('key')}: missing student prompt IDs")
    if not isinstance(completion_ids, list) or not completion_ids:
        raise ValueError(f"{row.get('key')}: missing completion IDs")
    if len(prompt_ids) + len(completion_ids) > CONTEXT_WINDOW:
        raise ValueError(f"{row.get('key')}: sequence exceeds context window")
    if not isinstance(distributions, list) or len(distributions) != len(
        completion_ids
    ):
        raise ValueError(f"{row.get('key')}: distribution length mismatch")
    if not math.isfinite(row_weight) or row_weight <= 0:
        raise ValueError(f"{row.get('key')}: invalid row weight")
    for position, entries in enumerate(distributions):
        if not isinstance(entries, list) or len(entries) != topk:
            raise ValueError(
                f"{row.get('key')}: position {position} does not have top-{topk}"
            )
        token_ids = [int(entry[0]) for entry in entries]
        probabilities = [float(entry[1]) for entry in entries]
        if len(set(token_ids)) != topk:
            raise ValueError(f"{row.get('key')}: duplicate top-k token")
        if not all(
            math.isfinite(probability) and probability >= 0
            for probability in probabilities
        ):
            raise ValueError(f"{row.get('key')}: invalid probability")
        if not math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"{row.get('key')}: probabilities do not sum to one"
            )


def expanded_datums(row: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    prompt_ids = [int(token) for token in row["student_prompt_ids"]]
    completion_ids = [int(token) for token in row["completion_ids"]]
    input_ids = [*prompt_ids, *completion_ids]
    row_weight = float(row["row_weight"])
    distributions = row["topk_by_position"]
    datums: list[dict[str, Any]] = []
    for rank in range(topk):
        targets = [0] * len(input_ids)
        weights = [0.0] * len(input_ids)
        for offset, entries in enumerate(distributions):
            completion_position = len(prompt_ids) + offset
            prediction_position = completion_position - 1
            targets[prediction_position] = int(entries[rank][0])
            weights[prediction_position] = (
                float(entries[rank][1]) * row_weight
            )
        datums.append(
            {
                "input_ids": input_ids,
                "target_tokens": targets,
                "weights": weights,
            }
        )
    return datums


def native_datum(row: dict[str, Any], topk: int) -> dict[str, Any]:
    """Single ``[T, K]`` top-k CE datum (native multi-target loss).

    Loss-equivalent to :func:`expanded_datums` (verified against it to within
    the FP8 forward-noise floor) but ~``topk``x cheaper: one datum per target
    instead of ``topk`` rank copies. Requires the server-side multi-target
    rollout and river-client >= 0.5.1 (2-D float ``weights`` encoding).
    """
    prompt_ids = [int(token) for token in row["student_prompt_ids"]]
    completion_ids = [int(token) for token in row["completion_ids"]]
    input_ids = [*prompt_ids, *completion_ids]
    row_weight = float(row["row_weight"])
    distributions = row["topk_by_position"]
    targets = [[0] * topk for _ in range(len(input_ids))]
    weights = [[0.0] * topk for _ in range(len(input_ids))]
    for offset, entries in enumerate(distributions):
        completion_position = len(prompt_ids) + offset
        prediction_position = completion_position - 1
        for rank in range(topk):
            targets[prediction_position][rank] = int(entries[rank][0])
            weights[prediction_position][rank] = (
                float(entries[rank][1]) * row_weight
            )
    return {
        "input_ids": input_ids,
        "target_tokens": targets,
        "weights": weights,
    }


def metrics_finite(metrics: dict[str, Any]) -> bool:
    return all(
        not isinstance(value, (int, float)) or math.isfinite(float(value))
        for value in metrics.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default=DEFAULT_TARGET_DIR)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--expanded-batch-size", type=int, default=256)
    parser.add_argument(
        "--native-topk",
        action="store_true",
        help="Use the native [T, K] multi-target cross_entropy (one datum "
        "per target) instead of the topk-way rank expansion. ~topk x cheaper "
        "and loss-equivalent. Requires river-client >= 0.5.1 and the "
        "server-side multi-target rollout. --expanded-batch-size still sets "
        "unique targets/step (= expanded_batch_size // topk).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    load_env(REPO_ROOT / ".env")
    target_dir = REPO_ROOT / args.target_dir
    cache_path = target_dir / "top20_targets.jsonl"
    manifest_path = target_dir / "target_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(
            f"target manifest is not complete: {manifest.get('status')}"
        )
    topk = int(manifest["topk"])
    if args.expanded_batch_size < topk:
        parser.error("--expanded-batch-size must be at least topk")

    rows = read_jsonl(cache_path)
    if len(rows) != int(manifest["target_completions"]):
        raise ValueError(
            f"cache has {len(rows)} rows, manifest expects "
            f"{manifest['target_completions']}"
        )
    keys = [str(row["key"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("target cache contains duplicate keys")
    for row in rows:
        validate_target(row, topk)

    source_weights = {
        kind: sum(
            float(row["row_weight"])
            for row in rows
            if row["kind"] == kind
        )
        for kind in ("repair", "preserve")
    }
    if not math.isclose(
        source_weights["repair"],
        source_weights["preserve"],
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise ValueError(f"source weights are not balanced: {source_weights}")

    out_dir = REPO_ROOT / "results/runs/psd/river_native_psd_train" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    checkpoints_path = out_dir / "progress_checkpoints.json"
    target_rows_per_step = max(1, args.expanded_batch_size // topk)
    expected_steps = math.ceil(len(rows) / target_rows_per_step) * args.epochs
    summary: dict[str, Any] = {
        "status": "training",
        "run_name": args.run_name,
        "project": args.project,
        "model": MODEL,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "expanded_batch_size": args.expanded_batch_size,
        "native_topk": args.native_topk,
        "target_rows_per_step": target_rows_per_step,
        "epochs": args.epochs,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "seed": args.seed,
        "target_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "target_cache_sha256": sha256_file(cache_path),
        "target_rows": len(rows),
        "target_rows_by_kind": dict(
            sorted(Counter(row["kind"] for row in rows).items())
        ),
        "source_total_weights": source_weights,
        "expected_train_steps": expected_steps,
        "progress_checkpoints": [],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    client = river.Client(api_key=os.environ["RIVER_API_KEY"])
    progress_checkpoints: list[dict[str, Any]] = []
    training_curve: list[dict[str, Any]] = []
    train_steps = 0
    training_token_presentations = 0
    started = time.time()
    try:
        with client.session(
            project=args.project,
            purpose="psd_training",
        ) as session:
            model = session.create_model(
                base_model=MODEL,
                lora=river.LoraConfig(
                    rank=args.lora_rank,
                    train_attn=True,
                    train_mlp=True,
                    train_unembed=False,
                    seed=args.seed,
                ),
            )
            stop = False
            for epoch in range(args.epochs):
                order = list(range(len(rows)))
                random.Random(args.seed + epoch).shuffle(order)
                for start in range(0, len(order), target_rows_per_step):
                    selected = [
                        rows[index]
                        for index in order[start : start + target_rows_per_step]
                    ]
                    if args.native_topk:
                        datums = [native_datum(row, topk) for row in selected]
                    else:
                        datums = [
                            datum
                            for row in selected
                            for datum in expanded_datums(row, topk)
                        ]
                    result = model.forward_backward(
                        datums,
                        loss_fn="cross_entropy",
                    )
                    optim = model.optim_step(
                        lr=args.learning_rate,
                        beta1=0.9,
                        beta2=0.999,
                        eps=1e-8,
                        weight_decay=0.0,
                        grad_clip_norm=args.grad_clip_norm,
                    )
                    train_steps += 1
                    batch_tokens = sum(
                        len(datum["input_ids"]) for datum in datums
                    )
                    training_token_presentations += batch_tokens
                    point = {
                        "step": train_steps,
                        "epoch": epoch + 1,
                        "target_rows": len(selected),
                        "expanded_datums": len(datums),
                        "batch_tokens": batch_tokens,
                        "kind_counts": dict(
                            sorted(
                                Counter(row["kind"] for row in selected).items()
                            )
                        ),
                        "loss": result.metrics,
                        "optimizer": optim.metrics,
                        "elapsed_sec": time.time() - started,
                    }
                    training_curve.append(point)
                    if (
                        not metrics_finite(result.metrics)
                        or not metrics_finite(optim.metrics)
                        or float(
                            optim.metrics.get("grad_norm_finite", 0.0)
                        )
                        != 1.0
                    ):
                        raise RuntimeError(
                            f"non-finite training metrics at step {train_steps}"
                        )
                    print(
                        "RIVER_PSD_TRAIN "
                        f"run={args.run_name} step={train_steps} "
                        f"loss={result.metrics.get('loss')} "
                        f"grad={optim.metrics.get('grad_norm')} "
                        f"targets={len(selected)} datums={len(datums)} "
                        f"tokens={training_token_presentations} "
                        f"elapsed={time.time() - started:.0f}s",
                        flush=True,
                    )

                    if (
                        args.checkpoint_every_steps
                        and train_steps % args.checkpoint_every_steps == 0
                    ):
                        checkpoint_name = (
                            f"{args.run_name}_step{train_steps:04d}"
                        )
                        inference = model.save_weights(
                            checkpoint_name,
                            mode="inference",
                        )
                        training = model.save_weights(
                            f"{checkpoint_name}_state",
                            mode="training",
                        )
                        checkpoint = {
                            "step": train_steps,
                            "sampler_path": inference.path,
                            "state_path": training.path,
                            "training_token_presentations": (
                                training_token_presentations
                            ),
                            "elapsed_sec": time.time() - started,
                        }
                        progress_checkpoints.append(checkpoint)
                        checkpoints_path.write_text(
                            json.dumps(progress_checkpoints, indent=2)
                        )
                        summary["progress_checkpoints"] = progress_checkpoints
                        summary["training_curve"] = training_curve
                        summary_path.write_text(
                            json.dumps(summary, indent=2, default=str)
                        )
                        print(
                            "RIVER_PSD_CHECKPOINT "
                            f"run={args.run_name} step={train_steps} "
                            f"sampler={inference.path} state={training.path}",
                            flush=True,
                        )

                    if (
                        args.max_train_steps
                        and train_steps >= args.max_train_steps
                    ):
                        stop = True
                        break
                if stop:
                    break

            final_name = f"{args.run_name}_final_step{train_steps:04d}"
            final_inference = model.save_weights(
                final_name,
                mode="inference",
            )
            final_training = model.save_weights(
                f"{final_name}_state",
                mode="training",
            )
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "error": repr(exc),
                "train_steps": train_steps,
                "training_token_presentations": training_token_presentations,
                "training_curve": training_curve,
                "progress_checkpoints": progress_checkpoints,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        raise
    finally:
        client.close()

    summary.update(
        {
            "status": "complete",
            "train_steps": train_steps,
            "training_token_presentations": training_token_presentations,
            "elapsed_train_sec": time.time() - started,
            "training_curve": training_curve,
            "progress_checkpoints": progress_checkpoints,
            "final_checkpoint": {
                "step": train_steps,
                "sampler_path": final_inference.path,
                "state_path": final_training.path,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"RIVER_PSD_TRAIN_DONE {json.dumps(summary, default=str)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
