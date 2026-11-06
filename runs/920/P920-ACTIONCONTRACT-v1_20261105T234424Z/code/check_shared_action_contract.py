"""Bounded CPU checks of a future shared action boundary; never runs a model."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

from shared_action_contract import ActionContract, context_from_request, schemas_from_observation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fixtures(context):
    asset = context["asset_catalog"][0]["asset_id"]
    objects = [oid for oid, obj in context["objects"].items() if obj["object_type"] == "furniture"]
    def call(name, **args):
        return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    good = [
        ("query", call("get_current_scene_state")),
        ("add", call("add_furniture_to_scene_tool", asset_id=asset, x=-1.25, y=2.5, yaw=90.0)),
        ("move", call("move_furniture_tool", object_id=objects[0], x=1, y=-2, yaw=0)),
        ("remove", call("remove_furniture_tool", object_id=objects[0])),
        ("facing", call("check_facing_tool", object_a_id=objects[0], object_b_id=objects[1], direction="toward")),
        ("done", call("__done__", unmet_or_unknown="Unverified comfort; 停止，保留未知。")),
        ("alternate_order", '{"arguments": {}, "name": "get_current_scene_state"}'),
    ]
    bad = [
        ("unknown_tool", call("invented_tool")),
        ("extra_argument", call("get_current_scene_state", success=True)),
        ("missing_required", call("move_furniture_tool", object_id=objects[0], x=1, y=2)),
        ("unknown_object", call("remove_furniture_tool", object_id="not_an_object")),
        ("unknown_asset", call("add_furniture_to_scene_tool", asset_id="not_an_asset", x=0, y=0, yaw=0)),
        ("bool_coordinate", call("move_furniture_tool", object_id=objects[0], x=True, y=0, yaw=0)),
        ("string_coordinate", call("move_furniture_tool", object_id=objects[0], x="1", y=0, yaw=0)),
        ("nan", call("move_furniture_tool", object_id=objects[0], x=float("nan"), y=0, yaw=0)),
        ("overflow", call("move_furniture_tool", object_id=objects[0], x=1, y=0, yaw=0).replace('"x": 1', '"x": 1e999')),
        ("duplicate_key", '{"name":"get_current_scene_state","name":"get_current_scene_state","arguments":{}}'),
        ("multiple_calls", good[0][1] + good[0][1]),
        ("partial", good[0][1][:-1]),
        ("outer_extra", '{"name":"get_current_scene_state","arguments":{},"success":true}'),
        ("array", '[' + good[0][1] + ']'),
        ("done_with_tool", '{"name":"__done__","arguments":{"unmet_or_unknown":"unknown","tool":"move"}}'),
    ]
    return good, bad


def guard_check(plan, out, contract, context, tools):
    from jsonschema import ValidationError
    good, bad = fixtures(context)
    checks = []
    for expected, rows in ((True, good), (False, bad)):
        for label, raw in rows:
            try:
                contract.validate(raw, context)
                accepted, error = True, None
            except (ValueError, ValidationError) as exc:
                accepted, error = False, type(exc).__name__
            checks.append({"case": label, "expected_accept": expected, "accepted": accepted, "error": error})
    changed = copy.deepcopy(context)
    changed["objects"].pop(next(oid for oid, x in changed["objects"].items() if x["object_type"] == "furniture"))
    try:
        contract.validate(good[0][1], changed)
        stale_rejected = False
    except ValueError:
        stale_rejected = True
    empty = copy.deepcopy(context)
    empty["objects"] = {}
    empty["asset_catalog"] = []
    empty_contract = ActionContract(tools, empty)
    empty_contract.validate(good[0][1], empty)
    empty_contract.validate(good[5][1], empty)
    historical = []
    root = Path(plan["source_run"])
    for turn in range(7):
        rd = root / "rollout" / f"round_{turn:03d}"
        raw_path = rd / "model/raw_output.txt"
        request_path = rd / "request.json"
        raw = raw_path.read_text()
        match = re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*", raw, re.DOTALL)
        ctx = context_from_request(json.loads(request_path.read_text()))
        gate = ActionContract(tools, ctx)
        try:
            if match is None:
                raise ValueError("Incomplete or non-single historical envelope")
            gate.validate(match.group(1), ctx)
            accepted, error = True, None
        except (ValueError, ValidationError) as exc:
            accepted, error = False, type(exc).__name__
        historical.append({"turn": turn, "accepted": accepted, "error": error,
                           "raw_path": str(raw_path), "raw_sha256": sha(raw_path),
                           "request_path": str(request_path), "request_sha256": sha(request_path)})
    result = {"fixtures": checks, "stale_rejected": stale_rejected,
              "empty_context_query_and_done_passed": True, "historical": historical,
              "historical_reexecution": False,
              "boundary": "A format-valid retained-shelf deletion and bad geometry remain accepted; this is not a task or design oracle."}
    write(out / "GUARD_CHECKS.json", result)
    assert all(row["accepted"] == row["expected_accept"] for row in checks)
    assert stale_rejected
    assert [row["accepted"] for row in historical] == [False, True, True, True, True, False, False]


def tokenizer_check(plan, out, contract, context):
    import torch
    from transformers import AutoTokenizer
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    model_plan = json.loads((Path(plan["source_run"]) / "PLAN.json").read_text())
    model_dir = Path(model_plan["model_directory"])
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"):
        assert sha(model_dir / name) == model_plan["model_configuration_sha256"][name]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    began = time.monotonic()
    data = build_token_enforcer_tokenizer_data(tokenizer)
    trie_seconds = time.monotonic() - began
    good, bad = fixtures(context)
    rows = []
    seed = tokenizer.encode("CPU tokenizer check only.\n", add_special_tokens=False)
    eos = tokenizer.eos_token_id
    # Negative cases whose exclusion is promised by the grammar, not by the
    # independent numeric/duplicate-key validator. Final validation is mandatory.
    selected_bad = [(label, text) for label, text in bad if label in (
        "unknown_tool", "extra_argument", "missing_required", "unknown_object",
        "unknown_asset", "bool_coordinate", "string_coordinate", "multiple_calls")]
    for expected, cases in ((True, good), (False, selected_bad)):
        for label, raw in cases:
            prefix = list(seed)
            fn = contract.prefix_function(data)
            ids = tokenizer.encode(raw, add_special_tokens=False)
            first_blocked = None
            early_eos = False
            for position, tid in enumerate(ids):
                allowed = fn(0, torch.tensor(prefix, dtype=torch.long))
                if eos in allowed:
                    early_eos = True
                if tid not in allowed:
                    first_blocked = position
                    break
                prefix.append(tid)
            eos_at_end = first_blocked is None and eos in fn(0, torch.tensor(prefix, dtype=torch.long))
            accepted = first_blocked is None and eos_at_end
            rows.append({"case": label, "expected_accept": expected, "accepted": accepted,
                         "tokens": len(ids), "first_blocked_token": first_blocked,
                         "eos_allowed_before_complete": early_eos, "eos_at_end": eos_at_end})
            write(out / "TOKENIZER_CHECKS.json", {"status": "running", "trie_seconds": trie_seconds, "cases": rows})
    result = {"trie_seconds": trie_seconds, "tokenizer_vocab_size": len(tokenizer),
              "tokenizer_eos_id": eos, "model_eos_ids": json.loads((model_dir / "generation_config.json").read_text()).get("eos_token_id"),
              "cases": rows, "model_loaded": False, "gpu_used": False,
              "interpretation": "Forced fixture token walks only, not sampled outputs or proof of native rollout success."}
    write(out / "TOKENIZER_CHECKS.json", result)
    assert all(row["accepted"] == row["expected_accept"] for row in rows)
    assert not any(row["eos_allowed_before_complete"] for row in rows if row["expected_accept"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--phase", choices=("guard", "tokenizer"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    for path, expected in plan["inputs"].items():
        assert sha(path) == expected
    record = json.loads((Path(plan["public"]) / "CODE_RECORD.json").read_text())
    for path, expected in record["files"].items():
        assert sha(path) == expected
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def deny(*unused, **kwargs):
        raise RuntimeError("Network disabled during CPU contract checks")
    socket.socket.connect = deny
    socket.create_connection = deny
    src = Path(plan["source_run"])
    tools = schemas_from_observation(json.loads((src / "prepare/observation.json").read_text()))
    ctx = context_from_request(json.loads((src / "prepare/request.json").read_text()))
    contract = ActionContract(tools, ctx)
    write(args.out / "SCHEMA.json", contract.schema)
    began = time.monotonic()
    result = {"started_at": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
              "status": "running", "model_calls": 0, "native_tool_calls": 0, "training_updates": 0}
    print(result["started_at"], "begin", args.phase, flush=True)
    try:
        if args.phase == "guard":
            guard_check(plan, args.out, contract, ctx, tools)
        else:
            tokenizer_check(plan, args.out, contract, ctx)
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(ended_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-began)
        write(args.out / "RESULT.json", result)
        print(result["ended_at"], result["status"], flush=True)


if __name__ == "__main__":
    main()
