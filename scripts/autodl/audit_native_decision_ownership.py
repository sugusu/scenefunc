"""Read archived native evidence; attribute decisions without replaying tools/models.

Uses only the standard library. Ownership annotations describe the pinned source
interface, not causal effect sizes, expert labels, or successful scene generation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time


ARCHIVE_SHA = "82e203bed22abfe371afe672fca139fa3175670b32d84cd7c29640db9d7257a1"

# Reviewed interface responsibilities. These are analysis metadata, not labels
# to be supplied to a policy or assertions that an operation was correct.
OWNERS = {
    "add_door": ("architecture_choice", "wall, opening type, position, dimensions", "opening geometry and object creation"),
    "add_window": ("architecture_choice", "wall, position, dimensions, sill", "opening geometry and object creation"),
    "get_material": ("retrieval_query", "text query", "material lookup and returned ID"),
    "set_room_materials": ("material_assignment", "room and material IDs", "material application"),
    "set_exterior_material": ("material_assignment", "material ID", "material application"),
    "generate_assets": ("asset_request", "descriptions, names, desired dimensions, style", "asset retrieval; mesh processing; separate VLM material/mass/front analysis"),
    "generate_wall_assets": ("asset_request", "descriptions, names, desired dimensions, style", "asset retrieval; mesh processing; separate VLM material/mass/front analysis"),
    "add_furniture_to_scene_tool": ("numeric_placement", "asset ID, x, y, yaw", "instance ID; fixed floor/upright axes; native placement processing"),
    "move_furniture_tool": ("numeric_move", "object ID, x, y, yaw", "transform application and native processing"),
    "snap_to_object_tool": ("relation_choice", "object, target, toward/away/none", "collision push-out; numeric orientation; snap algorithm and translation"),
    "rescale_furniture_tool": ("scale_choice", "object ID and scale multiplier", "shared-asset edit and all affected instances"),
    "check_physics": ("measurement_request", "whether/when to check", "physics computation; not independently established task truth"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_requests(archive: Path):
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError("Frozen archive hash mismatch")
    with tarfile.open(archive, "r:xz") as t:
        f = t.extractfile("paths.json")
        if f is None:
            raise ValueError("Missing archive manifest")
        manifest = json.load(f)
    rows = [x for x in manifest["files"] if re.fullmatch(r"actor/request_\d{3}(?:_receipt)?\.json", x["path"])]
    wanted = {x["sha256"]: x for x in rows}
    contents = {}
    with tarfile.open(archive, "r|xz") as t:
        for member in t:
            digest = member.name.removeprefix("blobs/")
            if digest not in wanted:
                continue
            f = t.extractfile(member)
            if f is None:
                raise ValueError("Expected regular JSON blob")
            b = f.read()
            row = wanted.pop(digest)
            if len(b) != row["bytes"] or hashlib.sha256(b).hexdigest() != digest:
                raise ValueError("Selected blob integrity mismatch")
            contents[row["path"]] = json.loads(b)
            if not wanted:
                break
    if wanted or len(rows) != 46 or len(contents) != 46:
        raise ValueError("Expected all 23 requests and 23 receipts")
    return manifest, rows, contents


def source_record(path: Path, root: Path, names: list[str]):
    text = path.read_text()
    tree = ast.parse(text)
    defs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            defs.append({"name": node.name, "start": node.lineno, "end": node.end_lineno})
    if set(names) - {x["name"] for x in defs}:
        raise ValueError(f"Missing reviewed definitions in {path}")
    return {"path": str(path.relative_to(root)), "sha256": sha(path), "definitions": defs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r, out = args.repository.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; never overwrite old audits")
    out.mkdir(parents=True)
    start = time.monotonic()
    archive = r / "runs/autodl/20261110_complete_native_runtime/evidence_solid_v2/actual_scene_evidence.tar.xz"
    packets_path = r / "runs/autodl/20261111_closed_loop_data/feedback_material_v2/offline_feedback_packets.json"
    manifest, request_rows, requests = read_requests(archive)
    archived_hash = {x["path"]: x["sha256"] for x in manifest["files"]}
    packets = json.loads(packets_path.read_text())
    receipts = {k: v for k, v in requests.items() if "_receipt" in k}
    all_calls, policy_rows, tool_rows = [], [], []
    used = set()
    for packet in packets:
        matching = [(k, v) for k, v in receipts.items() if v["endpoint"] == "/native_policy" and v["result"]["raw"] == packet["raw_proposal"]]
        if len(matching) != 1:
            raise ValueError("Policy proposal does not uniquely match archived raw response")
        receipt_name, receipt = matching[0]
        used.add(receipt_name)
        request_name = receipt_name.replace("_receipt", "")
        if receipt["request_sha256"] != archived_hash[request_name]:
            raise ValueError("Receipt request hash mismatch")
        raw_calls = [json.loads(s) for s in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", receipt["result"]["raw"], re.S)]
        if raw_calls != packet["source_tool_calls"]:
            raise ValueError("Archived raw tool calls differ from previous parsed evidence")
        if len(raw_calls) != len(packet["original_feedback"]):
            raise ValueError("Cannot align raw calls with saved outputs")
        system = requests[request_name]["messages"][0]["content"]
        policy_rows.append({"packet": packet["id"], "request": request_name, "receipt": receipt_name,
                            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                            "shared_asset_warning_in_actual_input": "Affects ALL instances of the same asset" in system,
                            "calls": len(raw_calls), "expert_target": False})
        for i, (call, feedback) in enumerate(zip(raw_calls, packet["original_feedback"])):
            if call["name"] != feedback["name"]:
                raise ValueError("Tool name alignment failure")
            category, policy, internal = OWNERS[call["name"]]
            all_calls.append({"packet": packet["id"], "batch_index": i, "request": request_name,
                              "name": call["name"], "arguments": call["arguments"], "category": category,
                              "policy_decides": policy, "tool_decides_or_computes": internal,
                              "saved_output": feedback["output"], "output_category": feedback["category"],
                              "expert_correctness": None, "complete_scene_success": None})
    for name, receipt in sorted(receipts.items()):
        if receipt["endpoint"] != "/completion":
            continue
        request_name = name.replace("_receipt", "")
        if receipt["request_sha256"] != archived_hash[request_name]:
            raise ValueError("Tool VLM receipt request hash mismatch")
        req = requests[request_name]
        content = req["messages"][0]["content"]
        texts = [x["text"] for x in content if x["type"] == "text"]
        prompt = "\n".join(texts)
        asset = re.search(r"asset name:\s*([^\s)]+)", prompt)
        tool_rows.append({"request": request_name, "receipt": name,
                          "asset_name_in_input": asset.group(1) if asset else None,
                          "input_image_blocks": sum(x["type"] == "image_url" for x in content),
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "requested_fields": ["material", "mass_range_kg", "mass_kg", "mass_source", "canonical_orientation"],
                          "dimensions_explicitly_excluded_in_prompt": "should NOT predict `dimensions_m`" in prompt,
                          "saved_output": json.loads(receipt["result"]["text"]),
                          "sdk_model_name_not_actual_backend_identity": req.get("model"),
                          "source_output_is_ground_truth": False})
    if len(used) != 19 or len(tool_rows) != 4 or len(all_calls) != 35:
        raise ValueError("Historical count mismatch")

    native = r / "runs/autodl/20261110_remote_native_controller/runtime/native_source/scenesmith"
    source_specs = {
        "furniture_agents/tools/furniture_tools.py": ["generate_assets", "add_furniture_to_scene_tool", "move_furniture_tool", "rescale_furniture_tool", "_rescale_furniture_impl"],
        "furniture_agents/tools/scene_tools.py": ["snap_to_object_tool", "_snap_to_object_impl"],
        "furniture_agents/tools/vision_tools.py": ["check_physics"],
        "wall_agents/tools/wall_tools.py": ["generate_wall_assets"],
        "agent_utils/asset_manager.py": ["_retrieve_hssd_assets"],
        "agent_utils/hssd_retrieval_server/server_app.py": ["_retrieve_internal"],
        "floor_plan_agents/tools/materials_resolver.py": ["get_material"],
    }
    sources = [source_record(native / path, r, names) for path, names in source_specs.items()]
    rescale = [x for x in all_calls if x["name"] == "rescale_furniture_tool"]
    shared = [x for x in rescale if len(x["saved_output"].get("affected_object_ids", [])) > 1]
    scale_groups = {}
    for x in shared:
        scale_groups.setdefault(x["saved_output"]["asset_id"], []).append(x)
    repeated = []
    for path, rows in scale_groups.items():
        if len(rows) < 2:
            continue
        repeated.append({"same_asset_path": path, "policy_objects": [x["arguments"]["object_id"] for x in rows],
                         "multipliers": [x["arguments"]["scale_factor"] for x in rows],
                         "saved_new_asset_scales": [x["saved_output"]["new_asset_scale"] for x in rows],
                         "saved_final_dimensions": rows[-1]["saved_output"]["new_dimensions"],
                         "warning_was_present": all(next(p for p in policy_rows if p["packet"] == x["packet"])["shared_asset_warning_in_actual_input"] for x in rows),
                         "ordinary_api_semantics_case_not_new_algorithm_evidence": True})
    dimensions = []
    for x in all_calls:
        if x["name"] != "generate_assets":
            continue
        args_, output = x["arguments"], x["saved_output"]
        by_name = {y["name"]: y for y in output.get("assets", [])}
        for name, dims in zip(args_["short_names"], args_["desired_dimensions"]):
            if name not in by_name:
                continue
            y = by_name[name]
            actual = [y[k] for k in ["width", "depth", "height"]]
            dimensions.append({"request": x["request"], "short_name": name, "object_id": y["object_id"],
                               "policy_desired_dimensions": dims, "saved_returned_dimensions": actual,
                               "returned_over_requested": [a / b for a, b in zip(actual, dims)],
                               "user_specified_dimensions": False, "semantic_category_verified": False,
                               "dimensions_difference_is_failure_label": False})
    write_json(out / "policy_decisions.json", policy_rows)
    write_json(out / "tool_vlm_decisions.json", tool_rows)
    write_json(out / "tool_calls.json", all_calls)
    write_json(out / "source_manifest.json", sources)
    write_json(out / "shared_asset_semantics.json", repeated)
    write_json(out / "asset_request_returned_dimensions.json", dimensions)
    with (out / "ownership.csv").open("w", newline="") as f:
        fields = ["packet", "batch_index", "name", "category", "policy_decides", "tool_decides_or_computes", "output_category"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(all_calls)
    result = {"status": "archived_evidence_and_source_audited", "archive_sha256": ARCHIVE_SHA,
              "packets_sha256": sha(packets_path), "requests_and_receipts_verified": len(request_rows),
              "policy_generate_requests": len(policy_rows), "tool_generate_requests": len(tool_rows),
              "policy_batches_with_no_tool_calls": sum(x["calls"] == 0 for x in policy_rows),
              "saved_native_tool_calls": len(all_calls), "tool_call_counts": dict(Counter(x["name"] for x in all_calls)),
              "ownership_category_counts": dict(Counter(x["category"] for x in all_calls)),
              "furniture_policy_inputs_with_shared_asset_warning": sum(x["shared_asset_warning_in_actual_input"] for x in policy_rows if x["packet"].startswith("furniture:")),
              "tool_vlm_images_total": sum(x["input_image_blocks"] for x in tool_rows),
              "repeated_shared_asset_scale_groups": len(repeated),
              "new_models_loaded": 0, "new_pretrained_model_forwards": 0, "new_generate_requests": 0,
              "new_sdk_tool_calls": 0, "new_training_updates": 0, "new_scene_successes": 0,
              "independent_tasks": 1, "complete_native_trial_was_partial": True,
              "ownership_is_causal_effect_estimate": False,
              "elapsed_seconds": time.monotonic() - start}
    write_json(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
