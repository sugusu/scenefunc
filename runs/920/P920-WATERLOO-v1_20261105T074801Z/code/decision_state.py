"""Source-preserving state/context adapter for the saved, bounded floor bridge.

This is not a general scene engine or a professional-data qualification gate.
It imports an existing checkpoint as context; it never invents placement calls
to make a room appear in upstream memory. Future outcomes stay outside episodes.
"""
import copy
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_document(document):
    state = document["payload"]
    if hashlib.sha256(canonical(state).encode()).hexdigest() != document["payload_sha256"]:
        raise ValueError("Checkpoint payload checksum mismatch")
    if state["format"] != "paris_floor_bridge_state_v1":
        raise ValueError("Unsupported checkpoint format")
    if state["cumulative_tool_calls"] != len(state["tool_history"]):
        raise ValueError("Tool-history count mismatch")
    for key in ("placed_instances", "asset_templates"):
        ids = [obj["object_id"] for obj in state[key]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs in " + key)
        for obj in state[key]:
            if obj["geometry_path"] not in state["geometry_hashes"]:
                raise ValueError("Unhashed object geometry")
    return state


def read_checkpoint(path):
    state = validate_document(json.loads(Path(path).read_text()))
    references = dict(state["geometry_hashes"])
    references[state["frozen_shell"]["path"]] = state["frozen_shell"]["sha256"]
    for filename, expected in references.items():
        if sha(filename) != expected:
            raise ValueError("Changed checkpoint dependency: " + filename)
    return state


def object_table(state):
    # Bind only from authentic successful placement receipts. An absent binding
    # is unknown, never inferred by splitting an ID or matching a mesh/name.
    bindings = {}
    for event in state["tool_history"]:
        call, reply = event["call"], event["response"]
        if reply.get("success") is not True:
            continue
        if call["name"] == "add_furniture_to_scene_tool":
            bindings[reply["object_id"]] = call["arguments"]["asset_id"]
        elif call["name"] == "remove_furniture_tool":
            bindings.pop(call["arguments"]["object_id"], None)
    rows = []
    for obj in sorted(state["placed_instances"], key=lambda x: x["object_id"]):
        rows.append({
            "object_id": obj["object_id"], "name": obj["name"],
            "template_asset_id": bindings.get(obj["object_id"]),
            "catalog_asset_id": obj["metadata"].get("original_asset"),
            "transform": copy.deepcopy(obj["transform"]),
            "local_bbox_min": obj["bbox_min"], "local_bbox_max": obj["bbox_max"],
            "immutable": obj["immutable"], "scale_factor": obj["scale_factor"],
            "geometry_sha256": state["geometry_hashes"][obj["geometry_path"]],
            "functional_quality_verified": obj["metadata"].get("functional_quality_verified"),
        })
    return rows


def verify_removals_or_failures(before, after, events):
    """Check this observed revision class only; do not simulate unknown tools."""
    if before["room_id"] != after["room_id"] or before["frozen_shell"] != after["frozen_shell"]:
        raise ValueError("Room or frozen shell changed")
    if after["tool_history"] != before["tool_history"] + events:
        raise ValueError("Revision history is not the exact recorded extension")
    current = {x["object_id"]: x for x in object_table(before)}
    removed = []
    for event in events:
        call, reply = event["call"], event["response"]
        if type(reply.get("success")) is not bool:
            raise ValueError("Ambiguous tool success")
        if not reply["success"]:
            continue
        if call["name"] != "remove_furniture_tool":
            raise ValueError("Successful tool outside this verification scope")
        oid = call["arguments"]["object_id"]
        if oid not in current or reply.get("object_id") != oid:
            raise ValueError("Successful removal has missing/mismatched object ID")
        del current[oid]
        removed.append(oid)
    actual = {x["object_id"]: x for x in object_table(after)}
    if current != actual:
        raise ValueError("Receipts do not account for saved final object state")
    return {"removed_object_ids": removed, "remaining_object_ids": sorted(current),
            "failed_calls": sum(e["response"]["success"] is False for e in events),
            "surviving_objects_unchanged": True}


def engineering_episode(*, state, original_brief, system, calls, views, provenance):
    """One observed assistant batch, not one invented sample per tool call.

    Original generated commentary is intentionally not a professional rationale
    target. The caller must separately verify calls, images and provenance.
    No final-state/metric argument is accepted by this interface.
    """
    observation = {"room_id": state["room_id"],
        "coordinate_convention": "Saved native SceneObject transform; metres, z-up, quaternion wxyz",
        "objects": object_table(state), "physics_ready": state["physics_ready"],
        "recent_actual_tool_receipts": copy.deepcopy(state["tool_history"][-2:])}
    context = (original_brief + "\n\nCurrent checkpoint observation (before this action batch):\n" +
               canonical(observation))
    return {"stage": "furniture", "system": system,
        "task_context": {"scene_description": context},
        "provenance": dict(provenance, kind="engineering_saved_model_action_extraction",
            training_eligible=False, professional_rationale=False,
            historical_prompt_reproduction=False,
            transformation="Original brief/system and actual actions; explicit pre-state/last two receipts. Original model commentary omitted."),
        "turns": [{"role": "assistant", "content": "", "images": list(views),
                   "tool_calls": copy.deepcopy(calls)}]}
