"""Portable, explicit checkpoints for the existing-object native session.

These are cooperative checkpoints between tool batches. They do not claim
recovery of a process killed during a tool, or restore model/generation state.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from native_project import contained_path, material_records, read_json, sha256, validate_bundle


SESSION_SCHEMA = "scenefunc-existing-session-v1"


def canonical_hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def runtime_fingerprint(designer, scenesmith):
    """Bind restoration to the actual editing, memory and observation code."""
    paths = {
        "designer": (designer, ["src/agent_rollout_decoupled.py", "src/tool_server.py",
            "src/memory.py", "src/render_observation.py", "src_critic/pil_annotate.py",
            "tools/render_five_views.py", "tools/render_cardinal_views.py"]),
        "scenesmith": (scenesmith, ["scenesmith/agent_utils/room.py",
            "scenesmith/agent_utils/house.py", "scenesmith/agent_utils/placement_noise.py",
            "scenesmith/furniture_agents/stateful_furniture_agent.py",
            "scenesmith/furniture_agents/tools/furniture_tools.py",
            "scenesmith/furniture_agents/tools/scene_tools.py"]),
        "adapter": (Path(__file__).parent, ["native_project.py", "native_checkpoint.py",
            "existing_project_session.py"]),
    }
    return {f"{group}/{name}": sha256(root / name)
            for group, (root, names) in paths.items() for name in names}


def config_hash(config, directory):
    # Output-directory relocation is expected; every other effective setting
    # remains part of the compatibility check.
    def normalize(value):
        if isinstance(value, str):
            return value.replace(str(directory.resolve()), "{PROJECT_DIRECTORY}")
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value
    normalized = normalize(config)
    # compose_full_cfg generates this logging label from wall-clock time on
    # every invocation. It does not describe the editing algorithm or budget.
    if re.fullmatch(r"chained_replay_[0-9]{6}", str(normalized.get("name", ""))):
        normalized["name"] = "{AUTOMATIC_RUN_LABEL}"
    return canonical_hash(normalized)


def rotation_roundtrip_differences(stored, restored, path=()):
    """Allow only sub-picounit quaternion serialization drift, not pose edits."""
    if isinstance(stored, dict) and isinstance(restored, dict):
        if set(stored) != set(restored):
            raise ValueError(f"Native restoration changed fields at {path}")
        return [item for key in stored for item in rotation_roundtrip_differences(
            stored[key], restored[key], path + (key,))]
    if isinstance(stored, list) and isinstance(restored, list):
        if len(stored) != len(restored):
            raise ValueError(f"Native restoration changed array size at {path}")
        return [item for index, (a, b) in enumerate(zip(stored, restored))
                for item in rotation_roundtrip_differences(a, b, path + (index,))]
    if stored == restored:
        return []
    if (len(path) >= 3 and path[-3:-1] == ("transform", "rotation_wxyz")
            and type(path[-1]) is int and 0 <= path[-1] < 4
            and type(stored) is float and type(restored) is float
            and abs(stored - restored) <= 1e-12):
        return [{"path": list(path), "before": stored, "after": restored}]
    raise ValueError(f"Native restoration changed non-roundoff state at {path}")


def load_session_record(project):
    relative = project.manifest["inputs"].get("session")
    if relative is None:
        return None
    if relative not in project.manifest["files"]:
        raise ValueError("Unlisted session checkpoint")
    record = read_json(contained_path(project.directory, relative))
    if record.get("schema") != SESSION_SCHEMA or record.get("stage") != "furniture":
        raise ValueError("Unsupported session checkpoint scope")
    if record["room_id"] != project.room.room_id:
        raise ValueError("Checkpoint room differs from selected room")
    step = record["step_idx"]
    if type(step) is not int or step < 0 or len(record["receipts"]) != step:
        raise ValueError("Checkpoint step count differs from complete batch history")
    previous = record["initial_state_sha256"]
    transitions = record.setdefault("restoration_transitions", [])
    event_index = 0
    for completed in range(step + 1):
        while event_index < len(transitions) and transitions[event_index]["after_step"] == completed:
            event = transitions[event_index]
            if event["before_sha256"] != previous or event["reason"] != "native_quaternion_roundtrip":
                raise ValueError("Checkpoint restoration transition is discontinuous")
            if not event["differences"]:
                raise ValueError("A restoration transition must record its numerical differences")
            for difference in event["differences"]:
                path = difference["path"]
                if (len(path) < 3 or path[-3:-1] != ["transform", "rotation_wxyz"]
                        or type(path[-1]) is not int or not 0 <= path[-1] < 4
                        or type(difference["before"]) is not float or type(difference["after"]) is not float
                        or not 0 < abs(difference["before"] - difference["after"]) <= 1e-12):
                    raise ValueError("Invalid quaternion round-trip difference")
            previous = event["after_sha256"]
            event_index += 1
        if completed == step:
            break
        receipt = record["receipts"][completed]
        if receipt["step_idx"] != completed + 1 or receipt["before_sha256"] != previous:
            raise ValueError("Checkpoint receipt chain is discontinuous")
        if len(receipt["tool_calls"]) != len(receipt["results"]):
            raise ValueError("Checkpoint calls/results are not paired")
        previous = receipt["after_sha256"]
    if event_index != len(transitions):
        raise ValueError("Checkpoint restoration transitions are out of order")
    serialized = read_json(contained_path(project.directory, project.manifest["inputs"]["native_state"]))
    stored = serialized["rooms"][project.room.room_id]
    if previous != canonical_hash(stored):
        raise ValueError("Checkpoint history does not end at its serialized scene")
    restored = project.room.to_state_dict()
    differences = rotation_roundtrip_differences(stored, restored)
    if differences:
        # Keep the original exact tool hashes; numerical reconstruction is an
        # explicit non-tool event instead of silently rounding all coordinates.
        transitions.append({"after_step": step, "reason": "native_quaternion_roundtrip",
                            "before_sha256": previous, "after_sha256": canonical_hash(restored),
                            "source_manifest_sha256": sha256(project.directory / "manifest.json"),
                            "differences": differences})
    if not isinstance(record["memory"], dict):
        raise ValueError("Invalid native memory checkpoint")
    return record


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_checkpoint(project, destination, session_record):
    """Validate a new bundle completely before publishing its directory."""
    source, destination = project.directory.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Checkpoint destination must be new and outside the active project")
    # Confirm input resource bytes still match the imported manifest. Runtime
    # outputs are deliberately not in that allowlist.
    validate_bundle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        manifest = copy.deepcopy(project.manifest)
        for relative in manifest["files"]:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        # Preserve the original input identity and state even after objects are
        # deleted. Later checkpoints copy these same immutable origin records.
        if "origin/manifest.json" not in manifest["files"]:
            for name in ("manifest.json", *manifest["inputs"].values()):
                target = staging / "origin" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
        state = project.house.to_state_dict()
        for material in material_records(state["layout"]):
            material["path"] = str(Path(material["path"]).resolve().relative_to(source))
        runtime = copy.deepcopy(project.runtime)
        for room_id, room in project.house.rooms.items():
            metadata = runtime["rooms"][room_id]
            metadata["next_surface_id"] = room._surface_id_counter
            metadata["room_type"] = room.room_type
            bindings = metadata["asset_bindings"]
            if set(room.objects) - set(bindings):
                raise ValueError("New objects lack recorded asset identity")
            metadata["asset_bindings"] = {str(key): bindings[str(key)] for key in room.objects}
        _write_json(staging / manifest["inputs"]["native_state"], state)
        _write_json(staging / manifest["inputs"]["runtime_metadata"], runtime)
        manifest["inputs"]["session"] = "session_checkpoint.json"
        _write_json(staging / "session_checkpoint.json", session_record)
        manifest["checkpoint_parent"] = {"manifest_sha256": sha256(source / "manifest.json")}
        manifest["files"] = {str(path.relative_to(staging)): {
            "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(staging.rglob("*")) if path.is_file()}
        _write_json(staging / "manifest.json", manifest)
        validate_bundle(staging)
        if destination.exists():
            raise ValueError("Checkpoint destination appeared during export")
        staging.rename(destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {"directory": str(destination), "manifest_sha256": sha256(destination / "manifest.json"),
                "state_sha256": canonical_hash(state), "step_idx": session_record["step_idx"]}
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
