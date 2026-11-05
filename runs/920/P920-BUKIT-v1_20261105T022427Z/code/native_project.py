"""Restore an explicit, portable SceneSmith project bundle in an isolated copy.

This is a state import, not a service/session start or a claim of physical
validity. It uses the fixed upstream HouseScene implementation unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
import xml.etree.ElementTree as ET


SCHEMA = "scenefunc-native-project-v1"
PATH_KEYS = {"sdf_path", "geometry_path", "image_path"}
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Non-finite JSON value: {value}")))


def contained_path(root: Path, relative: str, base: Path | None = None) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected portable relative path: {relative!r}")
    root = root.resolve()
    candidate = (base or root) / relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes project: {relative!r}")
    # Symlinks hide file provenance and can change after validation.
    current = candidate
    while current != root and current.is_relative_to(root):
        if current.is_symlink():
            raise ValueError(f"Symlink in project path: {relative!r}")
        current = current.parent
    return resolved


def native_paths(value: Any, root: Path, base: Path, files: dict) -> Any:
    """Normalize native path fields for comparison and check the allowlist."""
    if isinstance(value, list):
        return [native_paths(item, root, base, files) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in PATH_KEYS and item is not None:
            path = contained_path(root, item, base)
            relative = path.relative_to(root).as_posix()
            if relative not in files:
                raise ValueError(f"Unlisted native dependency: {relative}")
            result[key] = relative
        else:
            result[key] = native_paths(item, root, base, files)
    return result


def material_records(layout: dict):
    if layout.get("exterior_material"):
        yield layout["exterior_material"]
    for record in layout.get("room_materials", {}).values():
        for name in ("wall_material", "floor_material"):
            if record.get(name):
                yield record[name]


def validate_bundle(root: Path) -> tuple[dict, dict, dict, dict]:
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported native project schema")
    if manifest["coordinates"] != {"units": "metres", "up": "+Z", "frame": "native_room_local"}:
        raise ValueError("Native import requires explicit metre/Z-up room-local coordinates")
    project = manifest["project"]
    if not project.get("project_id") or not project.get("family_id") or not project.get("provenance"):
        raise ValueError("Project identity and provenance must be explicit")
    files = manifest["files"]
    for relative, record in files.items():
        path = contained_path(root, relative)
        if relative != path.relative_to(root).as_posix():
            raise ValueError("Manifest paths must be canonical")
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or changed file: {relative}")
    loaded = []
    for field in ("native_state", "runtime_metadata", "task"):
        relative = manifest["inputs"][field]
        if relative not in files:
            raise ValueError(f"Unlisted input: {field}")
        loaded.append(read_json(contained_path(root, relative)))
    state, runtime, task = loaded
    if not isinstance(task.get("design_request"), str) or not task["design_request"].strip():
        raise ValueError("An explicit design request is required")
    if task.get("request_provenance") not in {"original", "adapted", "researcher_engineering_fixture"}:
        raise ValueError("Request provenance must distinguish original and adapted input")
    if "original_request" not in task or "unknowns" not in task:
        raise ValueError("Original request (possibly null) and unknowns must be recorded")
    layout, rooms = state["layout"], state["rooms"]
    ids = set(rooms)
    if not ids or manifest["selected_room_id"] not in ids:
        raise ValueError("Missing selected room")
    specs = layout["rooms"]
    placed = layout["placed_rooms"]
    if (len(specs) != len(ids) or {r["id"] for r in specs} != ids
            or len(placed) != len(ids) or {r["room_id"] for r in placed} != ids
            or set(layout["room_geometries"]) != ids or set(runtime["rooms"]) != ids):
        raise ValueError("Layout, geometry, metadata and room identities disagree")
    for room_id, room in rooms.items():
        if not IDENTIFIER.fullmatch(room_id):
            raise ValueError("Room ID is not a portable directory identifier")
        base = root / f"room_{room_id}"
        normalized = native_paths(room, root, base, files)
        geometry = native_paths(layout["room_geometries"][room_id], root, root, files)
        if normalized["room_geometry"] != geometry:
            raise ValueError(f"Layout and room geometry disagree: {room_id}")
        objects = room["objects"]
        if any(key != obj["object_id"] for key, obj in objects.items()):
            raise ValueError("Object map key does not match native object ID")
        walls = {w["object_id"]: w for w in room["room_geometry"]["walls"]}
        expected_walls = {k: v for k, v in objects.items() if v["object_type"] == "wall"}
        if len(walls) != len(room["room_geometry"]["walls"]) or walls != expected_walls:
            raise ValueError("Geometry walls differ from restored wall objects")
        all_objects = list(objects.values())
        floor = room["room_geometry"].get("floor")
        if floor:
            if floor["object_id"] in objects:
                raise ValueError("Floor ID collides with a room object")
            all_objects.append(floor)
        surfaces = [s["surface_id"] for obj in all_objects for s in obj["support_surfaces"]]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("Duplicate support surface ID")
        for obj in all_objects:
            placement = obj.get("placement_info")
            if placement and placement["parent_surface_id"] not in surfaces:
                raise ValueError("Object refers to an absent parent support surface")
        metadata = runtime["rooms"][room_id]
        counter = metadata["next_surface_id"]
        if type(counter) is not int or counter < 0:
            raise ValueError("Surface counter must be a non-negative integer")
        used = [int(s[2:], 36) for s in surfaces if re.fullmatch(r"S_[0-9a-z]+", s)]
        if used and counter <= max(used):
            raise ValueError("Surface counter would reuse an existing ID")
        spec_type = next(r["type"] for r in specs if r["id"] == room_id)
        if metadata["room_type"] != spec_type:
            raise ValueError("Room type disagrees with layout")
        if set(metadata["asset_bindings"]) != set(objects):
            raise ValueError("Every native object needs an explicit asset binding")
        for binding in metadata["asset_bindings"].values():
            if not binding.get("asset_id") or not binding.get("identity_provenance"):
                raise ValueError("Missing asset identity/provenance")
    for material in material_records(layout):
        directory = contained_path(root, material["path"])
        prefix = directory.relative_to(root).as_posix() + "/"
        if not any(name.startswith(prefix) for name in files):
            raise ValueError("Material directory has no listed resources")
    # Check SDF resource closure before the native parser can use any file.
    for relative in files:
        if not relative.lower().endswith(".sdf"):
            continue
        path = contained_path(root, relative)
        for element in ET.parse(path).iter("uri"):
            uri = (element.text or "").strip()
            if uri.startswith("package://scene/"):
                target = contained_path(root, uri.removeprefix("package://scene/"))
            elif "://" in uri:
                raise ValueError(f"Unportable SDF URI: {uri}")
            else:
                target = contained_path(root, uri, path.parent)
            if target.relative_to(root).as_posix() not in files:
                raise ValueError(f"Unlisted SDF resource: {uri}")
    return manifest, state, runtime, task


@dataclass
class ImportedProject:
    house: Any
    manifest: dict
    runtime: dict
    task: dict
    directory: Path

    @property
    def room(self):
        return self.house.rooms[self.manifest["selected_room_id"]]

    def visible_initial_context(self) -> dict:
        """Current state and request, with no invented action history or labels."""
        return {
            "project_id": self.manifest["project"]["project_id"],
            "room_id": self.room.room_id,
            "design_request": self.task["design_request"],
            "request_provenance": self.task["request_provenance"],
            "original_request": self.task["original_request"],
            "unknowns": self.task["unknowns"],
            "objects": self.room.to_state_dict()["objects"],
            "asset_bindings": self.runtime["rooms"][self.room.room_id]["asset_bindings"],
            "prior_tool_receipts": [],
            "history_status": "not_imported",
        }


def import_project(source: Path, destination: Path) -> ImportedProject:
    """Validate, copy, and call actual upstream restoration; start no services."""
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Import destination must be new and outside the source bundle")
    manifest, _, _, _ = validate_bundle(source)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in ("manifest.json", *manifest["files"]):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    # Recheck copied bytes to detect source changes during copying.
    manifest, state, runtime, task = validate_bundle(destination)
    # Material.from_dict does not resolve relative paths; relocate these in
    # memory only. The copied source manifest/state retain their original bytes.
    for material in material_records(state["layout"]):
        material["path"] = str(contained_path(destination, material["path"]))
    from scenesmith.agent_utils.house import HouseScene
    house = HouseScene.from_state_dict(state, house_dir=destination)
    for room_id, room in house.rooms.items():
        metadata = runtime["rooms"][room_id]
        room.room_type = metadata["room_type"]
        room._surface_id_counter = metadata["next_surface_id"]
        # Upstream creates separate but equal geometry instances; bind downstream
        # wall/ceiling/layout access to the selected room's actual geometry.
        house.layout.room_geometries[room_id] = room.room_geometry
    return ImportedProject(house, manifest, runtime, task, destination)
