"""Bounded CPU checks against real native HouseScene, with synthetic fixtures.

No model, rendering, physics simulation, tool server or professional labels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

from native_project import SCHEMA, import_project, read_json, sha256, validate_bundle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def update_manifest(root):
    manifest = read_json(root / "manifest.json")
    manifest["files"] = {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"
    }
    write_json(root / "manifest.json", manifest)


def make_fixture(root):
    import numpy as np
    from pydrake.all import RigidTransform
    from scenesmith.agent_utils.house import HouseLayout, HouseScene, PlacedRoom, RoomGeometry, RoomSpec
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, SupportSurface, UniqueID
    root.mkdir()
    layout = HouseLayout(house_dir=root, house_prompt="Synthetic import engineering fixture.")
    house = HouseScene(layout)
    runtime = {"rooms": {}}
    for index, rid in enumerate(("studio_a", "store_b")):
        directory = root / f"room_{rid}"
        directory.mkdir()
        sdf = directory / "room.sdf"
        sdf.write_text('<sdf version="1.7"><model name="room"><static>true</static>'
                       '<link name="room_geometry_body_link"><visual name="floor">'
                       '<geometry><box><size>5 4 0.1</size></box></geometry>'
                       '</visual><visual name="north_wall"><pose>0 2 1.25 0 0 0</pose>'
                       '<geometry><box><size>5 0.05 2.5</size></box></geometry>'
                       '</visual></link></model></sdf>\n')
        table_sdf = directory / "table.sdf"
        table_sdf.write_text('<sdf version="1.7"><model name="table"><link name="body">'
                             '<visual name="box"><geometry><box><size>1 0.8 0.7</size>'
                             '</box></geometry></visual></link></model></sdf>\n')
        def surface(sid, height):
            return SupportSurface(UniqueID(sid), np.array([-0.5, -0.4, 0.0]),
                                  np.array([0.5, 0.4, 0.0]), RigidTransform([0, 0, height]))
        floor = SceneObject(UniqueID("floor"), ObjectType.FLOOR, "Fixture floor", "Synthetic",
                            RigidTransform(), sdf_path=sdf, immutable=True,
                            support_surfaces=[surface("S_0", 0.0)])
        wall = SceneObject(UniqueID("north_wall"), ObjectType.WALL, "North wall", "Synthetic wall",
                           RigidTransform([0, 2, 1.25]), sdf_path=sdf, immutable=True)
        geometry = RoomGeometry(ET.parse(sdf), sdf, walls=[wall], floor=floor, width=4.0, length=5.0,
                                wall_normals={"north_wall": np.array([0.0, -1.0])})
        room = RoomScene(geometry, directory, room_id=rid, room_type="studio",
                         text_description="Synthetic room, not a sourced design project.")
        table = SceneObject(UniqueID("table_0"), ObjectType.FURNITURE, "Fixture table", "Synthetic box",
                            RigidTransform([0.2 + index, 0.3, 0.35]), sdf_path=table_sdf,
                            bbox_min=np.array([-0.5,-0.4,-0.35]), bbox_max=np.array([0.5,0.4,0.35]),
                            support_surfaces=[surface("S_1", 0.7)],
                            metadata={"asset_source": "researcher_engineering_fixture"})
        room.add_object(table)
        room.add_object(wall)
        room._surface_id_counter = 7  # Includes historical IDs absent from this state.
        house.add_room(room)
        layout.room_specs.append(RoomSpec(rid, room_type="studio", width=4.0, length=5.0))
        layout.placed_rooms.append(PlacedRoom(rid, (index * 5.0, 0.0), 5.0, 4.0))
        layout.room_geometries[rid] = geometry
        runtime["rooms"][rid] = {
            "room_type": room.room_type, "next_surface_id": room._surface_id_counter,
            "asset_bindings": {"table_0": {"asset_id": f"fixture_box_{index}",
                                          "identity_provenance": "researcher_created_not_historical"},
                               "north_wall": {"asset_id": f"fixture_wall_{index}",
                                              "identity_provenance": "researcher_created_structure"}},
        }
    write_json(root / "house_state.json", house.to_state_dict())
    write_json(root / "runtime.json", runtime)
    write_json(root / "task.json", {"design_request": "Keep the existing table instance; inspect its current state.",
                                   "original_request": None, "request_provenance": "researcher_engineering_fixture",
                                   "unknowns": ["No real resident, source project or physics validation."],
                                   "evaluation_only_target": "DO_NOT_EXPOSE_TARGET"})
    write_json(root / "manifest.json", {
        "schema": SCHEMA, "coordinates": {"units": "metres", "up": "+Z", "frame": "native_room_local"},
        "project": {"project_id": "native_import_fixture", "family_id": "engineering_only",
                    "provenance": "researcher_engineering_fixture_not_professional_data"},
        "selected_room_id": "store_b",
        "inputs": {"native_state": "house_state.json", "runtime_metadata": "runtime.json", "task": "task.json"},
        "files": {},
    })
    update_manifest(root)
    return house


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenesmith", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scenesmith.resolve()))
    args.out.mkdir(parents=True, exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(), "begin real native project import audit", flush=True)
    from scenesmith.agent_utils.house import HouseScene
    from pydrake.all import RigidTransform
    fixture = args.out / "source"
    original = make_fixture(fixture)
    original_hashes = {p.relative_to(fixture).as_posix(): sha256(p) for p in fixture.rglob("*") if p.is_file()}
    checks = []
    def check(name, condition, evidence=None):
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})
        if not condition:
            write_json(args.out / "RESULT.json", {"status": "failed", "checks": checks})
            raise AssertionError(name)
    # Demonstrate the real upstream omission on a copy, not by mocking classes.
    raw_copy = args.out / "upstream_restore"
    shutil.copytree(fixture, raw_copy)
    raw = HouseScene.from_state_dict(read_json(raw_copy / "house_state.json"), raw_copy)
    next_raw = str(raw.rooms["store_b"].generate_surface_id())
    check("upstream_surface_counter_omission_reproduced", next_raw == "S_0", next_raw)
    check("upstream_room_type_omission_reproduced", raw.rooms["store_b"].room_type == "room")
    imported = import_project(fixture, args.out / "imported")
    check("nondefault_selected_room", imported.room.room_id == "store_b")
    check("all_native_room_ids_and_poses_preserved", imported.house.to_state_dict() == original.to_state_dict())
    check("room_type_restored", imported.room.room_type == "studio")
    check("structural_wall_binding_preserved", imported.room.room_geometry.walls[0] is imported.room.objects["north_wall"])
    check("layout_geometry_shares_actual_room_instance", all(imported.house.layout.room_geometries[k] is r.room_geometry
                                                            for k, r in imported.house.rooms.items()))
    check("historical_counter_restored", str(imported.room.generate_surface_id()) == "S_7")
    check("existing_object_id_not_reused", str(imported.room.generate_unique_id("table")) == "table_1")
    context = imported.visible_initial_context()
    check("input_context_excludes_scoring_target", "DO_NOT_EXPOSE_TARGET" not in json.dumps(context))
    check("no_fabricated_prior_receipts", context["prior_tool_receipts"] == [] and context["history_status"] == "not_imported")
    check("selected_room_asset_binding", context["asset_bindings"]["table_0"]["asset_id"] == "fixture_box_1")
    check("native_state_move", imported.room.move_object("table_0", RigidTransform([1.5, 0.4, 0.35])))
    check("unknown_object_move_rejected", imported.room.move_object("missing", RigidTransform()) is False)
    check("other_room_unchanged", imported.house.rooms["studio_a"].to_state_dict() == original.rooms["studio_a"].to_state_dict())
    # Save the edited native checkpoint and explicit runtime metadata, then
    # restore again. This is a state operation, not a tool/physics success.
    write_json(imported.directory / "house_state.json", imported.house.to_state_dict())
    for rid, room in imported.house.rooms.items():
        imported.runtime["rooms"][rid]["next_surface_id"] = room._surface_id_counter
    write_json(imported.directory / "runtime.json", imported.runtime)
    update_manifest(imported.directory)
    continued = import_project(imported.directory, args.out / "continued")
    check("edited_checkpoint_restored", continued.house.to_state_dict() == imported.house.to_state_dict())
    check("continued_surface_counter", str(continued.room.generate_surface_id()) == "S_8")
    check("source_bundle_unchanged", original_hashes == {
        p.relative_to(fixture).as_posix(): sha256(p) for p in fixture.rglob("*") if p.is_file()})
    def rejects(name, filename, mutate, rehash=True):
        directory = args.out / name
        shutil.copytree(fixture, directory)
        path = directory / filename
        if filename.endswith(".json"):
            value = read_json(path); mutate(value); write_json(path, value)
        else:
            mutate(path)
        if rehash:
            update_manifest(directory)
        try:
            validate_bundle(directory)
        except (ValueError, KeyError) as error:
            check(name, True, str(error)); return
        check(name, False, "Invalid bundle accepted")
    rejects("reject_stale_surface_counter", "runtime.json", lambda d: d["rooms"]["store_b"].update(next_surface_id=0))
    rejects("reject_layout_room_geometry_mismatch", "house_state.json", lambda d: d["layout"]["room_geometries"]["store_b"].update(width=99))
    rejects("reject_object_identity_mismatch", "house_state.json", lambda d: d["rooms"]["store_b"]["objects"]["table_0"].update(object_id="substituted"))
    rejects("reject_missing_asset_binding", "runtime.json", lambda d: d["rooms"]["store_b"]["asset_bindings"].clear())
    rejects("reject_unknown_selected_room", "manifest.json", lambda d: d.update(selected_room_id="absent"))
    rejects("reject_missing_sdf", "room_store_b/room.sdf", lambda p: p.unlink(), rehash=False)
    rejects("reject_changed_asset_bytes", "room_store_b/table.sdf", lambda p: p.write_text("changed"), rehash=False)
    rejects("reject_absolute_native_path", "house_state.json", lambda d: d["rooms"]["store_b"]["objects"]["table_0"].update(sdf_path="/tmp/other.sdf"))
    rejects("reject_unlisted_sdf_dependency", "room_store_b/table.sdf", lambda p: p.write_text('<sdf version="1.7"><model name="m"><uri>missing.obj</uri></model></sdf>'))
    result = {"status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "checks_passed": len(checks), "fixture_manifest_sha256": sha256(fixture / "manifest.json"),
              "native_class": f"{type(imported.house).__module__}.{type(imported.house).__name__}",
              "model_calls": 0, "training_updates": 0, "renders": 0, "physics_simulations": 0,
              "professional_samples": 0, "service_starts": 0,
              "limitations": ["Synthetic two-room fixture, not professional design evidence.",
                              "Native state import and editing only; no HTTP/session integration or five-view observation.",
                              "No render/physics/resource-service readiness claim; mesh-internal texture closure is not yet validated."]}
    write_json(args.out / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks_passed": len(checks)}), flush=True)


if __name__ == "__main__":
    main()
