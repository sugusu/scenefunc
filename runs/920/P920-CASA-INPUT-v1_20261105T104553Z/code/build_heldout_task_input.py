"""Build one explicit research-transfer shell and reuse geometry-only catalog.

No source article, completed layout or designer rationale is read. This creates
an initial condition, not a furnished design or a validated independent test.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

from build_reference_room import box, save_sdf, sdf_root, write
from native_project import SCHEMA, import_project, sha256, validate_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'build declared held-out input, no model or rendering', flush=True)
    plan = json.loads(args.plan.read_text())
    for ref in plan['inputs'].values():
        if sha256(Path(ref['path'])) != ref['sha256']:
            raise ValueError('Input hash changed')
    for path, digest in plan['code_hashes'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Code hash changed')
    task = json.loads(Path(plan['inputs']['task']['path']).read_text())
    g = task['geometry_assumptions']
    if g['interior_rectangle'] != [0, 0, 6, 5] or g['height'] != 2.7:
        raise ValueError('This bounded builder implements the reviewed 30m2 transfer only')
    if (g['entrance']['interval_m'] != [.2, 1.1]
            or g['window']['interval_m'] != [.6, 2.]
            or g['terrace']['rectangle'] != [0, 5, 6, 7]
            or g['terrace']['access_interval_m'] != [4.8, 5.7]
            or g['fixed_service_core']['rectangle'] != [4.2, 0, 6, 2]
            or g['fixed_service_core']['door_interval_y_m'] != [.5, 1.4]):
        raise ValueError('Opening or service-core assumption differs from this bounded builder')
    sys.path.insert(0, plan['scenesmith'])
    from pydrake.all import RigidTransform
    from scenesmith.agent_utils.house import (HouseLayout, HouseScene, RoomGeometry,
        RoomSpec, PlacedRoom, ClearanceOpeningData)
    from scenesmith.agent_utils.room import SceneObject, ObjectType, UniqueID, SupportSurface, RoomScene
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = args.out / 'bundle'
    rid = 'whole_room_transfer_01'
    roomdir = bundle / ('room_' + rid)
    roomdir.mkdir(parents=True)
    root, link = sdf_root('declared_transfer_shell', 'room_geometry_body_link')
    box(link, 'floor', [0, 0, -.025], [6, 5, .05], [.72, .64, .54, 1])
    box(link, 'ceiling', [0, 0, 2.725], [6, 5, .05], [1, 1, 1, 1], visual=False)
    # The terrace is a declared outside access surface, not furniture placement
    # space in the selected native room; it is not a source-measured terrace.
    box(link, 'terrace_floor', [0, 3.5, -.025], [6, 2, .05], [.55, .55, .55, 1])
    walls, normals, openings, objects, bindings = [], {}, [], [], {}
    # Coordinates below are native centered coordinates, translated exactly
    # from the prospective TASK_INPUT's southwest-origin assumptions.
    wall_specs = [
        ('south', 0, -2.525, -3.05, 3.05, 2.7, [0, 1],
         [('entry', 'door', -2.8, -1.9, 0., 2.1)]),
        ('north', 0, 2.525, -3.05, 3.05, 2.7, [0, -1],
         [('window', 'window', -2.4, -1., .9, 2.2),
          ('terrace_access', 'door', 1.8, 2.7, 0., 2.1)]),
        ('west', 1, -3.025, -2.55, 2.55, 2.7, [1, 0], []),
        ('east', 1, 3.025, -2.55, 2.55, 2.7, [-1, 0], []),
        ('bath_north', 0, -.475, 1.15, 3.05, 2.7, [0, 1], []),
        ('bath_west', 1, 1.175, -2.55, -.45, 2.7, [-1, 0],
         [('bath_access', 'door', -2., -1.1, 0., 2.1)]),
        ('terrace_north', 0, 4.525, -3.05, 3.05, 1., [0, -1], []),
        ('terrace_west', 1, -3.025, 2.55, 4.55, 1., [1, 0], []),
        ('terrace_east', 1, 3.025, 2.55, 4.55, 1., [-1, 0], []),
    ]
    for name, axis, fixed, low, high, height, normal, cuts in wall_specs:
        uc = sorted({low, high, *(v for c in cuts for v in c[2:4])})
        zc = sorted({0., height, *(v for c in cuts for v in c[4:6])})
        for i, (left, right) in enumerate(zip(uc, uc[1:])):
            for j, (bottom, top) in enumerate(zip(zc, zc[1:])):
                u, z = (left + right) / 2, (bottom + top) / 2
                if any(c[2] < u < c[3] and c[4] < z < c[5] for c in cuts):
                    continue
                center = [0., 0., z]; center[axis] = u; center[1-axis] = fixed
                size = [.05, .05, top-bottom]; size[axis] = right-left
                box(link, f'{name}_{i}_{j}', center, size, [.85, .85, .85, 1])
        center = [0., 0., height/2]; center[axis] = (low+high)/2; center[1-axis] = fixed
        size = np.array([.05, .05, height]); size[axis] = high-low
        oid = 'shell_' + name
        wall = SceneObject(UniqueID(oid), ObjectType.WALL, name,
            'Research-assumed fixed boundary, not original surveyed structure', RigidTransform(center),
            bbox_min=-size/2, bbox_max=size/2, immutable=True)
        walls.append(wall); normals[oid] = np.array(normal, dtype=float)
        bindings[oid] = {'asset_id': 'procedural_structure:' + name,
                         'identity_provenance': 'research_assumed_transfer_shell'}
        for label, kind, left, right, bottom, top in cuts:
            center = [0., 0., (bottom+top)/2]; center[axis] = (left+right)/2; center[1-axis] = fixed
            size = np.array([.015, .015, top-bottom]); size[axis] = right-left
            if kind == 'window':
                box(link, label, center, size, [.55, .75, .9, .22], collision=False)
            oid = 'opening_' + label
            objects.append(SceneObject(UniqueID(oid), ObjectType.WALL_MOUNTED, label,
                'Declared aperture; no door leaf, joint, track or operational validation',
                RigidTransform(center), bbox_min=-size/2, bbox_max=size/2, immutable=True,
                metadata={'structural_role': 'opening', 'door_operation': 'unknown'}))
            bindings[oid] = {'asset_id': 'procedural_opening:' + label,
                             'identity_provenance': 'research_assumed_transfer_shell'}
            start = center[:2].copy(); end = start.copy(); start[axis] = left; end[axis] = right
            side = name if name in ('south', 'north', 'east', 'west') else 'west'
            openings.append(ClearanceOpeningData(oid, kind, side, center, right-left,
                bottom, top-bottom, None, None, start, end, (right-left)/2))
    sdf = roomdir / 'room.sdf'; save_sdf(sdf, root)
    floor = SceneObject(UniqueID('shell_floor'), ObjectType.FLOOR, 'floor', 'Interior only; terrace is access context',
        RigidTransform(), immutable=True, support_surfaces=[SupportSurface(UniqueID('S_0'),
        np.array([-3., -2.5, 0.]), np.array([3., 2.5, 0.]), RigidTransform())])
    geometry = RoomGeometry(ET.parse(sdf), sdf, walls=walls, floor=floor, wall_normals=normals,
        length=6, width=5, wall_height=2.7, wall_thickness=.05, openings=openings)
    room = RoomScene(geometry, roomdir, room_id=rid, room_type='studio',
                     text_description='30m2 research transfer; fixed southeast washroom enclosure; all dimensions assumed.')
    for obj in walls + objects:
        room.add_object(obj)
    room._surface_id_counter = 1
    # Reuse only catalog entries and their mesh/SDF dependency directories.
    source = Path(plan['inputs']['catalog_manifest']['path']).parent
    old, _, _, _ = validate_bundle(source)
    oldroom = source / ('room_' + old['selected_room_id'])
    catalog = copy.deepcopy(json.loads((source / old['inputs']['asset_catalog']).read_text()))
    if len(catalog['entries']) != 16:
        raise ValueError('Unexpected fixed catalog')
    for entry in catalog['entries']:
        template = entry['template']
        if template['transform'] != {'translation': [0., 0., 0.], 'rotation_wxyz': [1., 0., 0., 0.]}:
            raise ValueError('Catalog contains a placed furniture pose')
        for key in ('sdf_path', 'geometry_path'):
            relative = Path(template[key]).parent
            shutil.copytree(oldroom/relative, roomdir/relative, dirs_exist_ok=True)
    layout = HouseLayout(house_dir=bundle, house_prompt=task['researcher_authored_brief'])
    layout.room_specs.append(RoomSpec(rid, room_type='studio', width=5, length=6))
    layout.placed_rooms.append(PlacedRoom(rid, (0, 0), 6, 5))
    layout.room_geometries[rid] = geometry
    house = HouseScene(layout); house.add_room(room)
    write(bundle/'house_state.json', house.to_state_dict())
    write(bundle/'runtime.json', {'rooms': {rid: {'room_type': 'studio', 'next_surface_id': 1, 'asset_bindings': bindings}}})
    write(bundle/'catalog.json', catalog)
    write(bundle/'task.json', {'design_request': task['researcher_authored_brief'], 'original_request': None,
        'request_provenance': 'researcher_transfer_assumptions', 'structured_requirements': task['required_activities'],
        'geometry_assumptions': g, 'unknowns': task['uncertainties']})
    write(bundle/'provenance.json', {'plan_sha256': sha256(args.plan), 'task_sha256': sha256(Path(plan['inputs']['task']['path'])),
        'source_final_layout_read': False, 'old_room_poses_or_brief_reused': False,
        'geometry_status': 'research assumptions, not survey', 'training_eligible': False,
        'catalog_role': 'Same preexisting unscaled geometry-only asset set; no furniture instances placed',
        'terrace': 'Extra contextual surface; no outdoor furniture placement or structural safety claim'})
    files = {f.relative_to(bundle).as_posix(): {'bytes': f.stat().st_size, 'sha256': sha256(f)}
             for f in sorted(bundle.rglob('*')) if f.is_file()}
    write(bundle/'manifest.json', {'schema': SCHEMA,
        'coordinates': {'units': 'metres', 'up': '+Z', 'frame': 'native_room_local'},
        'project': plan['project'], 'selected_room_id': rid,
        'inputs': {'native_state': 'house_state.json', 'runtime_metadata': 'runtime.json',
                   'task': 'task.json', 'asset_catalog': 'catalog.json'}, 'files': files})
    validate_bundle(bundle)
    restored = import_project(bundle, args.out/'restored')
    if set(restored.room.objects) != set(room.objects):
        raise ValueError('Native identities changed on restore')
    for oid, obj in room.objects.items():
        if not np.allclose(obj.transform.GetAsMatrix4(), restored.room.objects[oid].transform.GetAsMatrix4(), atol=1e-12, rtol=0):
            raise ValueError('Native pose changed on restore')
    size = sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file())
    if size > plan['budget']['bundle_max_bytes']:
        raise ValueError('Bundle exceeds frozen byte budget')
    result = {'status': 'initial_shell_catalog_built_and_restored', 'bundle': str(bundle),
        'manifest_sha256': sha256(bundle/'manifest.json'), 'bundle_bytes': size,
        'walls': len(walls), 'openings': len(openings), 'catalog_entries': 16, 'furniture_instances': 0,
        'model_calls': 0, 'training_updates': 0, 'renders': 0, 'formal_evaluation_ready': False,
        'limits': ['Door mechanics, privacy, installation and real functions unknown.',
                   'No furnished result, source recreation, human validation or model score.',
                   'Render acceptance and independent geometry checks pending.']}
    write(args.out/'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
