"""Build a labelled native whole-room reconstruction from an initial reference.

This is a development input, not a recovered historical design or a supervised
professional action. No final-layout file is read by this builder. Every mesh
substitution, procedural proxy and support correction remains in provenance.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from native_project import SCHEMA, import_project, sha256, validate_bundle


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def add(parent, tag, value=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if value is not None:
        node.text = str(value)
    return node


def numbers(values):
    return ' '.join(format(float(v), '.12g') for v in values)


def box(link, name, center, size, color, collision=True, visual=True):
    for role in (['visual'] if visual else []) + (['collision'] if collision else []):
        node = add(link, role, name=name)
        add(node, 'pose', numbers([*center, 0, 0, 0]))
        add(add(add(node, 'geometry'), 'box'), 'size', numbers(size))
        if role == 'visual':
            material = add(node, 'material')
            add(material, 'diffuse', numbers(color))
            add(material, 'ambient', numbers(color))


def rgba(text):
    return [int(text[k:k+2], 16) / 255 for k in (1, 3, 5)] + [1.0]


def sdf_root(name, link_name='body'):
    root = ET.Element('sdf', version='1.7')
    return root, add(add(root, 'model', name=name), 'link', name=link_name)


def save_sdf(path, root):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def source_position(record):
    x, up, south = record['location']
    return np.array([x, -south, up], dtype=float)


def prepare_mesh(spec, directory):
    """Preserve textures and size; record an explicit raw→native rotation."""
    path = Path(spec['source_path'])
    if sha256(path) != spec['sha256']:
        raise ValueError('Source asset changed: ' + spec['asset_id'])
    scene = trimesh.load(path, force='scene', process=False)
    rotation = np.asarray(spec['source_to_native_rotation'], dtype=float)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError('Asset transform must be a rigid proper rotation')
    raw_to_native = np.eye(4)
    raw_to_native[:3, :3] = rotation
    canonical = scene.copy()
    canonical.apply_transform(raw_to_native)
    bounds = canonical.bounds
    shift = np.array([-bounds[:, 0].mean(), -bounds[:, 1].mean(), -bounds[0, 2]])
    canonical.apply_translation(shift)
    directory.mkdir(parents=True)
    canonical_path = directory / 'canonical.glb'
    canonical.export(canonical_path)
    # Drake converts glTF +Y-up into +Z-up. Keep source glTF in its original
    # frame, then put only the remaining rigid transform on the SDF visual.
    exported = trimesh.exchange.gltf.export_gltf(scene, merge_buffers=True)
    for name, raw in exported.items():
        dest = directory / name
        if not dest.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Exported glTF resource escapes its asset directory')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    gltf_path = directory / 'model.gltf'
    if not gltf_path.is_file():
        raise ValueError('Missing exported glTF')
    # Explicit resource closure, including texture files and mesh buffers.
    gltf = json.loads(gltf_path.read_text())
    for key in ('images', 'buffers'):
        for item in gltf.get(key, []):
            uri = item.get('uri')
            if uri and not uri.startswith('data:'):
                target = (directory / uri).resolve()
                if not target.is_relative_to(directory.resolve()) or not target.is_file():
                    raise ValueError('Unresolved glTF dependency')
    yup_to_zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    return canonical_path, canonical.bounds, gltf_path, rotation @ yup_to_zup.T, shift


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--scenesmith', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'begin labelled native whole-room reconstruction', flush=True)
    sys.path.insert(0, str(a.scenesmith.resolve()))
    from pydrake.all import RigidTransform, RotationMatrix, RollPitchYaw
    from scenesmith.agent_utils.house import (HouseLayout, HouseScene, RoomGeometry,
        RoomSpec, PlacedRoom, ClearanceOpeningData)
    from scenesmith.agent_utils.room import (RoomScene, SceneObject, ObjectType, UniqueID,
        SupportSurface, PlacementInfo)
    plan = json.loads(a.plan.read_text())
    source = Path(plan['initial_reference']['path'])
    if sha256(source) != plan['initial_reference']['sha256']:
        raise ValueError('Initial reference changed')
    initial = json.loads(source.read_text())
    if set(plan['objects']) != {v['name'] for v in initial['furniture']}:
        raise ValueError('Every initial furniture record needs an explicit mapping')
    a.out.mkdir(parents=True, exist_ok=False)
    bundle = a.out / 'bundle'
    bundle.mkdir()
    rid = 'dancer_reconstruction'
    directory = bundle / ('room_' + rid)
    directory.mkdir()
    structure = {r['name']: r for r in initial['room_structure']}
    floor_source = structure['floor']
    length, _, width = floor_source['dimensions']
    height = structure['ceiling']['location'][1]
    shell_xml, shell_link = sdf_root('reconstructed_shell', 'room_geometry_body_link')
    box(shell_link, 'floor', [0, 0, -0.02], [length, width, 0.04], rgba(floor_source['color']))
    # A physical ceiling remains in the SDF, while the observation is an
    # explicit roof cutaway. Ceiling-mounted instances are not removed.
    box(shell_link, 'ceiling', [0, 0, height+0.02], [length, width, 0.04], [1, 1, 1, 1], visual=False)
    shell = directory / 'room.sdf'
    walls, normals, bindings, openings, structural_objects = [], {}, {}, [], []
    opening_rows = [r for r in structure.values() if 'door' in r['name'] or 'window' in r['name']]
    wall_rows = [r for r in structure.values() if 'wall' in r['name']]
    for record in wall_rows:
        name = record['name']
        center = source_position(record)
        wall_width, wall_height, thickness = record['dimensions']
        horizontal = record['facing'] in (0, 180)
        along = 0 if horizontal else 1
        normal_axis = 1-along
        cuts = []
        for item in opening_rows:
            position = source_position(item)
            w, h, _ = item['dimensions']
            if (abs(position[normal_axis]-center[normal_axis]) <= 0.04
                    and abs(position[along]-center[along])+w/2 <= wall_width/2+1e-6):
                cuts.append((position[along]-w/2, position[along]+w/2, position[2]-h/2, position[2]+h/2))
        low, high = center[along]-wall_width/2, center[along]+wall_width/2
        xcuts = sorted({low, high, *(v for c in cuts for v in c[:2])})
        zcuts = sorted({0.0, wall_height, *(v for c in cuts for v in c[2:])})
        index = 0
        for left, right in zip(xcuts, xcuts[1:]):
            for bottom, top in zip(zcuts, zcuts[1:]):
                mid_u, mid_z = (left+right)/2, (bottom+top)/2
                if any(c[0] < mid_u < c[1] and c[2] < mid_z < c[3] for c in cuts):
                    continue
                xyz = center.copy(); xyz[along] = mid_u; xyz[2] = mid_z
                size = [thickness, thickness, top-bottom]; size[along] = right-left
                box(shell_link, name+'_'+str(index), xyz, size, rgba(record['color']))
                index += 1
        size = np.array([wall_width, thickness, wall_height] if horizontal else [thickness, wall_width, wall_height])
        wall = SceneObject(UniqueID('recon_'+name), ObjectType.WALL, name, 'Generated-reference structural proxy',
                           RigidTransform(center), bbox_min=-size/2, bbox_max=size/2, immutable=True)
        walls.append(wall)
        angle = math.radians(-record['facing'])
        normals[str(wall.object_id)] = np.array([-math.sin(angle), math.cos(angle)])
        bindings[str(wall.object_id)] = {'asset_id':'procedural_structure:'+name,
                                       'identity_provenance':'researcher_reconstruction_not_historical'}
    for record in opening_rows:
        position = source_position(record)
        w, h, depth = record['dimensions']
        facing = record['facing']
        horizontal = facing in (0, 180)
        size = np.array([w, depth, h] if horizontal else [depth, w, h])
        color = rgba(record['color'])
        if 'window' in record['name']:
            color[3] = 0.22
        box(shell_link, record['name'], position, size, color, collision='door' in record['name'])
        oid = 'recon_'+record['name']
        structural_objects.append(SceneObject(UniqueID(oid), ObjectType.WALL_MOUNTED,
            record['name'], 'Generated proxy opening; hinge and operability unknown', RigidTransform(position),
            bbox_min=-size/2, bbox_max=size/2, immutable=True, metadata={'structural_role':'opening'}))
        bindings[oid] = {'asset_id':'procedural_opening:'+record['name'],
                        'identity_provenance':'researcher_reconstruction_not_historical'}
        direction = {180:'north', 0:'south', 270:'east', 90:'west'}[facing]
        axis = 0 if horizontal else 1
        start = [position[0], position[1]]; end = list(start)
        start[axis] -= w/2; end[axis] += w/2
        openings.append(ClearanceOpeningData(record['name'], 'door' if 'door' in record['name'] else 'window',
            direction, position.tolist(), w, position[2]-h/2, h, None, None, start, end, w/2))
    save_sdf(shell, shell_xml)
    floor_surface = SupportSurface(UniqueID('S_0'), np.array([-length/2,-width/2,0]),
        np.array([length/2,width/2,0]), RigidTransform())
    floor = SceneObject(UniqueID('recon_floor'), ObjectType.FLOOR, 'floor', 'Generated floor proxy',
        RigidTransform(), immutable=True, support_surfaces=[floor_surface])
    geometry = RoomGeometry(ET.parse(shell), shell, walls=walls, floor=floor, wall_normals=normals,
        width=width, length=length, wall_height=height, wall_thickness=0.02, openings=openings)
    room = RoomScene(geometry, directory, room_id=rid, room_type='studio',
        text_description='Researcher reconstruction from a generated development reference; not a surveyed or historical room.')
    for obj in walls+structural_objects:
        room.add_object(obj)
    assets = {}
    provenance = []
    surface_counter = 1
    # Support-dependent objects are constructed after their parent.
    records = sorted(initial['furniture'], key=lambda r: bool(plan['objects'][r['name']].get('parent')))
    for record in records:
        name = record['name']; spec = plan['objects'][name]; oid = 'recon_'+name+'_0'
        object_dir = directory / 'objects' / name
        object_dir.mkdir(parents=True)
        root, link = sdf_root(name)
        visual_path = None
        if spec['kind'] in ('mesh', 'mesh_with_reconstructed_base'):
            asset_spec = plan['assets'][spec['asset_id']]
            if spec['asset_id'] not in assets:
                assets[spec['asset_id']] = prepare_mesh(asset_spec, directory/'assets'/spec['asset_id'])
            visual_path, bounds, gltf, rotation, shift = assets[spec['asset_id']]
            bounds = bounds.copy()
            base_height = float(spec.get('reconstructed_base_height', 0))
            shift = shift.copy(); shift[2] += base_height
            node = add(link, 'visual', name='source_asset')
            rpy = RollPitchYaw(RotationMatrix(rotation)).vector()
            add(node, 'pose', numbers([*shift, *rpy]))
            add(add(add(node, 'geometry'), 'mesh'), 'uri', os.path.relpath(gltf, object_dir))
            if base_height:
                base_size = np.array(spec['reconstructed_base_width_depth']+[base_height])
                box(link, 'declared_proxy_base', [0,0,base_height/2], base_size, [0.35,0.35,0.35,1])
                bounds[1,2] += base_height
                bounds[0,:2] = np.minimum(bounds[0,:2], -base_size[:2]/2)
                bounds[1,:2] = np.maximum(bounds[1,:2], base_size[:2]/2)
            # Conservative declared envelope; not mesh-level collision truth.
            box(link, 'collision_envelope', bounds.mean(axis=0), bounds[1]-bounds[0], [1,1,1,1], visual=False)
            asset_id = spec['asset_id']
        else:
            w, h, d = record['dimensions']
            bounds = np.array([[-w/2,-d/2,0],[w/2,d/2,h]])
            box(link, 'declared_proxy', [0,0,h/2], [w,d,h], rgba(record['color']),
                collision=spec['object_type'] != 'thin_covering')
            asset_id = 'procedural_reference:'+name
        sdf = object_dir/'object.sdf'; save_sdf(sdf, root)
        center = source_position(record)
        position = np.array([center[0], center[1], float(spec.get('base_z', 0))])
        if spec.get('mount') == 'ceiling':
            position[2] = height-bounds[1,2]
        elif spec.get('mount') == 'wall':
            position[2] = center[2]-(bounds[1,2]-bounds[0,2])/2
        yaw = math.radians(-record.get('facing', 0))
        placement = None
        if spec.get('parent'):
            parent = room.objects['recon_'+spec['parent']+'_0']
            surface = parent.support_surfaces[0]
            position[2] = surface.transform.translation()[2]
            local = surface.transform.inverse().multiply(position)
            placement = PlacementInfo(surface.surface_id, local[:2], yaw, 'researcher_reconstructed_support')
        transform = RigidTransform(RollPitchYaw(0,0,yaw), position)
        metadata = {'asset_source': 'thin_covering' if spec['object_type']=='thin_covering' else spec['kind'],
                    'source_reference_name': name, 'source_geometry_is_generated': True,
                    'semantic_front_status': spec.get('front_status','not_applicable_or_unknown'),
                    'collision_representation':'declared_AABB_envelope_not_mesh_validation',
                    'physical_mass_known':False, 'historical_instance_identity':False,
                    'support_parent':spec.get('parent') or spec.get('mount') or 'floor'}
        obj = SceneObject(UniqueID(oid), ObjectType(spec['object_type']), name,
            'Labelled reconstruction: '+spec['kind'], transform, geometry_path=visual_path, sdf_path=sdf,
            bbox_min=bounds[0], bbox_max=bounds[1], placement_info=placement,
            immutable=spec.get('immutable',False), metadata=metadata)
        if spec.get('support_top'):
            world_top = position.copy(); world_top[2] += bounds[1,2]
            obj.support_surfaces = [SupportSurface(UniqueID('S_'+np.base_repr(surface_counter,36).lower()),
                np.array([bounds[0,0],bounds[0,1],0]), np.array([bounds[1,0],bounds[1,1],0]),
                RigidTransform(RollPitchYaw(0,0,yaw),world_top))]
            surface_counter += 1
        room.add_object(obj)
        bindings[oid] = {'asset_id':asset_id,'identity_provenance':'researcher_reconstruction_not_historical',
                        'representation':spec['kind']}
        provenance.append({'object_id':oid,'reference_name':name,'asset_id':asset_id,'kind':spec['kind'],
            'source_dimensions_width_height_depth_m':record['dimensions'],
            'actual_bounds_min_max_m':bounds.tolist(),'actual_position_m':position.tolist(),
            'source_position_x_up_south_m':record['location'],'rescaled':False,
            'support_parent':metadata['support_parent'],'semantic_front_status':metadata['semantic_front_status'],
            'reconstruction_note':spec['note']})
    room._surface_id_counter = surface_counter
    layout = HouseLayout(house_dir=bundle, house_prompt=plan['task']['design_request'])
    layout.room_specs.append(RoomSpec(rid, room_type='studio', width=width, length=length))
    layout.placed_rooms.append(PlacedRoom(rid,(0,0),length,width))
    layout.room_geometries[rid] = geometry
    house = HouseScene(layout); house.add_room(room)
    write(bundle/'house_state.json',house.to_state_dict())
    write(bundle/'runtime.json',{'rooms':{rid:{'room_type':'studio','next_surface_id':surface_counter,'asset_bindings':bindings}}})
    write(bundle/'task.json',plan['task'])
    write(bundle/'provenance.json',{'plan_sha256':sha256(a.plan),'reference_sha256':sha256(source),
        'objects':provenance,'ceiling':'collision-only structural ceiling; roof cutaway for views',
        'opening_hinges_and_operability':'unknown; proxy leaves retained as supplied',
        'training_eligible':False})
    files = {p.relative_to(bundle).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)}
             for p in sorted(bundle.rglob('*')) if p.is_file()}
    write(bundle/'manifest.json',{'schema':SCHEMA,
        'coordinates':{'units':'metres','up':'+Z','frame':'native_room_local'},
        'project':{'project_id':'nyc_chan_dancer_studio_reconstruction_v1','family_id':'nyc_chan_dancer_studio',
                   'provenance':'researcher_reconstruction_from_generated_development_reference_not_historical'},
        'selected_room_id':rid,'inputs':{'native_state':'house_state.json','runtime_metadata':'runtime.json','task':'task.json'},'files':files})
    validate_bundle(bundle)
    restored = import_project(bundle, a.out/'restored')
    if set(restored.room.objects) != set(room.objects) or restored.room._surface_id_counter != surface_counter:
        raise ValueError('Native inventory/counter did not survive restoration')
    for oid, obj in room.objects.items():
        other = restored.room.objects[oid]
        if not np.allclose(obj.transform.GetAsMatrix4(),other.transform.GetAsMatrix4(),rtol=0,atol=1e-12):
            raise ValueError('Native pose did not survive restoration: '+str(oid))
    result = {'status':'native_initial_bundle_built_and_restored','input_furniture':len(records),
        'native_objects':len(room.objects),'structural_walls':len(walls),'openings':len(openings),
        'mesh_backed_objects':sum(v['kind']!='procedural_proxy' for v in provenance),
        'procedural_furniture_proxies':sum(v['kind']=='procedural_proxy' for v in provenance),
        'manifest_sha256':sha256(bundle/'manifest.json'),'bundle':str(bundle),
        'source_final_layout_read':False,'professional_sample_admitted':False,'model_calls':0,
        'training_updates':0,'scene_edits_by_policy':0,'render_acceptance':'not_run',
        'geometry_quality':'not_validated; substitutions may introduce conflicts',
        'limits':['Development reconstruction, not surveyed dimensions or the historical design.',
                  'Collision envelopes and missing mass do not establish physical feasibility.',
                  'Procedural proxies retain unresolved functions explicitly; no whole-room success claim.']}
    write(a.out/'RESULT.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
