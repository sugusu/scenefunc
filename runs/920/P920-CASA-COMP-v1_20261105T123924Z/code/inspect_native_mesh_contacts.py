"""Inspect actual visual triangles in a frozen reconstructed native room.

Uses the same SDF visual transforms and glTF up-axis conversion as the recorded
Drake export. This is a surface-intersection diagnostic, not solid physics,
stability, door-sweep or activity acceptance. No scenes are modified.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import trimesh
from trimesh.collision import mesh_to_BVH
from trimesh.transformations import euler_matrix, quaternion_matrix

from native_project import sha256, validate_bundle


def sdf_visuals(path, root, cache):
    xml = ET.parse(path).getroot()
    if xml.find('.//model/pose') is not None or xml.find('.//link/pose') is not None:
        raise ValueError('Nonidentity model/link pose requires explicit handling')
    for node in xml.findall('.//link/visual'):
        geometry = node.find('geometry')
        box = geometry.find('box')
        if box is not None:
            mesh = trimesh.creation.box(extents=np.fromstring(box.findtext('size'), sep=' '))
        elif geometry.find('mesh') is not None:
            item = geometry.find('mesh')
            if item.find('scale') is not None:
                raise ValueError('Explicit mesh scale not supported by this frozen input contract')
            file = (path.parent/item.findtext('uri')).resolve()
            if not file.is_relative_to(root.resolve()) or file.suffix not in ('.gltf', '.glb'):
                raise ValueError('Unbounded or unsupported mesh resource')
            if file not in cache:
                scene = trimesh.load(file, force='scene', process=False)
                mesh = scene.to_mesh()
                # Drake converts glTF +Y up to native +Z up before SDF pose.
                rotation = np.eye(4)
                rotation[:3,:3] = [[1,0,0],[0,0,-1],[0,1,0]]
                mesh.apply_transform(rotation)
                cache[file] = mesh
            mesh = cache[file].copy()
        else:
            raise ValueError('Unsupported visual primitive')
        pose = np.fromstring(node.findtext('pose', '0 0 0 0 0 0'), sep=' ')
        if len(pose)!=6:
            raise ValueError('Unexpected SDF pose')
        transform = euler_matrix(*pose[3:])
        transform[:3,3] = pose[:3]
        mesh.apply_transform(transform)
        yield node.attrib['name'], mesh


def world_transform(obj):
    transform = quaternion_matrix(obj['transform']['rotation_wxyz'])
    transform[:3,3] = obj['transform']['translation']
    if obj['scale_factor']!=1:
        raise ValueError('Unexpected object scaling')
    return transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'begin actual native mesh surface inspection', flush=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    plan = json.loads(args.plan.read_text())
    root = Path(plan['bundle'])
    if sha256(root/'manifest.json')!=plan['manifest_sha256']:
        raise ValueError('Frozen bundle manifest changed')
    manifest, state, runtime, task = validate_bundle(root)
    rid = manifest['selected_room_id']
    native = state['rooms'][rid]
    room = root/('room_'+rid)
    meshes, cache, objects, kinds = {}, {}, {}, {}
    for oid, obj in native['objects'].items():
        if obj['sdf_path'] is None:
            continue
        parts = [mesh for _,mesh in sdf_visuals(room/obj['sdf_path'],root,cache)]
        mesh = trimesh.util.concatenate(parts)
        transform = world_transform(obj)
        mesh.apply_transform(transform)
        corners = np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))))
        world_corners = trimesh.transform_points(corners, transform)
        expected = np.array([world_corners.min(0),world_corners.max(0)])
        if not np.allclose(mesh.bounds,expected,atol=1e-6,rtol=0):
            raise ValueError('Actual visual bounds differ from native object: '+oid)
        meshes[oid]=mesh
        objects[oid]=obj
        kinds[oid]='furniture'
    if len(meshes)!=21:
        raise ValueError('Incomplete initial furniture inventory')
    structural_names = sorted((o['name'] for o in native['objects'].values()
        if o['sdf_path'] is None),key=len,reverse=True)
    structural = {}
    for name, mesh in sdf_visuals(room/native['room_geometry']['sdf_path'],root,cache):
        if name in ('floor','ceiling'):
            continue
        base = next((x for x in structural_names if name==x or name.startswith(x+'_')),None)
        if base is None:
            raise ValueError('Unknown structural visual '+name)
        structural.setdefault('recon_'+base,[]).append(mesh)
    for oid, parts in structural.items():
        meshes[oid]=trimesh.util.concatenate(parts)
        kinds[oid]='structure'
    fcl_objects = {oid:fcl.CollisionObject(mesh_to_BVH(mesh)) for oid,mesh in meshes.items()}
    pairs = []
    for left,right in itertools.combinations(sorted(meshes),2):
        if kinds[left]==kinds[right]=='structure':
            continue
        if any(oid in objects and objects[oid]['object_type']=='thin_covering' for oid in (left,right)):
            continue
        overlap = np.minimum(meshes[left].bounds[1],meshes[right].bounds[1])-np.maximum(meshes[left].bounds[0],meshes[right].bounds[0])
        if not np.all(overlap>1e-4):
            continue
        request = fcl.CollisionRequest(num_max_contacts=8,enable_contact=True)
        result = fcl.CollisionResult()
        fcl.collide(fcl_objects[left],fcl_objects[right],request,result)
        record={'object_ids':[left,right],'aabb_overlap_extents_m':overlap.tolist(),
            'pair_type':'furniture_structure' if 'structure' in (kinds[left],kinds[right]) else 'furniture_furniture',
            'triangle_surface_intersection_detected':bool(result.is_collision),
            'contact_samples':[{'position_m':c.pos.tolist(),'triangle_ids':[int(c.b1),int(c.b2)]} for c in result.contacts],
            'solid_containment_assessed':False,
            'contains_procedural_proxy':any(oid in objects and objects[oid]['metadata']['asset_source'] in ('procedural_proxy','mesh_with_reconstructed_base') for oid in (left,right))}
        pairs.append(record)
    support=[]
    for oid,obj in objects.items():
        placement=obj.get('placement_info')
        if not placement:
            continue
        parent='recon_'+obj['metadata']['support_parent']+'_0'
        child=meshes[oid]
        bottom=float(child.bounds[0,2])
        low,high=child.bounds[:,:2]
        points=[(low+high)/2,*[np.array(v) for v in itertools.product(*zip(low,high))]]
        distances=[]
        for point in points:
            origins=np.array([[*point,bottom+0.01]])
            locations,_,_=meshes[parent].ray.intersects_location(origins,[[0,0,-1]])
            heights=locations[:,2] if len(locations) else np.array([])
            heights=heights[heights<=bottom+1e-6]
            distances.append(None if not len(heights) else bottom-float(heights.max()))
        support.append({'child_id':oid,'parent_id':parent,'method':'five downward rays from child XY envelope center/corners',
                        'surface_gap_m':distances,'sampled_support_within_1mm':all(x is not None and abs(x)<=0.001 for x in distances),
                        'stability_assessed':False})
    result={'status':'actual_visual_mesh_surface_inspection_completed','plan_sha256':sha256(args.plan),
        'input_manifest_sha256':plan['manifest_sha256'],'furniture_objects':len(objects),
        'structural_groups':len(structural),'fcl_version':fcl.__version__,'trimesh_version':trimesh.__version__,
        'mesh_summary':{oid:{'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'bounds_m':mesh.bounds.tolist(),
                         'watertight_as_loaded':bool(mesh.is_watertight)} for oid,mesh in meshes.items()},
        'positive_aabb_pairs':pairs,'support_samples':support,'model_calls':0,'scene_edits':0,
        'whole_room_geometry_accepted':False,
        'limits':['Triangle intersection may include mounting/contact; no penetration depth or solid occupancy conclusion.',
                  'No surface intersection does not rule out containment; non-watertight and proxy geometry remain explicit.',
                  'Floor/ceiling contacts and thin coverings excluded; no door sweep, stability, reach or activity metric.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':result['status'],'positive_aabb_pairs':len(pairs),
        'surface_intersection_pairs':sum(p['triangle_surface_intersection_detected'] for p in pairs),
        'support_samples':support}),flush=True)


if __name__=='__main__':
    main()
