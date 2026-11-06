"""Read-only whole-room diagnostics under the prospectively frozen assumptions.

Object presence, conservative routes and triangle contacts are separate outputs.
No generated room receives an inferred professional or physical-validity label.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import fcl
import numpy as np
import trimesh
from trimesh.collision import mesh_to_BVH

from inspect_native_mesh_contacts import sdf_visuals, world_transform
from measure_room_access import measure
from measure_free_floor import largest_floor_rectangle
from native_project import sha256, validate_bundle


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());root=a.bundle.resolve()
    for name,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Audit code changed')
    manifest,state,runtime,task=validate_bundle(root);rid=manifest['selected_room_id'];native=state['rooms'][rid]
    room=root/('room_'+rid);initial=validate_bundle(Path(plan['initial_bundle']))[1]['rooms'][rid]
    catalog=json.loads((root/'catalog.json').read_text());byasset={x['binding']['asset_id']:x['template']['name'] for x in catalog['entries']}
    bindings=runtime['rooms'][rid]['asset_bindings'];meshes={};kinds={};cache={};inventory=[];targets={};bounds=[]
    for oid,obj in native['objects'].items():
        if obj['object_type']!='furniture':continue
        transform=world_transform(obj);mesh=trimesh.util.concatenate([m for _,m in sdf_visuals(room/obj['sdf_path'],root,cache)])
        mesh.apply_transform(transform);meshes[oid]=mesh;kinds[oid]='furniture'
        name=byasset[bindings[oid]['asset_id']]
        inventory.append({'object_id':oid,'name':name,'asset_id':bindings[oid]['asset_id'],'present':True,'functional_use':'unverified'})
        lo=np.array(obj['bbox_min']);hi=np.array(obj['bbox_max']);center=(lo+hi)/2
        points={'front':np.array([center[0],hi[1]+plan['approach_offset_m'],0])}
        if 'bed' in name:
            points={'left_side':np.array([lo[0]-plan['approach_offset_m'],center[1],0]),
                    'right_side':np.array([hi[0]+plan['approach_offset_m'],center[1],0])}
        for label,point in points.items():targets[oid+':'+label]=trimesh.transform_points([point],transform)[0,:2].tolist()
        geom=native['room_geometry'];actual=mesh.bounds
        bounds.append({'object_id':oid,'actual_visual_bounds_m':actual.tolist(),
            'outside_nominal_room_xy':bool(actual[0,0]<-geom['length']/2-1e-6 or actual[1,0]>geom['length']/2+1e-6 or
                actual[0,1]<-geom['width']/2-1e-6 or actual[1,1]>geom['width']/2+1e-6),
            'bottom_z_m':float(actual[0,2]),'top_z_m':float(actual[1,2]),'watertight':bool(mesh.is_watertight)})
    for name,mesh in sdf_visuals(room/native['room_geometry']['sdf_path'],root,cache):
        if name in ('floor','ceiling'):continue
        oid='structure:'+name;meshes[oid]=mesh;kinds[oid]='structure'
    collision={oid:fcl.CollisionObject(mesh_to_BVH(m)) for oid,m in meshes.items()};pairs=[]
    for left,right in itertools.combinations(sorted(meshes),2):
        if kinds[left]==kinds[right]=='structure':continue
        overlap=np.minimum(meshes[left].bounds[1],meshes[right].bounds[1])-np.maximum(meshes[left].bounds[0],meshes[right].bounds[0])
        if not np.all(overlap>plan['intersection_overlap_threshold_m']):continue
        output=fcl.CollisionResult();fcl.collide(collision[left],collision[right],fcl.CollisionRequest(num_max_contacts=8,enable_contact=True),output)
        pairs.append({'objects':[left,right],'triangle_surface_intersection_detected':bool(output.is_collision),
            'aabb_overlap_extents_m':overlap.tolist(),'contact_samples':[{'position_m':c.pos.tolist(),'triangle_ids':[int(c.b1),int(c.b2)]} for c in output.contacts],
            'solid_containment_assessed':False})
    settings={**plan['access_settings'],'approach_points_xy_m':targets}
    access=measure(native,room,root,settings)
    access['door_assumption']='Declared doorway assumed fully open; no hinge/sweep acceptance.'
    access['limits'][1]='Declared grid and square-body scenarios; not a human model or accessibility standard.'
    immutable={oid:native['objects'].get(oid)==obj for oid,obj in initial['objects'].items() if obj['immutable']}
    # Resource paths are relative in both portable snapshots. Changes here are
    # assessed on complete structural records, not just rendered silhouettes.
    result={'status':'whole_room_diagnostics_completed','evaluated_at':datetime.now(timezone.utc).isoformat(),
        'audit_plan_sha256':sha256(a.plan),'input_manifest_sha256':sha256(root/'manifest.json'),
        'inventory':inventory,'immutable_records_unchanged':immutable,'bounds':bounds,
        'positive_aabb_pairs':pairs,'surface_intersection_pairs':sum(x['triangle_surface_intersection_detected'] for x in pairs),
        'access':access,'free_floor':largest_floor_rectangle(native,room,root,settings),
        'whole_room_success':'unknown_not_accepted','professional_review':False,'scene_edits':0,'model_calls':0,
        'limits':['Presence does not establish use; static doors/lids/folds/services not simulated.',
            'Surface intersection is not penetration depth; absence does not rule out containment or instability.',
            'Conservative AABB routes may block gaps; endpoints are deterministic hypotheses, not measured resident needs.',
            'Sanitary enclosure, privacy, installation, comfort and aesthetics remain unknown.',
            'Development source-informed shell, no held-out generalization or learning result.']}
    from edit_workspace_metrics import evaluate_workspace
    result['activity']=evaluate_workspace(plan,root,native,initial,meshes,inventory,settings)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'furniture':len(inventory),'surface_intersection_pairs':result['surface_intersection_pairs'],
        'outside_room':sum(x['outside_nominal_room_xy'] for x in bounds),'immutable_records_unchanged':all(immutable.values())}),flush=True)


if __name__=='__main__':main()
