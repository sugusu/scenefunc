"""Export all recorded room visuals to an inspectable GLB with object-ID nodes.

This is a geometry preview. The native checkpoint remains the editable source;
materials, articulation and physical functioning are not newly validated here.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from inspect_native_mesh_contacts import sdf_visuals,world_transform
from native_project import sha256,validate_bundle


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    for name,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Export code changed')
    root=Path(plan['bundle'])
    if sha256(root/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Source manifest changed')
    manifest,state,runtime,task=validate_bundle(root);rid=manifest['selected_room_id'];native=state['rooms'][rid]
    room=root/('room_'+rid);cache={};scene=trimesh.Scene();records={}
    # glTF +Y-up convention: native Z maps to +Y, native Y maps to -Z.
    up=np.eye(4);up[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    def add(node,mesh,transform,metadata):
        actual=mesh.copy();actual.apply_transform(up@transform)
        actual.metadata.update(metadata)
        scene.add_geometry(actual,node_name=node,geom_name=node)
        records[node]={'bounds_y_up_m':actual.bounds.tolist(),'triangles':len(actual.faces),**metadata}
    bindings=runtime['rooms'][rid]['asset_bindings']
    included_types=set(plan.get('include_object_types',['furniture']))
    for oid,obj in native['objects'].items():
        if obj['object_type'] not in included_types or obj['sdf_path'] is None:continue
        for index,(visual,mesh) in enumerate(sdf_visuals(room/obj['sdf_path'],root,cache)):
            add(oid+'::'+str(index),mesh,world_transform(obj),{'object_id':oid,'asset_id':bindings[oid]['asset_id'],
                'kind':obj['object_type'],'source_visual':visual,'immutable':obj['immutable']})
    for index,(name,mesh) in enumerate(sdf_visuals(room/native['room_geometry']['sdf_path'],root,cache)):
        add('structure::'+name+'::'+str(index),mesh,np.eye(4),{'kind':'structure','source_visual':name})
    scene.metadata.update(source_manifest_sha256=plan['manifest_sha256'],units='metre',up_axis='+Y',
        scope='Recorded geometry preview; editable native state and source asset IDs remain authoritative.')
    output=a.out/'room_geometry.glb';scene.export(output,file_type='glb')
    if output.stat().st_size>plan['budget']['output_max_bytes']:raise ValueError('Export size budget exceeded')
    restored=trimesh.load(output,force='scene',process=False)
    if set(restored.graph.nodes_geometry)!=set(records):raise ValueError('Export lost geometry nodes')
    errors={}
    for node,expected in records.items():
        transform,geometry=restored.graph[node];mesh=restored.geometry[geometry].copy();mesh.apply_transform(transform)
        if len(mesh.faces)!=expected['triangles']:raise ValueError('Export changed triangle count')
        error=float(np.max(np.abs(mesh.bounds-np.array(expected['bounds_y_up_m']))));errors[node]=error
        if error>1e-5:raise ValueError('Export geometry bounds drift exceeded float32 tolerance')
    actual_ids={v['object_id'] for v in records.values() if v['kind']=='furniture'}
    expected_ids={k for k,v in native['objects'].items() if v['object_type']=='furniture'}
    if actual_ids!=expected_ids:raise ValueError('Incomplete furniture export')
    visual_ids={v['object_id'] for v in records.values() if 'object_id' in v}
    expected_visual_ids={k for k,v in native['objects'].items()
                         if v['object_type'] in included_types and v['sdf_path'] is not None}
    if visual_ids!=expected_visual_ids:raise ValueError('Incomplete selected object export')
    result={'status':'geometry_export_roundtrip_checked','recorded_at':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':sha256(a.plan),'input_manifest_sha256':plan['manifest_sha256'],
        'glb':{'path':str(output),'bytes':output.stat().st_size,'sha256':sha256(output)},
        'furniture_object_ids':sorted(actual_ids),'geometry_nodes':len(records),'nodes':records,
        'all_visual_object_ids':sorted(visual_ids),'included_object_types':sorted(included_types),
        'maximum_roundtrip_bound_error_m':max(errors.values(),default=0),'native_scene_edits':0,'model_calls':0,
        'limits':['Geometry preview only; no new texture/material fidelity, articulation or physical-use acceptance.',
            'Whole shell visuals include ceiling; a 3D viewer may need to hide ceiling/wall nodes to inspect interior.',
            'GLB uses float32 geometry; complete native checkpoint preserves original identity, precision and action history.']}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','geometry_nodes','maximum_roundtrip_bound_error_m','glb']}),flush=True)


if __name__=='__main__':main()
